#!/usr/bin/env python3
"""Tests for baseline metric/alignment helpers (synthetic inputs only)."""
from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_baseline", ROOT / "scripts" / "train_baseline.py")
tb = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tb
SPEC.loader.exec_module(tb)


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        y = np.array([0.0, 0.5, 1.0])
        result = tb.metrics(y, y.copy())
        self.assertAlmostEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["rmse"], 0.0)

    def test_brier_and_constant_baseline(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        scores = np.array([-2.0, -2.0, 2.0, 2.0])
        probability = 1 / (1 + np.exp(-scores))
        result = tb.metrics(y, y.copy(), ("rule", 0.5), probability)
        expected = float(((probability - y) ** 2).mean())
        self.assertAlmostEqual(result["brier"], expected, places=8)
        self.assertAlmostEqual(result["brier_constant"], 0.25, places=8)
        self.assertAlmostEqual(result["auc"], 1.0, places=8)

    def test_threshold_is_strict(self):
        y = np.array([0.05, 0.050001])
        result = tb.metrics(y, y.copy(), ("rule", 0.05), np.full(2, 0.5))
        self.assertAlmostEqual(result["unsafe_rate"], 0.5)

    def test_logit_is_stable_at_extremes(self):
        self.assertTrue(np.isfinite(tb.logit(np.array([0.0, 1.0]))).all())


class TestRiskCoverage(unittest.TestCase):
    def test_ordering_accepts_lowest_risk(self):
        # Risk ranking (1, 3, 2, 0) matches the labels: the two accepted tiles are safe.
        y = np.array([0.9, 0.0, 0.9, 0.0])
        risk = np.array([0.9, 0.1, 0.8, 0.2])
        rows = tb.risk_coverage(y, risk, "g10_gt_0.05", coverages=(0.5, 1.0))
        half, full = rows
        self.assertEqual(half["accepted_n"], 2)
        self.assertEqual(half["unsafe_accepted_n"], 0)
        self.assertEqual(full["accepted_n"], 4)
        self.assertEqual(full["unsafe_total"], 2)
        self.assertAlmostEqual(full["UAR"], 0.5)
        self.assertAlmostEqual(full["HMR"], 1.0)

    def test_uar_uses_actual_accepted_count(self):
        y = np.array([0.0, 0.9])
        rows = tb.risk_coverage(y, np.array([0.1, 0.9]), "g10_gt_0.05", coverages=(0.5,))
        self.assertEqual(rows[0]["accepted_n"], 1)
        self.assertEqual(rows[0]["unsafe_accepted_n"], 0)
        self.assertEqual(rows[0]["UAR"], 0.0)


class TestBootstrap(unittest.TestCase):
    def test_grouped_bootstrap_resamples_groups(self):
        y = np.array([1.0, 2.0, 3.0])
        groups = np.array(["a", "b", "c"])

        def statistic(values, index):
            return float(values.mean())

        result = tb.grouped_bootstrap(y, groups, statistic, 200, seed=0)
        self.assertEqual(result["n_groups"], 3)
        self.assertLessEqual(result["lo95"], result["hi95"])
        self.assertTrue(1.0 <= result["lo95"] <= 3.0)

    def test_single_group_has_zero_width(self):
        y = np.array([1.0, 3.0])
        groups = np.array(["a", "a"])
        result = tb.grouped_bootstrap(y, groups, lambda v, i: float(v.mean()), 50, seed=0)
        self.assertAlmostEqual(result["lo95"], 2.0)
        self.assertAlmostEqual(result["hi95"], 2.0)


class TestNormalize(unittest.TestCase):
    def test_robust_normalize_range(self):
        array = np.linspace(0, 1, 1000).reshape(100, 10)
        normalized = tb.robust_normalize(array)
        self.assertLessEqual(float(normalized.max()), 1.5)
        self.assertGreaterEqual(float(normalized.min()), 0.0)

    def test_constant_array_returns_zeros(self):
        normalized = tb.robust_normalize(np.full((4, 4), 3.0))
        self.assertTrue(np.allclose(normalized, 0.0))

    def test_nan_pixels_become_zero_and_keep_scale(self):
        array = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float64)
        array[0, 0] = np.nan
        normalized = tb.robust_normalize(array)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertEqual(float(normalized[0, 0]), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.5)

    def test_all_nan_array_returns_zeros(self):
        normalized = tb.robust_normalize(np.full((4, 4), np.nan))
        self.assertTrue(np.allclose(normalized, 0.0))


class TestAlignmentMath(unittest.TestCase):
    def test_map_coordinates_uses_coordinates_not_indices(self):
        """A half-pixel origin shift must move values, not be ignored."""
        from scipy.ndimage import map_coordinates
        nac_values = np.arange(16, dtype=np.float64).reshape(4, 4)
        nac_x = np.array([0.0, 1.0, 2.0, 3.0])
        slope_x = nac_x + 0.5  # half-pixel offset
        cols = (slope_x - nac_x[0]) / 1.0
        rows = cols.copy()
        fine, coarse = np.meshgrid(rows, cols, indexing="ij")
        shifted = map_coordinates(nac_values, [fine, coarse], order=1, mode="nearest")
        self.assertFalse(np.allclose(shifted, nac_values))
        self.assertAlmostEqual(float(shifted[0, 0]), 2.5, places=6)

    def test_platt_scale_recovers_monotone_mapping(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=2000)
        probability = 1 / (1 + np.exp(-scores))
        targets = (rng.random(2000) < probability).astype(int)
        slope, intercept, degenerate = tb.platt_scale(scores, targets)
        self.assertFalse(degenerate)
        self.assertGreater(slope, 0.5)
        self.assertLess(slope, 1.5)
        self.assertTrue(math.isfinite(intercept))

    def test_platt_scale_single_class_is_identity(self):
        slope, intercept, degenerate = tb.platt_scale(np.array([0.1, 0.2]), np.array([1, 1]))
        self.assertTrue(degenerate)
        self.assertEqual((slope, intercept), (1.0, 0.0))

    def test_fit_logistic_single_class_returns_none(self):
        model, degenerate = tb.fit_logistic(np.zeros((3, 2)), np.array([1, 1, 1]))
        self.assertIsNone(model)
        self.assertTrue(degenerate)
        fallback = tb.logistic_probability(model, np.zeros((2, 2)), 0.7)
        self.assertTrue(np.allclose(fallback, 0.7))


class TestEvaluateModel(unittest.TestCase):
    def test_each_rule_uses_its_own_regression_column(self):
        # Column 0 tracks g10, column 1 is deliberately unrelated.
        y_g10 = np.array([0.0, 0.5, 1.0])
        fractions = np.column_stack([y_g10, np.array([0.9, 0.1, 0.05])])
        payload = tb.evaluate_model(y_g10, fractions, np.zeros(3), "g10_gt_0.05")
        self.assertAlmostEqual(payload["metrics"]["mae"], 0.0, places=8)
        self.assertAlmostEqual(payload["metrics"]["rmse"], 0.0, places=8)

        y_g15 = np.array([1.0, 0.0, 0.25])
        fractions = np.column_stack([np.array([0.1, 0.2, 0.3]), y_g15])
        payload = tb.evaluate_model(y_g15, fractions, np.zeros(3), "g15_gt_0.02")
        self.assertAlmostEqual(payload["metrics"]["mae"], 0.0, places=8)

    def test_rule_column_index_matches_rules(self):
        self.assertEqual(tb.rule_column_index("g10_gt_0.05"), 0)
        self.assertEqual(tb.rule_column_index("g15_gt_0.02"), 1)


class TestChannelsAndCache(unittest.TestCase):
    def test_batch_tensor_selects_channels(self):
        import torch
        cache = {
            "a": np.stack([np.ones((4, 4)), np.zeros((4, 4))]).astype(np.float16),
            "b": np.stack([np.full((4, 4), 2.0), np.full((4, 4), 3.0)]).astype(np.float16),
        }
        device = torch.device("cpu")
        nac = tb.batch_tensor(["a", "b"], cache, device, (0,))
        self.assertEqual(tuple(nac.shape), (2, 1, 4, 4))
        self.assertAlmostEqual(float(nac[1, 0, 0, 0]), 2.0)
        coarse = tb.batch_tensor(["a", "b"], cache, device, (1,))
        self.assertAlmostEqual(float(coarse[1, 0, 0, 0]), 3.0)
        fusion = tb.batch_tensor(["a", "b"], cache, device, (0, 1))
        self.assertEqual(tuple(fusion.shape), (2, 2, 4, 4))

    def test_variants_map_to_channel_sets(self):
        self.assertEqual(tb.CNN_VARIANTS["nac"], (0,))
        self.assertEqual(tb.CNN_VARIANTS["coarse_grid"], (1,))
        self.assertEqual(tb.CNN_VARIANTS["fusion"], (0, 1))


class TestCoarseFeatures(unittest.TestCase):
    def test_feature_list_present_in_hydrated_labels(self):
        path = ROOT / "outputs" / "labels_hydrated.csv"
        if not path.exists():
            self.skipTest("labels_hydrated.csv not built yet")
        columns = set(pd.read_csv(path, nrows=1).columns)
        for feature in tb.COARSE_FEATURES:
            self.assertIn(feature, columns)


if __name__ == "__main__":
    unittest.main()
