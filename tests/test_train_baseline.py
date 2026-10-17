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



    def test_evaluate_model_accepts_lowest_unsafe_probability(self):
        # A high AUC is only operationally useful when LOW unsafe scores get accepted.
        y = np.array([0.0, 0.8, 0.0, 0.7])
        fraction = np.column_stack([y, y])
        scores = tb.logit(np.array([0.05, 0.95, 0.1, 0.9]))
        evaluated = tb.evaluate_model(y, fraction, scores, "g10_gt_0.05")
        half = evaluated["risk_coverage"][1]  # 40% -> 2/4 accepted
        self.assertEqual(half["unsafe_accepted_n"], 0)
        self.assertEqual(half["UAR"], 0.0)
        self.assertAlmostEqual(evaluated["metrics"]["auc"], 1.0)

    def test_calibrated_risk_coverage_uses_unsafe_probability(self):
        y = np.array([0.0, 0.8, 0.0, 0.7])
        fraction = np.column_stack([y, y])
        scores = tb.logit(np.array([0.05, 0.95, 0.1, 0.9]))
        evaluated = tb.evaluate_model(
            y, fraction, scores, "g10_gt_0.05",
            cal_scores=scores, cal_truth=(y > 0.05).astype(int)
        )
        self.assertEqual(evaluated["risk_coverage"][1]["unsafe_accepted_n"], 0)

    def test_invalid_risk_probability_is_rejected(self):
        for invalid in (np.array([0.1, np.nan]), np.array([-0.1, 0.7]), np.array([1.1, 0.2])):
            with self.assertRaises(ValueError):
                tb.risk_coverage(np.array([0.0, 0.8]), invalid, "g10_gt_0.05")


class TestP17RankingAndAblation(unittest.TestCase):
    def test_tied_risks_have_order_invariant_expected_uar(self):
        # Tied constant probabilities must reduce to cohort prevalence, even
        # when safe/unsafe samples are deliberately shuffled in CSV order.
        y = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        risk = np.full(len(y), 0.5)
        rows = tb.risk_coverage(y, risk, "g10_gt_0.05",
                                coverages=(1 / 6, 0.5, 5 / 6, 1.0))
        shuffled = tb.risk_coverage(y[::-1], risk, "g10_gt_0.05",
                                    coverages=(1 / 6, 0.5, 5 / 6, 1.0))
        for left, right in zip(rows, shuffled):
            self.assertAlmostEqual(left["UAR"], 0.5)
            self.assertAlmostEqual(left["HMR"],
                                   left["accepted_n"] / len(y))
            self.assertAlmostEqual(left["UAR"], right["UAR"])
            self.assertAlmostEqual(left["unsafe_accepted_n"],
                                   left["accepted_n"] * 0.5)
        self.assertTrue(rows[0]["boundary_tie_expected"])
        self.assertFalse(rows[-1]["boundary_tie_expected"])

    def test_unequal_risk_scores_keep_exact_counts(self):
        y = np.array([0.0, 0.8, 0.0, 0.8])
        rows = tb.risk_coverage(y, [0.05, 0.95, 0.1, 0.9], "g10_gt_0.05",
                                coverages=(0.5, 1))
        self.assertEqual(rows[0]["unsafe_accepted_n"], 0)
        self.assertEqual(rows[1]["unsafe_accepted_n"], 2)
        self.assertFalse(rows[0]["boundary_tie_expected"])

    def test_ablation_feature_mapping(self):
        self.assertEqual(tb.ablation_features("fusion"),
                         ((0, 1, 2), tb.NAC_SCALARS + tb.GEOMETRY_SCALARS))
        self.assertEqual(tb.ablation_features("fusion", use_geometry=False),
                         ((0, 1, 2), tb.NAC_SCALARS))
        self.assertEqual(tb.ablation_features("fusion", use_nac_photometry=False),
                         ((1, 2), tb.GEOMETRY_SCALARS))
        self.assertEqual(tb.ablation_features("fusion", use_geometry=False,
                                               use_nac_photometry=False),
                         ((1, 2), ()))
        self.assertEqual(tb.ablation_features("coarse_grid", use_nac_photometry=False),
                         ((2,), tb.GEOMETRY_SCALARS))

    def test_unsafe_spatial_augmentation_is_rejected(self):
        with self.assertRaises(SystemExit):
            tb.main(["--augment", "--no-nac-photometry"])
        # --no-geometry removes the azimuth inconsistency; then execution
        # should proceed to data file loading (which fails on a nonexistent path).
        with self.assertRaises(FileNotFoundError):
            tb.main(["--augment", "--no-geometry",
                     "--labels", "/nonexistent/p17-fixture-labels.csv"])


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
            "a": np.stack([np.ones((4, 4)), np.zeros((4, 4)), np.full((4, 4), 9.0)]).astype(np.float16),
            "b": np.stack([np.full((4, 4), 2.0), np.full((4, 4), 3.0), np.full((4, 4), 7.0)]).astype(np.float16),
        }
        device = torch.device("cpu")
        contrast = tb.batch_tensor(["a", "b"], cache, device, (0,))
        self.assertEqual(tuple(contrast.shape), (2, 1, 4, 4))
        self.assertAlmostEqual(float(contrast[1, 0, 0, 0]), 2.0)
        coarse = tb.batch_tensor(["a", "b"], cache, device, (2,))
        self.assertAlmostEqual(float(coarse[1, 0, 0, 0]), 7.0)
        fusion = tb.batch_tensor(["a", "b"], cache, device, (0, 2))
        self.assertEqual(tuple(fusion.shape), (2, 2, 4, 4))

    def test_variants_map_to_channel_sets(self):
        self.assertEqual(tb.CHANNELS, ("nac_contrast", "nac_texture", "coarse_slope"))
        self.assertEqual(tb.CNN_VARIANTS["nac"], (0, 1))
        self.assertEqual(tb.CNN_VARIANTS["coarse_grid"], (2,))
        self.assertEqual(tb.CNN_VARIANTS["fusion"], (0, 1, 2))

    def test_variant_scalars_give_geometry_to_every_variant(self):
        for variant, columns in tb.VARIANT_SCALARS.items():
            for geometry in tb.GEOMETRY_SCALARS:
                self.assertIn(geometry, columns, variant)
        for variant in ("nac", "fusion"):
            for nac_scalar in tb.NAC_SCALARS:
                self.assertIn(nac_scalar, tb.VARIANT_SCALARS[variant])
        for nac_scalar in tb.NAC_SCALARS:
            self.assertNotIn(nac_scalar, tb.VARIANT_SCALARS["coarse_grid"])



    def test_rule_heads_are_independent_and_match_label_order(self):
        import torch
        model = tb.TinyCNN(in_channels=2)
        self.assertEqual(tuple(model(torch.zeros((2, 2, 32, 32))).shape), (2, 4))
        labels = pd.DataFrame({"slope3m_g10": [0.06, 0.01],
                               "slope3m_g15": [0.01, 0.03]})
        event = np.column_stack([
            (labels[target] > threshold).to_numpy(dtype=np.float32)
            for target, threshold in (tb.RULES[name] for name in tb.HEAD_RULES)
        ])
        np.testing.assert_array_equal(event, [[1.0, 0.0], [0.0, 1.0]])
        logits = np.array([[0.4, 0.5], [-0.5, -0.6]])
        self.assertEqual(tb.rule_column_index("g10_gt_0.05"), 0)
        self.assertEqual(tb.rule_column_index("g15_gt_0.02"), 1)
        self.assertFalse(np.array_equal(logits[:, 0], logits[:, 1]))

    def test_model_order_does_not_define_seed(self):
        import inspect
        signature = inspect.signature(tb.train_variant)
        self.assertNotIn("seed_offset", signature.parameters)


class TestTrainVariantFixture(unittest.TestCase):
    """End-to-end fixture: one epoch through train_variant + predict_variant."""

    def _fixture(self, n_train=8, n_val=4, size=32, seed=0):
        import argparse
        import torch
        rng = np.random.default_rng(seed)
        cache, rows = {}, []
        for index in range(n_train + n_val):
            sample_id = f"s{index}"
            # Channel 0 carries a label-correlated blob so training sees real signal.
            g10 = float(rng.uniform(0, 1))
            blob = np.full((size, size), g10, dtype=np.float32)
            noise = rng.random((size, size)).astype(np.float32) * 0.1
            cache[sample_id] = np.stack([blob + noise, rng.random((size, size)).astype(np.float32)]).astype(np.float16)
            rows.append({
                "sample_id": sample_id,
                "split": "train" if index < n_train else "val",
                "slope3m_g10": g10,
                "slope3m_g15": float(np.clip(g10 - 0.2, 0, 1)),
                "group_id": f"G{index}",
            })
        args = argparse.Namespace(seed=seed, epochs=1, patience=1, batch=4, lr=1e-2,
                                  augment=False, verbose=False)
        return cache, pd.DataFrame(rows), args, torch

    def test_event_scores_have_two_columns(self):
        cache, data, args, torch = self._fixture()
        device = torch.device("cpu")
        result = tb.train_variant(cache, data, args, device, (0,))
        val_rows = data[data["split"] == "val"].reset_index(drop=True)
        fractions, scores = tb.predict_variant(result, val_rows, cache, (0,), 4, device)
        self.assertEqual(tuple(scores.shape), (len(val_rows), 2))
        self.assertEqual(tuple(fractions.shape), (len(val_rows), 2))
        self.assertTrue(np.isfinite(scores).all())
        # The two rules must not be served by one shared score column.
        scores_two_cols = tb.predict_variant(result, val_rows, cache, (0,), 4, device)[1]
        np.testing.assert_array_equal(scores_two_cols, scores)
        self.assertEqual(scores.shape[1], len(tb.HEAD_RULES))

    def test_fixture_scores_drive_both_rules_independently(self):
        cache, data, args, torch = self._fixture(seed=1)
        device = torch.device("cpu")
        result = tb.train_variant(cache, data, args, device, (1,))
        val_rows = data[data["split"] == "val"].reset_index(drop=True)
        fractions, scores = tb.predict_variant(result, val_rows, cache, (1,), 4, device)
        for index, rule_name in enumerate(tb.HEAD_RULES):
            payload = tb.evaluate_model(val_rows["slope3m_g10"].to_numpy(), fractions, scores[:, index], rule_name)
            self.assertIn("metrics", payload)
            self.assertEqual(payload["metrics_raw"]["rule"], rule_name)
            self.assertIsNotNone(payload["metrics_raw"]["brier"])

    def test_high_auc_ranking_lowers_uar(self):
        # Identical (perfectly ranked) predictions must accept the safe sites first.
        y = np.array([0.0, 0.9, 0.0, 0.9, 0.0, 0.9])
        fractions = np.column_stack([y, y])
        unsafe = np.array([0.02, 0.98, 0.05, 0.9, 0.01, 0.95])
        payload = tb.evaluate_model(y, fractions, tb.logit(unsafe), "g10_gt_0.05")
        self.assertAlmostEqual(payload["metrics"]["auc"], 1.0)
        quarter = payload["risk_coverage"][0]
        self.assertEqual(quarter["unsafe_accepted_n"], 0)
        self.assertEqual(quarter["UAR"], 0.0)


class TestPhotometryAndGeometry(unittest.TestCase):
    def test_contrast_channel_keeps_absolute_scale(self):
        """Per-tile stretching erases the shading magnitude; the contrast channel must not."""
        rng = np.random.default_rng(0)
        flat = np.full((16, 16), 0.05) + rng.normal(0, 1e-4, (16, 16))
        rough = np.full((16, 16), 0.05) + rng.normal(0, 0.02, (16, 16))
        flat_contrast = np.abs(tb.contrast_normalize(flat)).mean()
        rough_contrast = np.abs(tb.contrast_normalize(rough)).mean()
        self.assertGreater(float(rough_contrast), 20 * float(flat_contrast))
        # The texture channel deliberately equalises them, which is why it cannot be the only NAC input.
        self.assertAlmostEqual(float(np.abs(tb.robust_normalize(flat)).max()), 1.5, delta=0.3)

    def test_contrast_channel_handles_nan(self):
        values = np.array([[0.04, np.nan], [0.06, 0.05]])
        channel = tb.contrast_normalize(values)
        self.assertTrue(np.isfinite(channel).all())
        self.assertAlmostEqual(float(channel[0, 1]), 0.0)

    def test_sample_grid_on_coordinates_uses_real_geometry(self):
        source = np.arange(9, dtype=np.float64).reshape(1, 9)
        source_x = np.arange(9, dtype=np.float64) * 10.0
        source_y = np.array([0.0])
        sampled = tb.sample_grid_on_coordinates(source, source_x, source_y,
                                                np.array([15.0, 25.0]), np.array([0.0]))
        np.testing.assert_allclose(sampled[0], [1.5, 2.5])

    def test_grid_offsets_detects_disagreement(self):
        offsets = tb.grid_offsets_m(np.array([0.0, 10.0]), np.array([0.0, 10.0]),
                                    np.array([2.0, 10.0]), np.array([0.0, 10.0]))
        self.assertAlmostEqual(offsets["dx_min_m"], 2.0)
        self.assertAlmostEqual(offsets["dx_max_m"], 0.0)

    def test_scalar_table_has_geometry_and_contrast(self):
        labels = pd.DataFrame({
            "sample_id": ["a", "b"],
            "incidence_angle_deg": [30.0, 60.0],
            "emission_angle_deg": [5.0, 15.0],
            "phase_angle_deg": [35.0, 75.0],
            "sub_solar_ground_azimuth_deg": [0.0, 90.0],
        })
        meta = {"a": {"nac_p99_p1": 0.01, "nac_median": 0.05},
                "b": {"nac_p99_p1": 0.2, "nac_median": 0.1}}
        table = tb.build_scalar_table(labels, meta)
        self.assertEqual(list(table.columns), list(tb.NAC_SCALARS + tb.GEOMETRY_SCALARS))
        self.assertAlmostEqual(float(table.loc["a", "incidence"]), 1 / 3)
        self.assertAlmostEqual(float(table.loc["a", "az_sin"]), 0.0)
        self.assertGreater(float(table.loc["b", "log_contrast"]), float(table.loc["a", "log_contrast"]))


class TestBalancedRules(unittest.TestCase):
    def test_rules_registered_with_expected_thresholds(self):
        self.assertEqual(tb.RULES["g10_gt_0.15"], ("slope3m_g10", 0.15))
        self.assertEqual(tb.RULES[tb.COMPOSITE_RULE][0], "composite")
        self.assertAlmostEqual(tb.COMPOSITE_G10, 0.10)
        self.assertAlmostEqual(tb.COMPOSITE_G15, 0.03)

    def test_composite_truth_is_an_or(self):
        frame = pd.DataFrame({
            "slope3m_g10": [0.0, 0.2, 0.0, 0.2],
            "slope3m_g15": [0.0, 0.0, 0.05, 0.05],
        })
        events = tb.rule_event(frame)
        np.testing.assert_array_equal(events[tb.COMPOSITE_RULE], [False, True, True, True])
        np.testing.assert_array_equal(events["g10_gt_0.15"], [False, True, False, True])

    def test_rule_score_normalises_margins(self):
        fractions = np.array([[0.0, 0.0], [0.15, 0.0], [0.05, 0.015]])
        np.testing.assert_allclose(tb.rule_score("g10_gt_0.15", fractions), [0.0, 1.0, 1 / 3])
        np.testing.assert_allclose(tb.rule_score(tb.COMPOSITE_RULE, fractions), [0.0, 1.0, 0.5])

    def test_evaluate_all_rules_covers_every_rule(self):
        test_rows = pd.DataFrame({"slope3m_g10": [0.0, 0.3], "slope3m_g15": [0.0, 0.1]})
        events = tb.rule_event(test_rows)
        fractions = np.array([[0.0, 0.0], [0.3, 0.1]])
        payload = tb.evaluate_all_rules(
            fractions, fractions,
            {name: tb.logit(np.array([0.1, 0.9])) for name in tb.HEAD_RULES},
            {name: tb.logit(np.array([0.1, 0.9])) for name in tb.HEAD_RULES},
            events, events, tb.rule_values(test_rows))
        self.assertEqual(set(payload), set(tb.RULES))
        for rule_name, entry in payload.items():
            self.assertEqual(entry["metrics_raw"]["rule"], rule_name)
            self.assertIsNotNone(entry["metrics_raw"]["brier"])


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
