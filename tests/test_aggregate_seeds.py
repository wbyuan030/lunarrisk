#!/usr/bin/env python3
"""Tests for the multi-seed aggregation helpers (synthetic inputs only)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aggregate_seeds", ROOT / "scripts" / "aggregate_seeds.py")
agg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agg
SPEC.loader.exec_module(agg)


class TestPairedBootstrap(unittest.TestCase):
    def test_clear_winner_excludes_zero(self):
        y = np.array([0.0, 0.5, 1.0, 0.25, 0.75, 0.1, 0.9, 0.4])
        good = np.array([0.0, 0.5, 1.0, 0.25, 0.75, 0.1, 0.9, 0.4])   # perfect
        bad = np.full(8, 0.5)                                          # constant
        groups = np.array(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"])
        result = agg.paired_bootstrap_difference(
            agg.mae, y, {42: (good, bad)}, groups, 200, seed=0)
        self.assertLess(result["mean_over_seeds"], 0)
        self.assertLess(result["hi95"], 0)
        self.assertTrue(result["excludes_zero"])
        self.assertEqual(result["n_groups"], 4)

    def test_identical_models_have_zero_difference(self):
        y = np.array([0.0, 0.5, 1.0, 0.25])
        same = np.array([0.1, 0.4, 0.9, 0.3])
        groups = np.array(["g1", "g1", "g2", "g2"])
        result = agg.paired_bootstrap_difference(
            agg.mae, y, {42: (same, same)}, groups, 100, seed=0)
        self.assertAlmostEqual(result["mean_over_seeds"], 0.0)
        self.assertAlmostEqual(result["lo95"], 0.0)
        self.assertAlmostEqual(result["hi95"], 0.0)
        self.assertFalse(result["excludes_zero"])

    def test_differences_are_averaged_over_seeds(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        good = y.copy()
        bad = np.array([0.5, 0.5, 0.5, 0.5])
        groups = np.array(["g1", "g1", "g2", "g2"])
        two_seeds = agg.paired_bootstrap_difference(
            agg.mae, y, {42: (good, bad), 43: (good, bad)}, groups, 50, seed=1)
        one_seed = agg.paired_bootstrap_difference(
            agg.mae, y, {42: (good, bad)}, groups, 50, seed=1)
        self.assertAlmostEqual(two_seeds["mean_over_seeds"], one_seed["mean_over_seeds"])
        self.assertEqual(set(two_seeds["per_seed"]), {"42", "43"})

    def test_brier_statistic_uses_probabilities(self):
        truth = np.array([0.0, 1.0, 0.0, 1.0])
        confident = np.array([0.01, 0.99, 0.02, 0.98])
        uninformative = np.full(4, 0.5)
        groups = np.array(["g1", "g1", "g2", "g2"])
        result = agg.paired_bootstrap_difference(
            agg.brier, truth, {42: (confident, uninformative)}, groups, 100, seed=2)
        self.assertLess(result["mean_over_seeds"], 0)
        self.assertTrue(result["excludes_zero"])


class TestCohortChecks(unittest.TestCase):
    def _frame(self, ids, groups):
        return pd.DataFrame({
            "sample_id": ids, "group_id": groups, "LTM_CODE": ["L1"] * len(ids),
            "y_g10": [0.0] * len(ids),
        })

    def test_identical_cohorts_pass(self):
        frames = {42: self._frame(["a", "b"], ["g1", "g2"]),
                  43: self._frame(["a", "b"], ["g1", "g2"])}
        cohort = agg.assert_identical_cohorts([42, 43], frames)
        self.assertEqual(cohort["tiles"], 2)
        self.assertEqual(cohort["groups"], 2)

    def test_different_tile_order_fails(self):
        frames = {42: self._frame(["a", "b"], ["g1", "g2"]),
                  43: self._frame(["b", "a"], ["g2", "g1"])}
        with self.assertRaises(SystemExit):
            agg.assert_identical_cohorts([42, 43], frames)

    def test_split_independence_detects_group_leak(self):
        labels = pd.DataFrame({
            "group_id": ["g1", "g1", "g2"], "split": ["train", "test", "train"],
            "NAC_EDR_ID": ["a", "a", "b"], "nac_valid_fraction": [1.0, 1.0, 0.5],
        })
        audit = agg.check_split_independence(labels)
        self.assertEqual(audit["groups_in_multiple_splits"], 1)
        self.assertEqual(audit["products_in_multiple_splits"], 1)
        self.assertEqual(audit["usable"], 2)




class TestP17AuditGuards(unittest.TestCase):
    def test_split_audit_refuses_leaked_groups_and_products(self):
        leaked = {"groups_in_multiple_splits": 1, "products_in_multiple_splits": 0}
        with self.assertRaises(SystemExit):
            agg.require_split_independence(leaked)
        leaked = {"groups_in_multiple_splits": 0, "products_in_multiple_splits": 1}
        with self.assertRaises(SystemExit):
            agg.require_split_independence(leaked)
        agg.require_split_independence({
            "groups_in_multiple_splits": 0, "products_in_multiple_splits": 0
        })

    def test_raw_event_head_audit_detects_identical_predictions(self):
        keys = {
            "cnn_nac_raw_event_prob_g10_gt_0.05": [0.2, 0.8],
            "cnn_nac_raw_event_prob_g15_gt_0.02": [0.2, 0.8],
        }
        for variant in ("coarse_grid", "fusion"):
            keys[f"cnn_{variant}_raw_event_prob_g10_gt_0.05"] = [0.2, 0.8]
            keys[f"cnn_{variant}_raw_event_prob_g15_gt_0.02"] = [0.3, 0.7]
        with self.assertRaises(SystemExit):
            agg.event_head_divergence({42: pd.DataFrame(keys)}, [42])

    def test_raw_event_head_audit_passes_distinct_outputs(self):
        columns = {}
        for variant in ("nac", "coarse_grid", "fusion"):
            model = f"cnn_{variant}"
            columns[f"{model}_raw_event_prob_g10_gt_0.05"] = [0.15, 0.85]
            columns[f"{model}_raw_event_prob_g15_gt_0.02"] = [0.25, 0.75]
        result = agg.event_head_divergence({42: pd.DataFrame(columns)}, [42])
        self.assertFalse(result["cnn_fusion"]["42"]["identical"])
        self.assertAlmostEqual(result["cnn_fusion"]["42"]["mean_abs_gap"], 0.1)

    def test_raw_event_head_audit_rejects_missing_exports(self):
        with self.assertRaises(SystemExit):
            agg.event_head_divergence({42: pd.DataFrame({"x": [1, 2]})}, [42])


class TestNewConsistencyGuards(unittest.TestCase):
    """P18-03: truth/probability/protocol fail-fast and explicit NLL provenance."""

    def frame(self, seed=42):
        size = 4
        rows = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(size)],
            "group_id": ["g1", "g1", "g2", "g2"],
            "LTM_CODE": ["L1"] * size,
            "y_g10": [0.0, 0.1, 0.6, 0.9],
            "y_g15": [0.0, 0.0, 0.3, 0.5],
        })
        for rule in agg.RULES:
            rows[f"event_{rule}"] = np.zeros(size, dtype=int)
        for model in agg.MODELS:
            for rule in agg.RULES:
                rows[f"{model}_prob_{rule}"] = np.full(size, 0.5)
        for variant in agg.CNN_VARIANTS:
            for rule in agg.HEAD_RULES:
                rows[f"cnn_{variant}_raw_event_prob_{rule}"] = np.full(size, 0.5)
        return rows

    def metrics(self, seed=42):
        rule_payload = {rule: {"primary": "raw", "metrics_calibrated": None,
                               "metrics_raw": {"nll": 0.5}} for rule in agg.RULES}
        model_payload = {model: {"test": rule_payload} for model in agg.MODELS}
        return {seed: {"models": model_payload}}

    def test_nll_matches_manual_computation(self):
        truth = np.array([0.0, 1.0])
        probability = np.array([0.25, 0.75])
        # probability is P(class=1); truth 0 uses log(1 - p)
        expected = -(np.log(1 - 0.25) + np.log(0.75)) / 2
        self.assertAlmostEqual(agg.nll(probability, truth), float(expected), places=8)

    def test_nll_clips_extreme_probabilities(self):
        self.assertTrue(np.isfinite(agg.nll(np.array([0.0, 1.0]), np.array([0.0, 1.0]))))

    def test_identical_seeds_pass(self):
        frames = {42: self.frame(), 43: self.frame()}
        audit = agg.assert_truth_consistency(frames, [42, 43])
        self.assertIn("event_g10_gt_0.05", audit)

    def test_truth_drift_is_fatal(self):
        frames = {42: self.frame(), 43: self.frame()}
        frames[43].loc[2, "y_g10"] = 0.7
        with self.assertRaises(SystemExit):
            agg.assert_truth_consistency(frames, [42, 43])

    def test_event_label_drift_is_fatal(self):
        frames = {42: self.frame(), 43: self.frame()}
        frames[43].loc[1, "event_g10_gt_0.05"] = 1
        with self.assertRaises(SystemExit):
            agg.assert_truth_consistency(frames, [42, 43])

    def test_group_drift_is_fatal(self):
        frames = {42: self.frame(), 43: self.frame()}
        frames[43].loc[0, "group_id"] = "g9"
        with self.assertRaises(SystemExit):
            agg.assert_truth_consistency(frames, [42, 43])

    def test_probability_quality_accepts_valid_frames(self):
        frames = {42: self.frame()}
        audit = agg.assert_probability_quality(frames, [42], self.metrics(42))
        self.assertGreater(audit["checked"], 0)

    def test_probability_quality_rejects_nan(self):
        frames = {42: self.frame()}
        frames[42].loc[0, "cnn_fusion_prob_g10_gt_0.05"] = np.nan
        with self.assertRaises(SystemExit):
            agg.assert_probability_quality(frames, [42], self.metrics(42))

    def test_probability_quality_rejects_claimed_calibration_without_metrics(self):
        frames = {42: self.frame()}
        metrics = self.metrics(42)
        metrics[42]["models"]["cnn_fusion"]["test"]["g10_gt_0.05"]["primary"] = "calibrated"
        metrics[42]["models"]["cnn_fusion"]["test"]["g10_gt_0.05"]["metrics_calibrated"] = None
        with self.assertRaises(SystemExit):
            agg.assert_probability_quality(frames, [42], metrics)

    def _seed_metrics(self, seed, ablation=None, provenance=None, batch=16):
        protocol = {
            "seed": seed, "input_size": 128, "augment": False, "augmentation_policy": "none",
            "epochs_max": 60, "patience": 10, "batch": batch, "lr": 0.001,
            "min_nac_valid": 0.99, "channels": ["a"], "scalar_scaling": "train-only",
            "calibration": "cal", "rules": {}, "composite_rule": "r", "contrast_channel": "c",
            "ablation": ablation or {"use_geometry": True, "use_nac_photometry": True},
            "provenance": provenance or {"labels": {"sha256": "abc"}},
        }
        return {seed: {"protocol": protocol}}

    def test_seed_protocol_consistency_passes(self):
        metrics = {}
        for seed in (42, 43):
            metrics.update(self._seed_metrics(seed))
        audit = agg.assert_seed_protocol_consistency(metrics, [42, 43])
        self.assertEqual(audit["missing"], [])

    def test_seed_batch_drift_is_fatal(self):
        metrics = self._seed_metrics(42)
        metrics.update(self._seed_metrics(43, batch=32))
        with self.assertRaises(SystemExit):
            agg.assert_seed_protocol_consistency(metrics, [42, 43])

    def test_seed_provenance_drift_is_fatal(self):
        metrics = self._seed_metrics(42)
        metrics.update(self._seed_metrics(43, provenance={"labels": {"sha256": "def"}}))
        with self.assertRaises(SystemExit):
            agg.assert_seed_protocol_consistency(metrics, [42, 43])

    def test_wrong_seed_label_is_fatal(self):
        metrics = self._seed_metrics(42)
        with self.assertRaises(SystemExit):
            agg.assert_seed_protocol_consistency(metrics, [43])

    def test_missing_hyperparameters_are_reported_not_guessed(self):
        metrics = self._seed_metrics(42)
        del metrics[42]["protocol"]["batch"]
        audit = agg.assert_seed_protocol_consistency(metrics, [42])
        self.assertIn("batch", audit["missing"])


if __name__ == "__main__":
    unittest.main()
