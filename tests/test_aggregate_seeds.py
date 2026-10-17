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


if __name__ == "__main__":
    unittest.main()
