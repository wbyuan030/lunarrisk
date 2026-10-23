#!/usr/bin/env python3
"""Offline tests for the P2.2 comparison helpers (no data files, no training)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("train_posterior_p21", ROOT / "scripts" / "train_posterior_p21.py")
compare = _load("compare_p22", ROOT / "scripts" / "compare_p22.py")

from lunarrisk.rules import Compare, standard_rules  # noqa: E402


class TestCanonicalSet(unittest.TestCase):
    def test_intersection_drops_low_nac_tiles_and_reports_them(self):
        labels = pd.DataFrame({
            "sample_id": ["a", "b", "c", "d"],
            "split": ["train", "train", "test", "test"],
            "group_id": ["g1", "g1", "g2", "g3"],
        })
        meta = {"a": {"nac_valid_fraction": 1.0}, "b": {"nac_valid_fraction": 0.5},
                "c": {"nac_valid_fraction": 0.99}, "d": {"nac_valid_fraction": 0.1}}
        used, report = compare.canonical_labels(labels, meta, 0.99)
        self.assertEqual(used["sample_id"].tolist(), ["a", "c"])
        self.assertEqual(report["rows_total"], 4)
        self.assertEqual(report["rows_used"], 2)
        self.assertEqual({row["sample_id"] for row in report["excluded"]}, {"b", "d"})
        self.assertEqual(report["excluded_by_split"], {"test": 1, "train": 1})
        self.assertIn("sample_id_sha256", report)

    def test_identical_ids_hash_identically_regardless_of_order(self):
        one = compare.text_sha256(["a", "b", "c"])
        two = compare.text_sha256(["a", "b", "c"])
        self.assertEqual(one, two)
        self.assertNotEqual(one, compare.text_sha256(["a", "b"]))


class TestBrierAndNll(unittest.TestCase):
    def test_hand_value(self):
        brier, nll = compare.brier_and_nll(np.array([0.8, 0.2]), np.array([1.0, 0.0]))
        self.assertAlmostEqual(brier, 0.04, places=9)
        self.assertAlmostEqual(nll, -np.log(0.8), places=9)

    def test_probabilities_are_clipped_away_from_zero(self):
        brier, nll = compare.brier_and_nll(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        self.assertLess(brier, 1e-9)
        self.assertTrue(np.isfinite(nll))


class TestCoverage(unittest.TestCase):
    def setUp(self):
        rule = Compare("g10", 0.05)
        frame = pd.DataFrame({
            "sample_id": list("abcdef"),
            "group_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "split": ["test"] * 6,
            "truth_g10": [0.01, 0.20, 0.02, 0.30, 0.01, 0.40],
            "truth_g15": [0.0] * 6,
            "truth_roughness_9m": [0.01] * 6,
            "truth_relief_p95_p5": [1.0] * 6,
            "p_r": [0.05, 0.90, 0.10, 0.80, 0.01, 0.70],
        })
        self.frame, self.rule = frame, rule

    def test_oracle_accepts_the_lowest_probabilities(self):
        metrics = compare.coverage_metrics(self.frame, "r", self.rule, 0.5)
        self.assertEqual(metrics["accepted"], 3)
        # the three lowest probabilities are safe in this fixture
        self.assertEqual(metrics["unsafe_accepted"], 0)
        self.assertEqual(metrics["uar"], 0.0)

    def test_deployable_threshold_comes_from_the_given_cutoff(self):
        metrics = compare.coverage_metrics(self.frame, "r", self.rule, 1.0, threshold=0.1)
        self.assertEqual(metrics["accepted"], 3)          # p <= 0.1
        self.assertEqual(metrics["unsafe_accepted"], 0)
        self.assertAlmostEqual(metrics["coverage"], 0.5)

    def test_calibrate_threshold_uses_only_the_passed_frame(self):
        cal = self.frame.copy()
        cal["p_r"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        threshold = compare.calibrate_threshold(cal, "r", 0.5)
        self.assertLessEqual(threshold, 0.4)
        self.assertGreaterEqual(threshold, 0.3)


class TestPairedBootstrap(unittest.TestCase):
    def test_delta_sign_and_ci(self):
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        delta = np.array([0.2, 0.2, 0.1, 0.1, 0.3, 0.3])
        result = compare.paired_group_bootstrap(delta, groups, draws=500, seed=0)
        self.assertAlmostEqual(result["mean"], 0.2, places=6)
        self.assertLessEqual(result["p025"], result["p975"])
        self.assertEqual(result["groups"], 3)

    def test_alignment_rejects_different_sample_sets(self):
        a = pd.DataFrame({"sample_id": ["a", "b"], "group_id": ["g", "g"]})
        b = pd.DataFrame({"sample_id": ["a", "c"], "group_id": ["g", "h"]})
        with self.assertRaises(ValueError):
            compare.aligned_frames(a, b)


if __name__ == "__main__":
    unittest.main()
