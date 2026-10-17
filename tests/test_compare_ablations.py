#!/usr/bin/env python3
"""Tests for the cross-config ablation statistics (synthetic inputs only)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compare_ablations", ROOT / "scripts" / "compare_ablations.py")
ca = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ca
SPEC.loader.exec_module(ca)

PRIMARY_RULE = ca.PRIMARY
PROB = f"cnn_fusion_prob_{PRIMARY_RULE}"
PROB_COARSE = f"cnn_coarse_grid_prob_{PRIMARY_RULE}"


def frame(fusion_values, coarse_values, y_values, fusion_prob, coarse_prob, groups=None):
    size = len(y_values)
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(size)],
        "group_id": groups if groups is not None else ["g1"] * (size // 2) + ["g2"] * (size - size // 2),
        "y_g10": y_values,
        "cnn_fusion_g10": fusion_values,
        "cnn_coarse_grid_g10": coarse_values,
        f"cnn_fusion_prob_{PRIMARY_RULE}": fusion_prob,
        f"cnn_coarse_grid_prob_{PRIMARY_RULE}": coarse_prob,
    })


class TestPairedDifference(unittest.TestCase):
    def test_clear_win_excludes_zero(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        good = y.copy()
        bad = np.full(4, 0.5)
        rows = {
            ("cfg", 42): frame(good, bad, y,
                               np.array([0.01, 0.01, 0.99, 0.99]),
                               np.full(4, 0.5)),
        }
        groups = rows[("cfg", 42)]["group_id"].to_numpy()
        result = ca.paired_difference(rows, "cfg", [42], groups,
                                      "cnn_fusion", "cnn_coarse_grid", "mae", 200, seed=0)
        self.assertLess(result["mean_over_seeds"], 0)
        self.assertTrue(result["excludes_zero"])

    def test_identical_models_are_zero(self):
        y = np.array([0.0, 0.4, 1.0, 0.2])
        same = np.array([0.1, 0.3, 0.9, 0.25])
        rows = {("cfg", 42): frame(same, same, y, np.full(4, 0.5), np.full(4, 0.5))}
        groups = rows[("cfg", 42)]["group_id"].to_numpy()
        result = ca.paired_difference(rows, "cfg", [42], groups,
                                      "cnn_fusion", "cnn_coarse_grid", "mae", 100, seed=0)
        self.assertAlmostEqual(result["mean_over_seeds"], 0.0)
        self.assertFalse(result["excludes_zero"])

    def test_brier_uses_probability_columns(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        same = y.copy()
        rows = {("cfg", 42): frame(same, same, y,
                                   np.array([0.01, 0.01, 0.99, 0.99]),
                                   np.full(4, 0.5))}
        groups = rows[("cfg", 42)]["group_id"].to_numpy()
        result = ca.paired_difference(rows, "cfg", [42], groups,
                                      "cnn_fusion", "cnn_coarse_grid", "brier", 100, seed=0)
        self.assertLess(result["mean_over_seeds"], 0)
        self.assertTrue(result["excludes_zero"])

    def test_seeds_are_averaged(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        good = y.copy()
        bad = np.full(4, 0.5)
        rows = {
            ("cfg", 42): frame(good, bad, y, np.full(4, 0.5), np.full(4, 0.5)),
            ("cfg", 43): frame(good, bad, y, np.full(4, 0.5), np.full(4, 0.5)),
        }
        groups = rows[("cfg", 42)]["group_id"].to_numpy()
        two = ca.paired_difference(rows, "cfg", [42, 43], groups,
                                   "cnn_fusion", "cnn_coarse_grid", "mae", 50, seed=0)
        one = ca.paired_difference(rows, "cfg", [42], groups,
                                   "cnn_fusion", "cnn_coarse_grid", "mae", 50, seed=0)
        self.assertAlmostEqual(two["mean_over_seeds"], one["mean_over_seeds"])
        self.assertEqual(set(two["per_seed"]), {"42", "43"})


class TestDifferenceOfDifferences(unittest.TestCase):
    def test_reference_is_skipped(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        same = np.array([0.2, 0.3, 0.8, 0.9])
        rows = {("ref", 42): frame(same, same, y, np.full(4, 0.5), np.full(4, 0.5))}
        groups = rows[("ref", 42)]["group_id"].to_numpy()
        result = ca.difference_of_differences(rows, ["ref"], "ref", [42], groups, "mae", 20, 0)
        self.assertEqual(result, [])

    def test_increment_change_is_detected(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        bad = np.full(4, 0.5)
        # Reference: fusion perfect, coarse constant -> gap = -0.5 (fusion better).
        rows = {
            ("ref", 42): frame(y, bad, y, np.full(4, 0.5), np.full(4, 0.5)),
            ("same", 42): frame(y, bad, y, np.full(4, 0.5), np.full(4, 0.5)),   # unchanged
            ("gone", 42): frame(bad, bad, y, np.full(4, 0.5), np.full(4, 0.5)),  # increment removed
        }
        groups = rows[("ref", 42)]["group_id"].to_numpy()
        result = ca.difference_of_differences(
            rows, ["same", "gone"], "ref", [42], groups, "mae", 200, seed=0)
        by_config = {entry["config"]: entry for entry in result}
        self.assertAlmostEqual(by_config["same"]["mean"], 0.0)
        self.assertFalse(by_config["same"]["excludes_zero"])
        self.assertGreater(by_config["gone"]["mean"], 0)
        self.assertTrue(by_config["gone"]["excludes_zero"])


class TestCohortTable(unittest.TestCase):
    def test_headline_metrics_from_exported_columns(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        frame_rows = frame(y, np.full(4, 0.5), y, np.array([0.01, 0.02, 0.98, 0.99]),
                           np.full(4, 0.5))
        for model in ca.MODELS:
            column = ca.PROBABILITY_COLUMNS[model]
            frame_rows[column] = np.full(4, 0.5)
        frame_rows[ca.PROBABILITY_COLUMNS["cnn_fusion"]] = np.array([0.01, 0.02, 0.98, 0.99])
        for model in ca.MODELS:
            fraction = ca.FRACTION_COLUMNS[model]
            if fraction not in frame_rows.columns:
                frame_rows[fraction] = np.full(4, 0.5)
        frames = {("cfg", 42): frame_rows}
        metrics = {
            ("cfg", 42): {
                "models": {
                    model: {"test": {PRIMARY_RULE: {
                        "metrics": {"mae": 0.1, "brier": 0.2},
                        "metrics_raw": {"brier": 0.25, "nll": 0.5, "auc": 1.0},
                    }}} for model in ca.MODELS
                }
            }
        }
        table = ca.cohort_table(frames, metrics, ["cfg"], [42])
        self.assertAlmostEqual(table["cfg"]["cnn_fusion"]["mae"]["mean"], 0.1)
        self.assertEqual(table["cfg"]["cnn_fusion"]["mae"]["std"], 0.0)
        self.assertAlmostEqual(table["cfg"]["cnn_fusion"]["uar20"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
