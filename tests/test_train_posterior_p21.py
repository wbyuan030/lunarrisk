#!/usr/bin/env python3
"""Tests for the A3 runner's pure helpers (no data files, no training)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("train_posterior_p21",
                                              ROOT / "scripts" / "train_posterior_p21.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

from lunarrisk.rules import Compare, standard_rules  # noqa: E402


class TestEventMetrics(unittest.TestCase):
    def truth(self):
        return {"g10": np.array([0.01, 0.2, 0.3, 0.6]),
                "g15": np.array([0.0, 0.1, 0.2, 0.1]),
                "roughness_9m": np.array([0.01, 0.06, 0.2, 0.2]),
                "relief_p95_p5": np.array([3.0, 40.0, 120.0, 120.0])}

    def test_probabilities_are_bounded_and_accept_counts_add_up(self):
        predicted = {k: np.repeat(v[None, :], 32, axis=0) for k, v in self.truth().items()}
        metrics = runner.event_metrics(self.truth(), predicted, standard_rules(), seed=0)
        for name, entry in metrics.items():
            self.assertGreaterEqual(entry["brier"], 0.0)
            self.assertLessEqual(entry["prevalence"], 1.0)
            for coverage in (10, 20, 40):
                self.assertEqual(entry[f"accepted@{coverage}"],
                                 max(1, int(round(coverage / 100 * 4))))

    def test_delta_posterior_is_handled_without_draws(self):
        # point predictions have no draw axis; the metric must still produce a per-row probability
        predicted = dict(self.truth())
        metrics = runner.event_metrics(self.truth(), predicted, {"r": Compare("g10", 0.05)}, seed=0)
        self.assertIn("brier", metrics["r"])

    def test_perfect_prediction_has_zero_brier(self):
        truth = self.truth()
        draws = np.repeat(truth["g10"][None, :], 8, axis=0)
        predicted = dict(truth)
        predicted["g10"] = np.where(draws > 0.5, 1.0, 0.0)
        metrics = runner.event_metrics(truth, predicted, {"r": Compare("g10", 0.5)}, seed=0)
        self.assertLess(metrics["r"]["brier"], 1e-9)


class TestLowRankAblation(unittest.TestCase):
    def test_zeroing_removes_only_the_u_columns_and_restores_exactly(self):
        import torch
        from lunarrisk.probabilistic import condition_posterior
        model = condition_posterior("joint", width=8, rank=2, n_scalars=0)
        before = model.head.weight.detach().clone()
        saved = runner.zero_low_rank(model.head)
        after = model.head.weight.detach()
        self.assertTrue(torch.all(after[:8] == before[:8]))      # mu and log sigma untouched
        self.assertTrue(torch.all(after[8:] == 0))               # U columns zeroed
        runner.restore_low_rank(model.head, saved)
        self.assertTrue(torch.all(model.head.weight.detach() == before))


class TestAttributeMetrics(unittest.TestCase):
    def test_mae_and_rmse(self):
        truth = {"g10": np.array([0.0, 1.0]), "g15": np.array([0.0, 1.0]),
                 "roughness_9m": np.array([0.0, 2.0]), "relief_p95_p5": np.array([0.0, 4.0])}
        predicted = {"g10": np.array([[0.1, 0.9]] * 4), "g15": np.array([[0.1, 0.9]] * 4),
                     "roughness_9m": np.array([[0.5, 1.5]] * 4),
                     "relief_p95_p5": np.array([[1.0, 3.0]] * 4)}
        metrics = runner.attribute_metrics(truth, predicted)
        self.assertAlmostEqual(metrics["g10"]["mae"], 0.1, places=6)
        self.assertAlmostEqual(metrics["roughness_9m"]["mae"], 0.5, places=6)


class TestGroupBootstrap(unittest.TestCase):
    def test_ci_brackets_the_point_estimate(self):
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        values = {"g10": np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])}
        result = runner.group_bootstrap(None, values, values, groups,
                                        lambda rows: float(np.mean(values["g10"][rows])),
                                        draws=200, seed=0)
        self.assertEqual(result["groups"], 3)
        self.assertLessEqual(result["p05"], result["p50"])
        self.assertLessEqual(result["p50"], result["p95"])
        self.assertGreaterEqual(result["p05"], 0.0)
        self.assertLessEqual(result["p95"], 1.0)


if __name__ == "__main__":
    unittest.main()
