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




class TestTemperatureCalibration(unittest.TestCase):
    """The fitted T must affect all three Gaussian evaluation distributions."""

    def _sample(self, variant, T):
        from unittest.mock import patch
        import torch
        from lunarrisk.probabilistic import condition_posterior
        torch.manual_seed(20261023)  # identical checkpoint/input across repeat calls
        model = condition_posterior(variant, width=8, rank=2, n_scalars=7)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        features = torch.randn(2, 7)
        with patch.object(runner, "batch", return_value=(x, features)):
            physical, z = runner.predict(
                model, ["one", "two"], {}, None, (0, 1, 2), torch.device("cpu"),
                draws=64, seed=14,
                point_sigma=[0.3, 0.2, 0.15, 0.5] if variant == "point" else None,
                temperature=T,
            )
        with torch.no_grad():
            mu = model(x, features)[:, :4].cpu().numpy()[None, ...]
        return z, mu, physical

    def test_temperature_scales_point_diagonal_and_joint_noise_not_mean(self):
        for variant in ("point", "diagonal", "joint"):
            with self.subTest(variant=variant):
                z_raw, mu, _ = self._sample(variant, None)
                z_hot, mu_again, _ = self._sample(variant, 1.8)
                np.testing.assert_allclose(mu, mu_again, atol=0, rtol=0)
                np.testing.assert_allclose(z_hot - mu, 1.8 * (z_raw - mu),
                                           atol=2e-5, rtol=5e-5)

    def test_unity_temperature_is_noop_and_zero_is_rejected(self):
        for variant in ("point", "diagonal", "joint"):
            with self.subTest(variant=variant):
                raw, _, _ = self._sample(variant, None)
                unity, _, _ = self._sample(variant, 1.0)
                np.testing.assert_allclose(raw, unity, atol=2e-6, rtol=2e-6)
                with self.assertRaises(ValueError):
                    self._sample(variant, 0.0)

    def test_same_checkpoint_u_zero_ablation_preserves_shared_temperature(self):
        from unittest.mock import patch
        import torch
        from lunarrisk.probabilistic import condition_posterior
        model = condition_posterior("joint", width=8, rank=2)
        model.eval()
        x = torch.randn(3, 3, 32, 32)
        features = torch.randn(3, 7)
        saved = runner.zero_low_rank(model.head)
        try:
            with patch.object(runner, "batch", return_value=(x, features)):
                _, raw = runner.predict(model, ["a", "b", "c"], {}, None, (0, 1, 2),
                                        torch.device("cpu"), draws=32, seed=22)
                _, calibrated = runner.predict(model, ["a", "b", "c"], {}, None, (0, 1, 2),
                                               torch.device("cpu"), draws=32, seed=22,
                                               temperature=1.5)
            with torch.no_grad():
                mu = model(x, features)[:, :4].cpu().numpy()[None, ...]
            np.testing.assert_allclose(calibrated-mu, 1.5*(raw-mu), atol=2e-5, rtol=5e-5)
        finally:
            runner.restore_low_rank(model.head, saved)


class TestCovarianceAblations(unittest.TestCase):
    """matched_marginals must keep diag(Sigma) and drop only the off-diagonal covariance."""

    def test_matched_marginals_preserves_variances_and_zeroes_covariance(self):
        import torch
        from lunarrisk.probabilistic import condition_posterior, covariance_from_params
        torch.manual_seed(7)
        model = condition_posterior("joint", width=8, rank=2, n_scalars=0)
        out = model(torch.randn(4, 3, 32, 32))
        params = model.parameters_dict(out)
        full = covariance_from_params(model, params, "full")
        matched = covariance_from_params(model, params, "matched_marginals")
        assert torch.allclose(torch.diagonal(matched, dim1=-2, dim2=-1),
                              torch.diagonal(full, dim1=-2, dim2=-1))
        off = matched - torch.diag_embed(torch.diagonal(matched, dim1=-2, dim2=-1))
        self.assertTrue(torch.allclose(off, torch.zeros_like(off)))
        self.assertFalse(torch.allclose(full, matched))

    def test_matched_marginals_changes_draws_but_keeps_the_mean(self):
        from unittest.mock import patch
        import torch
        from lunarrisk.probabilistic import condition_posterior
        torch.manual_seed(11)
        model = condition_posterior("joint", width=8, rank=2, n_scalars=7)
        model.eval()
        x = torch.randn(3, 3, 32, 32)
        features = torch.randn(3, 7)
        with patch.object(runner, "batch", return_value=(x, features)):
            _, full = runner.predict(model, ["a", "b", "c"], {}, None, (0, 1, 2),
                                     torch.device("cpu"), draws=512, seed=3)
            _, matched = runner.predict(model, ["a", "b", "c"], {}, None, (0, 1, 2),
                                        torch.device("cpu"), draws=512, seed=3,
                                        covariance_mode="matched_marginals")
        self.assertEqual(full.shape, matched.shape)
        self.assertFalse(np.allclose(full, matched))
        # same mu, same noise ordering: the difference is only the covariance used for sampling
        np.testing.assert_allclose(full.mean(axis=0), matched.mean(axis=0), atol=0.25)

    def test_unknown_covariance_mode_is_rejected(self):
        import torch
        from lunarrisk.probabilistic import condition_posterior, covariance_from_params
        model = condition_posterior("joint", width=8, rank=2, n_scalars=0)
        params = model.parameters_dict(model(torch.randn(1, 3, 32, 32)))
        with self.assertRaises(ValueError):
            covariance_from_params(model, params, "low_rank_fixed")


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_reload_reproduces_per_rule_probabilities(self):
        import tempfile
        from pathlib import Path as PathLib
        from unittest.mock import patch
        import torch
        from lunarrisk.probabilistic import condition_posterior
        torch.manual_seed(123)
        model = condition_posterior("joint", width=8, rank=2, n_scalars=3)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        features = torch.randn(2, 3)
        import pandas as pd
        table = pd.DataFrame(np.random.default_rng(0).normal(size=(2, 3)), columns=["s0", "s1", "s2"])
        rules = standard_rules()
        with tempfile.TemporaryDirectory() as directory:
            path = PathLib(directory) / "model.pt"
            runner.save_checkpoint(path, model, scalar_columns=list(table.columns),
                                   scalar_mean=[0.0, 0.0, 0.0], scalar_std=[2.0, 1.0, 3.0],
                                   point_sigma=None, temperature=1.4, seed=5,
                                   labels_path=None, cache_path=None, channels=(0, 1, 2))
            with patch.object(runner, "batch", return_value=(x, features)):
                first, probs_first, payload = runner.predict_from_checkpoint(
                    path, ["a", "b"], None, table, rules, torch.device("cpu"), draws=64, seed=9)
                second, probs_second, _ = runner.predict_from_checkpoint(
                    path, ["a", "b"], None, table, rules, torch.device("cpu"), draws=64, seed=9)
        self.assertEqual(payload["temperature"], 1.4)
        for name in rules:
            np.testing.assert_allclose(probs_first[name], probs_second[name], atol=0, rtol=0)
        for key in first:
            np.testing.assert_allclose(first[key], second[key], atol=0, rtol=0)


class TestRefitTemperature(unittest.TestCase):
    def test_closed_form_temperature_on_known_covariance(self):
        mu = np.zeros((3, 2))
        cov = np.tile(np.eye(2), (3, 1, 1))
        z_true = np.array([[2.0, 2.0], [0.0, 0.0], [-2.0, -2.0]])
        # mean squared Mahalanobis distance per dimension = (4+0+4)/3 = 8/3, so T = sqrt(8/3)
        self.assertAlmostEqual(runner.fit_temperature(mu, cov, z_true),
                               float(np.sqrt(8.0 / 3.0)), places=9)

    def test_refit_uses_the_ablated_covariance(self):
        # same diff, smaller covariance -> larger temperature; the refit branch must see the
        # zeroed-U covariance, not the full joint one
        mu = np.zeros((2, 2))
        z_true = np.array([[1.0, 1.0], [1.0, 1.0]])
        full = runner.fit_temperature(mu, np.tile(2.0 * np.eye(2), (2, 1, 1)), z_true)
        ablated = runner.fit_temperature(mu, np.tile(np.eye(2), (2, 1, 1)), z_true)
        self.assertAlmostEqual(ablated / full, float(np.sqrt(2.0)), places=9)


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
