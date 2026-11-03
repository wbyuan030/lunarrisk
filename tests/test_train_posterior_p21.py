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

    def test_latent_matched_marginals_can_change_physical_g15_distribution(self):
        import torch
        from lunarrisk.probabilistic import from_internal
        noise = torch.randn((8192, 2), generator=torch.Generator().manual_seed(7)).numpy()
        correlated = noise @ np.linalg.cholesky([[1.0, 0.9], [0.9, 1.0]]).T
        # Each latent variance is one in both branches, but the product's mean changes.
        def physical(values):
            return from_internal({"z1": values[:, 0], "z2": values[:, 1],
                                  "z3": np.zeros(len(values)), "z4": np.zeros(len(values))})
        full, matched = physical(correlated), physical(noise)
        np.testing.assert_array_equal(full["g10"], matched["g10"])
        self.assertGreater(abs(full["g15"].mean()-matched["g15"].mean()), 0.02)

    def test_unknown_covariance_mode_is_rejected(self):
        import torch
        from lunarrisk.probabilistic import condition_posterior, covariance_from_params
        model = condition_posterior("joint", width=8, rank=2, n_scalars=0)
        params = model.parameters_dict(model(torch.randn(1, 3, 32, 32)))
        with self.assertRaises(ValueError):
            covariance_from_params(model, params, "low_rank_fixed")


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_original_inference_matches_fresh_process_for_all_variants_and_branches(self):
        import json
        import os
        import subprocess
        import tempfile
        import pandas as pd
        import torch
        from lunarrisk.probabilistic import condition_posterior
        torch.manual_seed(123)
        ids = ["a", "b"]
        rng = np.random.default_rng(0)
        cache = {key: rng.normal(size=(3, 32, 32)).astype(np.float32) for key in ids}
        raw = pd.DataFrame(rng.normal(size=(2, 3)), index=ids, columns=["s0", "s1", "s2"])
        mean, std = np.array([0.7, -0.4, 0.9]), np.array([2.0, 1.0, 3.0])
        scaled = (raw - mean) / std
        rules = standard_rules()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez(directory / "inputs.npz", **cache, raw=raw.to_numpy())
            expected = {}
            for variant in ("point", "diagonal", "joint"):
                model = condition_posterior(variant, width=8, rank=2, n_scalars=3).eval()
                sigma = [0.3, 0.2, 0.1, 0.4] if variant == "point" else None
                runner.save_checkpoint(
                    directory / f"{variant}.pt", model, scalar_columns=list(raw.columns),
                    scalar_mean=mean, scalar_std=std, point_sigma=sigma, temperature=1.4,
                    seed=5, labels_path=None, cache_path=None, channels=(0, 1, 2),
                    extra={"temperatures_u0": {"shared": 1.4, "refit": 1.9}})
                branches = [("full", "full", 1.4)]
                if variant == "joint":
                    branches += [("matched", "matched_marginals", 1.4),
                                 ("u0_shared", "full", 1.4), ("u0_refit", "full", 1.9)]
                for branch, mode, temperature in branches:
                    saved = runner.zero_low_rank(model.head) if branch.startswith("u0") else None
                    try:
                        physical, _ = runner.predict(
                            model, ids, cache, scaled, (0, 1, 2), "cpu", draws=128, seed=9,
                            point_sigma=sigma, temperature=temperature, covariance_mode=mode)
                        expected[f"{variant}:{branch}"] = {
                            name: rule.evaluate(physical).mean(axis=0).tolist()
                            for name, rule in rules.items()}
                    finally:
                        if saved is not None:
                            runner.restore_low_rank(model.head, saved)
            child = '''
import sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, sys.argv[1])
import train_posterior_p21 as runner
from pathlib import Path
from lunarrisk.rules import standard_rules
directory = Path(sys.argv[2])
data = np.load(directory / "inputs.npz")
cache = {key: data[key] for key in ("a", "b")}
raw = pd.DataFrame(data["raw"], index=["a", "b"], columns=["s0", "s1", "s2"])
result = {}
for variant in ("point", "diagonal", "joint"):
    branches = ["full"] if variant != "joint" else ["full", "matched", "u0_shared", "u0_refit"]
    for branch in branches:
        _, probabilities, _ = runner.predict_from_checkpoint(
            directory / (variant + ".pt"), ["a", "b"], cache, raw, standard_rules(), "cpu",
            draws=128, seed=9, covariance_mode="matched_marginals" if branch == "matched" else "full",
            ablation=branch if branch.startswith("u0") else "full")
        result[variant + ":" + branch] = {k: v.tolist() for k, v in probabilities.items()}
print(json.dumps(result))
'''
            result = subprocess.run(
                [sys.executable, "-c", child, str(ROOT / "scripts"), str(directory)],
                check=True, capture_output=True, text=True, timeout=60,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            actual = json.loads(result.stdout)
            self.assertEqual(actual, expected)


class TestRunnerExport(unittest.TestCase):
    def test_report_csv_and_checkpoint_replay_use_identical_split_draws(self):
        import contextlib
        import io
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import pandas as pd
        import torch
        rng = np.random.default_rng(11)
        ids = [f"tile_{i}" for i in range(16)]
        splits = np.repeat(["train", "val", "cal", "test"], 4)
        labels = pd.DataFrame({"sample_id": ids, "group_id": [f"g{i}" for i in range(16)],
                               "split": splits})
        for key, column in runner.TARGET_COLUMNS.items():
            labels[column] = rng.uniform(0.1, 0.5, size=16)
        labels["slope3m_g15"] *= labels["slope3m_g10"]
        cache = {key: rng.normal(size=(3, 32, 32)).astype(np.float32) for key in ids}
        raw = pd.DataFrame(rng.normal(size=(16, 2)), index=ids, columns=["s1", "s2"])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            labels.to_csv(directory / "labels.csv", index=False)
            checkpoint = directory / "model.pt"
            with patch.object(runner, "load_baseline_module", return_value=SimpleNamespace(CHANNELS=(0, 1, 2))), \
                 patch.object(runner, "load_dataset", return_value=(labels, cache, raw)), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main(["--variant", "joint", "--device", "cpu", "--seed", "42",
                             "--epochs", "1", "--draws", "64", "--sampling-seed", "99",
                             "--labels", str(directory / "labels.csv"), "--calibrate",
                             "--ablate-u", "--ablate-u-calibration-mode", "both", "--matched-marginals",
                             "--save-checkpoint", str(checkpoint),
                             "--predictions-out", str(directory / "pred.csv"),
                             "--out", str(directory / "report.json")])
            report = json.loads((directory / "report.json").read_text())
            frame = pd.read_csv(directory / "pred.csv")
            _, payload = runner.load_checkpoint(checkpoint)
            self.assertEqual(payload["sample_ids_by_split"]["test"], ids[12:])
            self.assertEqual(payload["sample_groups"][ids[-1]], "g15")
            self.assertEqual(payload["sampling_seed"], 99)
            self.assertIn("refit", payload["temperatures_u0"])
            self.assertTrue(payload["source_sha256"])
            branches = [("", "splits", "full", "full"),
                        ("__u0", "splits_u0", "u0_shared", "full"),
                        ("__u0_refit", "splits_u0_refit", "u0_refit", "full"),
                        ("__matched", "splits_matched_marginals", "full", "matched_marginals")]
            for split in ("train", "val", "cal", "test"):
                subset = frame[frame.split == split]
                split_ids = subset.sample_id.tolist()
                for suffix, metric_key, ablation, mode in branches:
                    _, replay, _ = runner.predict_from_checkpoint(
                        checkpoint, split_ids, cache, raw, standard_rules(), "cpu", draws=64,
                        seed=99, covariance_mode=mode, ablation=ablation)
                    for name in standard_rules():
                        p = subset[f"p_{name}{suffix}"].to_numpy()
                        y = subset[f"event_{name}"].to_numpy()
                        np.testing.assert_array_equal(replay[name], p)
                        self.assertAlmostEqual(float(np.mean((p-y)**2)),
                                               report[metric_key][split]["events"][name]["brier"])


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
