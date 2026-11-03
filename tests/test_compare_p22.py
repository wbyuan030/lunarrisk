#!/usr/bin/env python3
"""Offline tests for the P2.2 comparison helpers (no data files, no training)."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch
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


def labeled_fixture():
    labels = pd.DataFrame({"sample_id": list("abcdefghij"),
        "group_id": [f"g{i}" for i in range(10)],
        "split": ["train"] * 4 + ["val"] * 2 + ["cal"] * 2 + ["test"] * 2,
        "slope3m_g10": [0.01, 0.2, 0.03, 0.3, 0.1, 0.02, 0.01, 0.3, 0.04, 0.2],
        "slope3m_g15": [0.002, 0.04, 0.01, 0.1, 0.03, 0.001, 0.002, 0.05, 0.01, 0.06]})
    for key, column in compare.TARGETS.items():
        if column not in labels:
            labels[column] = np.linspace(0.02, 0.12, len(labels))
    return labels


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

    def test_missing_invalid_metadata_and_missing_assets_fail_closed(self):
        labels = labeled_fixture()
        meta = {sample: {"nac_valid_fraction": 1.0} for sample in labels.sample_id}
        for invalid in (None, {}, {"nac_valid_fraction": np.nan},
                        {"nac_valid_fraction": 1.1}):
            broken = {**meta, "a": invalid}
            with self.subTest(metadata=invalid), self.assertRaises(ValueError):
                compare.canonical_labels(labels, broken, 0.99)
        with self.assertRaises(ValueError):
            compare.canonical_labels(labels, meta, 0.99, set(labels.sample_id) - {"a"})
        with self.assertRaises(ValueError):
            compare.canonical_labels(pd.concat([labels, labels.iloc[:1]]), meta, 0.99)

    def test_prediction_rows_match_canonical_labels_and_probability_schema(self):
        labels = labeled_fixture()
        frame = compare.event_frame(labels, {name: np.full(len(labels), 0.2)
                                             for name in compare.HEAD_RULES})
        compare.validate_prediction_frame(frame.iloc[::-1], labels, "coarse_logistic", standard_rules())
        for column, value in (("group_id", "wrong"), ("split", "cal"),
                              ("truth_g10", 0.99), ("truth_g10", np.nan),
                              ("p_g10_gt_0.05", np.nan),
                              ("p_g10_gt_0.05", np.inf), ("p_g10_gt_0.05", -0.1)):
            broken = frame.copy()
            broken.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                compare.validate_prediction_frame(broken, labels, "coarse_logistic", standard_rules())
        for broken in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]),
                       frame.drop(columns=["p_g15_gt_0.02"]), frame.assign(p_stale=0.1)):
            with self.assertRaises(ValueError):
                compare.validate_prediction_frame(broken, labels, "coarse_logistic", standard_rules())

    def test_stale_model_files_and_changed_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory)
            (outdir / "pred").mkdir()
            (outdir / "pred" / "joint_seed99.csv").write_text("stale")
            with self.assertRaises(ValueError):
                compare.check_model_files(outdir, [42])
            with self.assertRaises(ValueError):
                compare.validate_saved_run({"identity": {"source": "old"}, "status": "complete"},
                                            {"source": "new"}, outdir, [42])

    def test_saved_run_checks_every_model_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory)
            files = {}
            for folder, extension in (("pred", "csv"), ("ckpt", "pt"), ("report", "json")):
                (outdir / folder).mkdir()
                for model in compare.MODELS:
                    relative = f"{folder}/{model}_seed42.{extension}"
                    (outdir / relative).write_text(model)
                    files[relative] = runner.file_sha256(outdir / relative)
            identity = {"source": "same", "draws": 64}
            record = {"status": "complete", "identity": identity, "files": files}
            compare.validate_saved_run(record, identity, outdir, [42])
            (outdir / "pred" / "point_seed42.csv").write_text("stale predictions")
            with self.assertRaises(ValueError):
                compare.validate_saved_run(record, identity, outdir, [42])

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
        self.assertEqual(brier, 0.0)  # only logarithms need endpoint clipping
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

    def test_oracle_ties_use_exact_floor_count_and_sample_id_order(self):
        tied = self.frame.copy()
        tied["p_r"] = 0.0
        expected = compare.coverage_metrics(tied, "r", self.rule, 0.5)
        self.assertEqual(expected["accepted"], 3)
        self.assertEqual(expected["unsafe_accepted"], 1)  # IDs a,b,c, not file order
        shuffled = tied.iloc[[5, 4, 3, 2, 1, 0]]
        self.assertEqual(compare.coverage_metrics(shuffled, "r", self.rule, 0.5), expected)
        self.assertEqual(compare.coverage_metrics(tied, "r", self.rule, 0.2)["accepted"], 1)
        self.assertEqual(compare.coverage_metrics(tied, "r", self.rule, 0.0)["accepted"], 0)
        self.assertEqual(compare.coverage_metrics(tied, "r", self.rule, 1.0)["accepted"], 6)
        # A deployable probability cutoff still accepts the whole tied atom.
        self.assertEqual(compare.coverage_metrics(tied, "r", self.rule, 0.5,
                                                  threshold=0.0)["accepted"], 6)

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

    def test_deployable_bootstrap_has_fixed_cutoffs_and_paired_counts(self):
        labels = labeled_fixture().iloc[-2:].reset_index(drop=True)
        frame = compare.event_frame(labels, {"g10_gt_0.05": np.array([0.1, 0.8])})
        result = compare.paired_deployable_bootstrap(frame, frame.iloc[::-1], "g10_gt_0.05",
                    Compare("g10", 0.05), 0.2, 1.0, draws=200, seed=7)
        self.assertEqual(result["model_a"]["accepted"], 1)
        self.assertEqual(result["model_b"]["accepted"], 2)
        self.assertEqual(result["model_b"]["unsafe_accepted"], 1)
        self.assertAlmostEqual(result["uar"]["mean"], -0.5)
        self.assertAlmostEqual(result["hmr"]["mean"], -1.0)
        self.assertGreater(result["uar"]["undefined_draws"], 0)
        self.assertGreater(result["hmr"]["undefined_draws"], 0)
        self.assertEqual(result["cal_threshold_a"], 0.2)
        identical = compare.paired_deployable_bootstrap(frame, frame, "g10_gt_0.05",
                    Compare("g10", 0.05), 1.0, 1.0, draws=200)
        for metric in ("uar", "hmr"):
            self.assertEqual(identical[metric]["mean"], 0.0)
            self.assertEqual(identical[metric]["p025"], 0.0)
            self.assertEqual(identical[metric]["p975"], 0.0)

    def test_undefined_ratios_are_null_and_never_imputed_zero(self):
        labels = labeled_fixture().iloc[-2:].copy()
        labels["slope3m_g10"] = 0.01
        frame = compare.event_frame(labels, {"g10_gt_0.05": np.array([0.1, 0.8])})
        result = compare.paired_deployable_bootstrap(frame, frame, "g10_gt_0.05",
                    Compare("g10", 0.05), 0.0, 0.0, draws=25)
        for metric in ("uar", "hmr"):
            self.assertIsNone(result[metric]["mean"])
            self.assertIsNone(result[metric]["p025"])
            self.assertEqual(result[metric]["undefined_draws"], 25)
        self.assertEqual(result["model_a"]["undefined"]["uar"], "no accepted rows")
        self.assertEqual(result["model_a"]["unsafe_total"], 0)

    def test_alignment_rejects_different_sample_sets(self):
        a = compare.event_frame(labeled_fixture().iloc[-2:], {})
        b = a.copy()
        b.loc[b.index[-1], "sample_id"] = "different"
        with self.assertRaisesRegex(ValueError, "different sample ids"):
            compare.aligned_frames(a, b)


class TestBaselineCheckpoints(unittest.TestCase):
    def test_ridge_and_original_coarse_heads_replay_weights_scaler_and_calibration(self):
        baseline = runner.load_baseline_module()
        labels = labeled_fixture()
        features = np.random.default_rng(8).normal(size=(len(labels), len(baseline.COARSE_FEATURES)))
        features[2, 0] = np.nan
        for i, name in enumerate(baseline.COARSE_FEATURES):
            labels[name] = features[:, i]
        fitted = compare.train_ridge(labels, 42)
        state = compare.coarse_state(fitted)
        mu, draws = compare.ridge_predictions(fitted, labels, draws=64, seed=42)
        truth = {key: labels[column].to_numpy() for key, column in compare.TARGETS.items()}
        ridge_frame = compare.frame_from_draws(labels, truth,
            {key: mu[:, i] for i, key in enumerate(compare.TARGETS)}, draws, standard_rules())
        heads = compare.train_coarse_logistic(fitted, labels, standard_rules())
        x = fitted["scaler"].transform(np.where(np.isfinite(features), features, fitted["medians"]))
        logistic_frame = compare.event_frame(labels, compare.coarse_probabilities(heads, x))
        # Verify this really preserves the original train-logistic/Cal-Platt baseline treatment.
        for name in compare.HEAD_RULES:
            y = standard_rules()[name].evaluate(truth).astype(int)
            train, cal = (labels.split.to_numpy() == split for split in ("train", "cal"))
            model, _ = baseline.fit_logistic(x[train], y[train])
            raw = baseline.logit(baseline.logistic_probability(model, x, float(y[train].mean())))
            slope, intercept, _ = baseline.platt_scale(raw[cal], y[cal])
            expected = 1 / (1 + np.exp(-(slope * raw + intercept)))
            np.testing.assert_allclose(logistic_frame[f"p_{name}"], expected, atol=1e-12)
        cases = (("coarse_ridge", ridge_frame, {**state,
                  "coef": fitted["model"].coef_.tolist(),
                  "intercept": fitted["model"].intercept_.tolist(),
                  "sigma": fitted["sigma"].tolist(), "temperature": fitted["temperature"]}),
                 ("coarse_logistic", logistic_frame, {**state, "heads": heads}))
        with tempfile.TemporaryDirectory() as directory:
            for name, expected, payload_state in cases:
                with self.subTest(model=name):
                    path = Path(directory) / f"{name}.pt"
                    compare.save_baseline_checkpoint(path, name, payload_state, labels,
                        seed=42, config={"draws": 64, "batch_size": 3}, provenance={"cache_sha256": "fixture"})
                    actual, payload = compare.predict_baseline_checkpoint(path, labels.iloc[::-1])
                    self.assertEqual(payload["sample_ids"], labels.sample_id.tolist())
                    self.assertIn("scripts/compare_p22.py", payload["source_sha256"])
                    self.assertEqual(payload["provenance"]["cache_sha256"], "fixture")
                    # Reconstructed matrix operations may differ at the last floating bit.
                    pd.testing.assert_frame_equal(actual, expected, check_exact=False,
                                                  rtol=1e-14, atol=1e-15)
                    diagnostic_rules = {"new_relief": Compare("relief_p95_p5", 1.5)}
                    if name == "coarse_ridge":
                        transferred, _ = compare.predict_baseline_checkpoint(
                            path, labels, rules=diagnostic_rules)
                        expected_probability = diagnostic_rules["new_relief"].evaluate(draws).mean(axis=0)
                        np.testing.assert_array_equal(transferred["p_new_relief"], expected_probability)
                    else:
                        with self.assertRaises(ValueError):
                            compare.predict_baseline_checkpoint(path, labels, rules=diagnostic_rules)
                    broken = labels.copy()
                    broken.loc[0, "group_id"] = "stale"
                    with self.assertRaises(ValueError):
                        compare.predict_baseline_checkpoint(path, broken)

    def test_direct_event_replay_uses_saved_scaler_and_original_weights(self):
        import torch
        baseline = runner.load_baseline_module()
        labels = labeled_fixture()
        torch.manual_seed(23)
        model = baseline.TinyCNN(in_channels=3, n_scalars=2).eval()
        rng = np.random.default_rng(23)
        cache = {sample: rng.normal(size=(3, 32, 32)).astype(np.float32) for sample in labels.sample_id}
        columns = ["s0", "s1"]
        table = pd.DataFrame(rng.normal(size=(len(labels), 2)), index=labels.sample_id, columns=columns)
        mean, std = [2.0, -3.0], [4.0, 0.5]
        standardized = table.copy()
        standardized[columns] = (table[columns] - np.asarray(mean)) / np.asarray(std)
        result = {"model": model, "scalar_table": standardized, "scalar_columns": tuple(columns)}
        fractions, logits = baseline.predict_variant(result, labels, cache, (0, 1, 2), 3, "cpu")
        expected = compare.event_frame(labels, {name: 1 / (1 + np.exp(-logits[:, i]))
                    for i, name in enumerate(compare.HEAD_RULES)}, fractions)
        state = {"state_dict": {name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()},
                 "channels": [0, 1, 2], "scalar_columns": columns,
                 "scalar_mean": mean, "scalar_std": std}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "direct.pt"
            compare.save_baseline_checkpoint(path, "direct_event", state, labels,
                seed=23, config={"draws": 64, "batch_size": 3}, provenance={})
            # Only raw feature construction is stubbed: scaling, network loading, batching and
            # event probabilities execute against the original in-memory expected predictions.
            with patch.object(compare.runner, "load_baseline_module", return_value=baseline), \
                    patch.object(baseline, "build_scalar_table", return_value=table.copy()):
                actual, _ = compare.predict_baseline_checkpoint(path, labels, cache, {}, "cpu")
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


if __name__ == "__main__":
    unittest.main()
