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
    truth = (np.asarray(y_values, dtype=np.float64) > ca.THRESHOLD).astype(int)
    data = {
        "sample_id": [f"s{i}" for i in range(size)],
        "group_id": groups if groups is not None else ["g1"] * (size // 2) + ["g2"] * (size - size // 2),
        "LTM_CODE": ["L1"] * size,
        "y_g10": y_values,
        "y_g15": np.asarray(y_values, dtype=np.float64) / 2,
        "event_g10_gt_0.05": truth,
        "event_g15_gt_0.02": truth,
        "cnn_fusion_g10": fusion_values,
        "cnn_coarse_grid_g10": coarse_values,
        f"cnn_fusion_prob_{PRIMARY_RULE}": fusion_prob,
        f"cnn_coarse_grid_prob_{PRIMARY_RULE}": coarse_prob,
    }
    return pd.DataFrame(data)


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


def protocol(seed=42, ablation=None, provenance=None, channels=None, scalars=None):
    return {
        "seed": seed,
        "input_size": 128,
        "augment": False,
        "augmentation_policy": "none",
        "epochs_max": 60,
        "patience": 10,
        "batch": 16,
        "lr": 0.001,
        "min_nac_valid": 0.99,
        "channels": ["nac_contrast", "nac_texture", "coarse_slope"],
        "scalar_scaling": "train-only",
        "calibration": "cal Platt",
        "rules": {"g10_gt_0.05": {"target": "slope3m_g10", "threshold": 0.05}},
        "composite_rule": "rule",
        "contrast_channel": "formula",
        "ablation_mode": "legacy",
        "ablation": ablation or {"use_geometry": True, "use_nac_photometry": True},
        "variant_channels": channels or {
            variant: [ca.tb.CHANNELS[c] for c in ca.tb.ablation_features(
                variant, use_geometry=(ablation or {}).get("use_geometry", True),
                use_nac_photometry=(ablation or {}).get("use_nac_photometry", True))[0]]
            for variant in ca.tb.CNN_VARIANTS},
        "variant_scalars": scalars or {
            variant: list(ca.tb.ablation_features(
                variant, use_geometry=(ablation or {}).get("use_geometry", True),
                use_nac_photometry=(ablation or {}).get("use_nac_photometry", True))[1])
            for variant in ca.tb.CNN_VARIANTS},
        "provenance": provenance or {"labels": {"sha256": "abc"}},
    }


def metrics_payload(seed=42, ablation=None, provenance=None, channels=None, scalars=None,
                    counts=None):
    values = protocol(seed, ablation, provenance, channels, scalars)
    counts = counts or {"cnn_nac": 100, "cnn_coarse_grid": 100, "cnn_fusion": 100}
    models = {name: {"parameters": {"total": count}} for name, count in counts.items()}
    for name in ca.MODELS:
        models.setdefault(name, {})["test"] = {
            rule: {"metrics": {}, "metrics_raw": {}} for rule in ca.tb.RULES}
    return {"protocol": values, "models": models}


class TestProtocolConsistency(unittest.TestCase):
    def test_identical_configs_pass(self):
        configs = ["full", "no_geo"]
        metrics = {}
        ablation = {}
        for config in configs:
            flags = {"use_geometry": config != "no_geo", "use_nac_photometry": True}
            ablation[config] = flags
            for seed in (42, 43):
                metrics[(config, seed)] = metrics_payload(seed=seed, ablation=flags)
        audit = ca.assert_config_protocol_consistency(metrics, ablation, configs, [42, 43])
        self.assertEqual(audit["configs"]["full"], ablation["full"])
        self.assertEqual(audit["missing"], [])

    def test_hyperparameter_drift_is_fatal(self):
        ablation = {"a": {"use_geometry": True, "use_nac_photometry": True},
                    "b": {"use_geometry": True, "use_nac_photometry": True}}
        metrics = {("a", 42): metrics_payload(42, ablation["a"]),
                   ("b", 42): metrics_payload(42, ablation["b"])}
        metrics[("b", 42)]["protocol"]["batch"] = 32
        with self.assertRaises(SystemExit):
            ca.assert_config_protocol_consistency(metrics, ablation, ["a", "b"], [42])

    def test_provenance_drift_is_fatal(self):
        ablation = {"a": {"use_geometry": True, "use_nac_photometry": True},
                    "b": {"use_geometry": True, "use_nac_photometry": True}}
        metrics = {("a", 42): metrics_payload(42, ablation["a"], {"labels": {"sha256": "abc"}}),
                   ("b", 42): metrics_payload(42, ablation["b"], {"labels": {"sha256": "def"}})}
        with self.assertRaises(SystemExit):
            ca.assert_config_protocol_consistency(metrics, ablation, ["a", "b"], [42])

    def test_channel_list_must_match_ablation_flags(self):
        flags = {"use_geometry": True, "use_nac_photometry": True}
        ablation = {"a": flags}
        wrong = {variant: ["nac_texture"] for variant in ca.tb.CNN_VARIANTS}
        metrics = {("a", 42): metrics_payload(42, flags, channels=wrong)}
        with self.assertRaises(SystemExit):
            ca.assert_config_protocol_consistency(metrics, ablation, ["a"], [42])

    def test_mixing_legacy_and_fixed_mask_is_fatal(self):
        ablation = {"a": {"use_geometry": True, "use_nac_photometry": True},
                    "b": {"use_geometry": True, "use_nac_photometry": True}}
        metrics = {("a", 42): metrics_payload(42, ablation["a"]),
                   ("b", 42): metrics_payload(42, ablation["b"])}
        metrics[("b", 42)]["protocol"]["ablation_mode"] = "fixed-mask"
        with self.assertRaises(SystemExit):
            ca.assert_config_protocol_consistency(metrics, ablation, ["a", "b"], [42])

    def test_fixed_mask_capacity_must_match_across_configs(self):
        ablation = {"a": {"use_geometry": True, "use_nac_photometry": True},
                    "b": {"use_geometry": False, "use_nac_photometry": True}}
        metrics = {("a", 42): metrics_payload(42, ablation["a"]),
                   ("b", 42): metrics_payload(42, ablation["b"])}
        for key in metrics:
            metrics[key]["protocol"]["ablation_mode"] = "fixed-mask"
        metrics[("b", 42)]["models"]["cnn_fusion"]["parameters"]["total"] = 90
        with self.assertRaises(SystemExit):
            ca.assert_config_protocol_consistency(metrics, ablation, ["a", "b"], [42])

    def test_mixed_ablation_flags_within_a_config_are_fatal(self):
        metrics = {
            ("a", 42): metrics_payload(42, {"use_geometry": True, "use_nac_photometry": True}),
            ("a", 43): metrics_payload(43, {"use_geometry": False, "use_nac_photometry": True}),
        }
        with self.assertRaises(SystemExit):
            ca.assert_config_protocol_consistency(metrics, {"a": {"use_geometry": True,
                                                                  "use_nac_photometry": True}},
                                                  ["a"], [42, 43])


class TestProbabilityQuality(unittest.TestCase):
    def _case(self, **override):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        rows = frame(y, np.full(4, 0.5), y, np.full(4, 0.5), np.full(4, 0.5))
        for model in ca.MODELS:
            for rule in ca.tb.RULES:
                rows[f"{model}_prob_{rule}"] = np.full(4, 0.5)
        for variant in ca.tb.CNN_VARIANTS:
            for rule in ca.tb.HEAD_RULES:
                rows[f"cnn_{variant}_raw_event_prob_{rule}"] = np.full(4, 0.5)
        for key, value in override.items():
            rows[key] = value
        metrics = {("cfg", 42): {"models": {model: {"test": {
            rule: {"primary": "raw", "metrics": {}, "metrics_raw": {}} for rule in ca.tb.RULES
        }} for model in ca.MODELS}}}
        return {("cfg", 42): rows}, metrics

    def test_valid_frame_passes(self):
        frames, metrics = self._case()
        audit = ca.assert_probability_quality(frames, metrics, ["cfg"], [42])
        self.assertGreater(audit["checked"], 0)

    def test_nan_probability_is_fatal(self):
        frames, metrics = self._case()
        frames[("cfg", 42)].loc[0, ca.PROBABILITY_COLUMNS["cnn_fusion"]] = np.nan
        with self.assertRaises(SystemExit):
            ca.assert_probability_quality(frames, metrics, ["cfg"], [42])

    def test_out_of_range_probability_is_fatal(self):
        frames, metrics = self._case()
        frames[("cfg", 42)].loc[0, ca.PROBABILITY_COLUMNS["coarse_ridge"]] = 1.5
        with self.assertRaises(SystemExit):
            ca.assert_probability_quality(frames, metrics, ["cfg"], [42])

    def test_missing_raw_event_head_is_fatal(self):
        frames, metrics = self._case()
        frames[("cfg", 42)] = frames[("cfg", 42)].drop(columns=["cnn_nac_raw_event_prob_g10_gt_0.05"])
        with self.assertRaises(SystemExit):
            ca.assert_probability_quality(frames, metrics, ["cfg"], [42])


class TestCohortTable(unittest.TestCase):
    def test_headline_metrics_from_exported_columns(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        frame_rows = frame(y, np.full(4, 0.5), y, np.array([0.01, 0.02, 0.98, 0.99]),
                           np.full(4, 0.5))
        truth = frame_rows["event_g10_gt_0.05"].to_numpy(dtype=float)
        shared_prob = np.array([0.01, 0.02, 0.98, 0.99])
        brier = float(((shared_prob - truth) ** 2).mean())
        for model in ca.MODELS:
            for rule in ca.tb.RULES:
                frame_rows[f"{model}_prob_{rule}"] = np.full(4, 0.5)
            frame_rows[ca.PROBABILITY_COLUMNS[model]] = np.array([0.01, 0.02, 0.98, 0.99])
        for model in ca.MODELS:
            fraction = ca.FRACTION_COLUMNS[model]
            if fraction not in frame_rows.columns:
                frame_rows[fraction] = np.full(4, 0.5)
        frames = {("cfg", 42): frame_rows}
        metrics = {
            ("cfg", 42): {
                "models": {
                    model: {"test": {PRIMARY_RULE: {
                        "metrics": {"mae": 0.1, "brier": brier},
                        "metrics_raw": {"brier": 0.25, "nll": 0.5, "auc": 1.0},
                    }}} for model in ca.MODELS
                }
            }
        }
        table = ca.cohort_table(frames, metrics, ["cfg"], [42])
        self.assertAlmostEqual(table["cfg"]["cnn_fusion"]["mae"]["mean"], 0.1)
        self.assertIn("nll_primary", table["cfg"]["cnn_fusion"])
        self.assertIn("nll_raw", table["cfg"]["cnn_fusion"])
        self.assertEqual(table["cfg"]["cnn_fusion"]["mae"]["std"], 0.0)
        self.assertAlmostEqual(table["cfg"]["cnn_fusion"]["uar20"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
