"""Synthetic checks for the paired frozen-feature probe driver."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p33_probe", ROOT / "scripts/p33_frozen_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

PROVENANCE = {
    "input_space": "raw_physical",
    "units": {"elevation": "m", "slope": "degree"},
    "spatial_transform": "none",
    "domain_shift": "MOONSTONE_TERRAIN_NOT_NASA_NATIVE",
    "spatial_policy": "moonstone_native256_v1",
    "meters_per_pixel": 236.90117518866782,
    "ground_sampling": {"grid": "equirectangular", "pixels_per_degree": 128, "lunar_radius_m": 1737400,
                        "east_west_spacing": "latitude_dependent", "nasa_native_60m_equivalence": False},
    "elevation_source": "pinned Moonstone elevation_blended",
    "slope_source": "pinned Moonstone derived slope",
    "units_evidence": "pinned config annotations",
    "nodata_evidence": "published product identity verified bitwise; upstream ingestion not re-executed",
}


class PlanTests(unittest.TestCase):
    def test_repository_plan_yields_the_declared_budget(self):
        plan, budget = probe.load_probe_plan(ROOT / probe.PLAN_PATH)
        self.assertEqual(plan["protocol"], "P33_FROZEN_FEATURE_PROBE_V1")
        self.assertEqual(budget["seeds"], [42, 43, 44])
        self.assertEqual(budget["adaptation"], "frozen")
        self.assertFalse(budget["test_scored"])
        self.assertEqual(budget["epochs"], 20)
        self.assertEqual(budget["patience"], 5)

    def test_alters_to_protocol_policy_or_seeds_are_rejected(self):
        base = json.loads((ROOT / probe.PLAN_PATH).read_text())
        for mutate in ("protocol", "policy", "seeds"):
            plan = copy.deepcopy(base)
            if mutate == "protocol":
                plan["protocol"] = "P33_OTHER"
            elif mutate == "policy":
                plan["design"]["spatial_policy"] = "strict"
            else:
                plan["budget"]["seeds"] = [42]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "plan.json"
                path.write_text(json.dumps(plan))
                with self.assertRaises(ValueError):
                    probe.load_probe_plan(path)


class DatasetGuardTests(unittest.TestCase):
    class Source:
        def __init__(self, split, count=2):
            self.split, self.indices = split, list(range(count))
            self.ids = [str(index) for index in range(count)]
            self.targets = [0] * count

        def close(self):
            pass

    def test_test_split_is_rejected(self):
        source = self.Source(2)
        with self.assertRaises(ValueError):
            probe.ProbeDataset(source, {"0": 0, "1": 0},
                               {"dtm": {"mean": 0.0, "std": 1.0}, "slope": {"mean": 0.0, "std": 1.0}},
                               PROVENANCE, "0" * 64)

    def test_cohort_mismatch_is_rejected(self):
        source = self.Source(0)
        with self.assertRaises(ValueError):
            probe.ProbeDataset(source, {"1": 0},
                               {"dtm": {"mean": 0.0, "std": 1.0}, "slope": {"mean": 0.0, "std": 1.0}},
                               PROVENANCE, "0" * 64)


class HeadTrainingTests(unittest.TestCase):
    def build_cache(self, directory, arms=("random", "pretrained")):
        train_rows, val_rows, width, classes = 48, 24, 8, 49
        rng = np.random.default_rng(7)
        targets = np.repeat(np.arange(classes), 12)[:train_rows]
        val_targets = np.repeat(np.arange(classes), 6)[:val_rows]
        manifest = {"plan": "P33_FROZEN_FEATURE_PROBE_V1", "adaptation": "frozen",
                    "splits": {
                        "train": {"rows": train_rows, "shape": [train_rows, width],
                                  "ordered_ids_sha256": "0" * 64,
                                  "ids": [str(index) for index in range(train_rows)],
                                  "targets": targets.tolist()},
                        "val": {"rows": val_rows, "shape": [val_rows, width],
                                "ordered_ids_sha256": "0" * 64,
                                "ids": [str(200 + index) for index in range(val_rows)],
                                "targets": val_targets.tolist()}},
                    "arms": {arm: {"state_sha256": "0" * 64} for arm in arms}}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        for arm in arms:
            signal = 1.0 if arm == "pretrained" else 0.0
            train = rng.normal(size=(train_rows, width)).astype(np.float32)
            val = rng.normal(size=(val_rows, width)).astype(np.float32)
            if signal:
                train[np.arange(train_rows), targets % width] += 3.0
                val[np.arange(val_rows), val_targets % width] += 3.0
            np.asarray(train).tofile(directory / f"{arm}_train.f32")
            np.asarray(val).tofile(directory / f"{arm}_val.f32")
        return manifest

    def test_paired_head_training_is_deterministic(self):
        budget = probe.validate_budget({
            "epochs": 3, "patience": 2, "batch_size": 8, "lr": 0.001, "weight_decay": 0.01,
            "grad_clip": 1.0, "seeds": [42], "adaptation": "frozen", "precision": "fp32",
            "optimizer": "AdamW", "schedule": "constant",
            "selection": "val_macro_f1_full_earliest_tie", "test_scored": False})
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "features"
            cache.mkdir()
            self.build_cache(cache)
            plan = json.loads((ROOT / probe.PLAN_PATH).read_text())
            first = probe.train_heads(None, plan, budget, cache, Path(directory) / "run_a")
            second = probe.train_heads(None, plan, budget, cache, Path(directory) / "run_b")
            for tag in ("random_seed42", "pretrained_seed42"):
                self.assertEqual(first["arms"][tag]["metrics"]["macro_f1_full"],
                                 second["arms"][tag]["metrics"]["macro_f1_full"])
                self.assertEqual(first["arms"][tag]["prediction_sha256"],
                                 second["arms"][tag]["prediction_sha256"])
            self.assertEqual(first["per_seed_difference"]["42"],
                             first["arms"]["pretrained_seed42"]["metrics"]["macro_f1_full"]
                             - first["arms"]["random_seed42"]["metrics"]["macro_f1_full"])
            self.assertEqual(first["trainable_parameters"], 49 * 8 + 49)
            self.assertFalse(first["test_scored"])
            self.assertEqual(len(first["arms"]["random_seed42"]["history"]), 3)

    def test_heads_start_identical_between_arms(self):
        import torch
        budget = probe.validate_budget({
            "epochs": 1, "patience": 1, "batch_size": 8, "lr": 0.001, "weight_decay": 0.01,
            "grad_clip": 1.0, "seeds": [42], "adaptation": "frozen", "precision": "fp32",
            "optimizer": "AdamW", "schedule": "constant",
            "selection": "val_macro_f1_full_earliest_tie", "test_scored": False})
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "features"
            cache.mkdir()
            self.build_cache(cache)
            plan = json.loads((ROOT / probe.PLAN_PATH).read_text())
            probe.train_heads(None, plan, budget, cache, Path(directory) / "run")
            payloads = [torch.load(Path(directory) / "run" / f"{arm}_seed42.pt", weights_only=True)
                        for arm in ("random", "pretrained")]
            self.assertEqual(payloads[0]["state"].keys(), payloads[1]["state"].keys())


if __name__ == "__main__":
    unittest.main()
