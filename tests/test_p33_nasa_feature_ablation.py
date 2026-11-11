"""Synthetic checks for the frozen feature-source ablation driver."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p33_ablation", ROOT / "scripts/p33_feature_ablation.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


class PlanTests(unittest.TestCase):
    def test_repository_plan_is_valid(self):
        plan, budget = ablation.load_plan(ROOT / ablation.PLAN_PATH)
        self.assertEqual(plan["protocol"], "P33_FEATURE_ABLATION_V1")
        self.assertEqual(budget["seeds"], [42, 43, 44])
        self.assertFalse(budget["test_scored"])

    def test_wrong_protocol_is_rejected(self):
        plan = json.loads((ROOT / ablation.PLAN_PATH).read_text())
        plan["protocol"] = "OTHER"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                ablation.load_plan(path)


class WeightTests(unittest.TestCase):
    def test_inverse_sqrt_weights_ignore_absent_classes(self):
        import torch
        targets = np.array([0] * 16 + [1] * 4 + [7] * 1)
        weights = ablation.class_weights(targets, torch).numpy()
        self.assertEqual(len(weights), 49)
        self.assertEqual(float(weights[2]), 0.0)
        self.assertLess(float(weights[0]), float(weights[1]))
        present = weights[[0, 1, 7]]
        self.assertAlmostEqual(float(present.mean()), 1.0, places=6)


class CellTests(unittest.TestCase):
    def features(self, rows, width, labels, signal):
        rng = np.random.default_rng(11)
        values = rng.normal(size=(rows, width)).astype(np.float32)
        if signal:
            values[np.arange(rows), labels % width] += 3.0
        return values

    def test_cell_training_is_deterministic_across_heads(self):
        import torch
        from torch import nn
        rows, width = 48, 8
        labels = np.repeat(np.arange(6), 8)
        features = {("x", "train"): self.features(rows, width, labels, True),
                    ("x", "val"): self.features(24, width, np.repeat(np.arange(6), 4), True)}
        targets = {"train": labels.astype(np.int64), "val": np.repeat(np.arange(6), 4).astype(np.int64)}
        budget = {"epochs": 2, "patience": 2, "batch_size": 8, "lr": 0.001, "weight_decay": 0.01,
                  "grad_clip": 1.0, "seeds": [42]}
        first = ablation.train_cell("linear", features, targets, 42, budget, torch, nn)
        second = ablation.train_cell("linear", features, targets, 42, budget, torch, nn)
        self.assertEqual(first["metrics"]["macro_f1_full"], second["metrics"]["macro_f1_full"])
        self.assertEqual(first["parameters"], 8 * 49 + 49)
        self.assertEqual(first["best_epoch"], second["best_epoch"])
        self.assertTrue(all(entry["val_macro_f1"] <= 1.0 for entry in first["history"]))


if __name__ == "__main__":
    unittest.main()
