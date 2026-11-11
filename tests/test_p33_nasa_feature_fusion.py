"""Synthetic checks for the feature fusion driver."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p33_fusion", ROOT / "scripts/p33_feature_fusion.py")
fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion)


class PlanTests(unittest.TestCase):
    def test_repository_plan_is_valid(self):
        plan, budget = fusion.load_plan(ROOT / fusion.PLAN_PATH)
        self.assertEqual(plan["protocol"], "P33_FEATURE_FUSION_V1")
        self.assertEqual(budget["seeds"], [42, 43, 44])
        self.assertFalse(budget["test_scored"])

    def test_wrong_protocol_is_rejected(self):
        plan = json.loads((ROOT / fusion.PLAN_PATH).read_text())
        plan["protocol"] = "OTHER"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                fusion.load_plan(path)


class MetricTests(unittest.TestCase):
    def test_absent_class_stays_in_macro_denominator(self):
        probabilities = np.array([[0.9, 0.1, 0.0, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0, 0.0]])
        targets = np.array([0, 1])
        result = fusion.macro_metrics(probabilities, targets, 5)
        self.assertAlmostEqual(result["macro_f1_full"], 2.0 / 5, places=9)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["support"], [1, 1, 0, 0, 0])

    def test_invalid_probabilities_are_rejected(self):
        for probabilities in (np.array([[0.5, 0.6]]), np.array([[np.nan, 1.0]]), np.zeros((2, 5))):
            with self.assertRaises(ValueError):
                fusion.macro_metrics(probabilities, np.zeros(len(probabilities), dtype=np.int64), 5)


class CellTests(unittest.TestCase):
    def test_cell_training_is_deterministic(self):
        import torch
        from torch import nn
        rows, width, classes = 48, 8, 5
        rng = np.random.default_rng(3)
        labels = np.repeat(np.arange(classes), 8) if rows == 40 else np.repeat(np.arange(6), 8)[:rows]
        labels = labels % classes
        features = {("x", "train"): rng.normal(size=(rows, width)).astype(np.float32),
                    ("x", "val"): rng.normal(size=(30, width)).astype(np.float32)}
        targets = {"train": labels.astype(np.int64), "val": np.repeat(np.arange(classes), 6).astype(np.int64)}
        budget = {"epochs": 3, "patience": 2, "batch_size": 8, "lr": 0.001, "weight_decay": 0.01,
                  "grad_clip": 1.0, "seeds": [42]}
        first = fusion.train_cell("linear", features, targets, 42, budget, classes, torch, nn)
        second = fusion.train_cell("linear", features, targets, 42, budget, classes, torch, nn)
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["best_epoch"], second["best_epoch"])
        self.assertEqual(first["parameters"], width * classes + classes)
        self.assertTrue(0.0 <= first["metrics"]["macro_f1_full"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
