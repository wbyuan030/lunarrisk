"""Synthetic checks for the composition label extraction and regression driver."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


labels_mod = load("p33_labels", "p33_composition_labels.py")
regression = load("p33_regression", "p33_composition_regression.py")


class RegressionMetricTests(unittest.TestCase):
    def test_perfect_prediction_scores_zero_error(self):
        target = np.array([1.0, 2.0, 3.0])
        result = regression.regression_metrics(target, target, 0.0, 1.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["r2"], 1.0, places=9)

    def test_constant_target_has_no_r2(self):
        target = np.array([2.0, 2.0, 2.0])
        result = regression.regression_metrics(np.array([2.5, 2.5, 2.5]), target, 0.0, 1.0)
        self.assertIsNone(result["r2"])
        self.assertAlmostEqual(result["rmse"], 0.5, places=9)

    def test_original_scale_is_restored(self):
        prediction = np.array([0.0, 1.0])
        target = np.array([0.0, 1.0])
        result = regression.regression_metrics(prediction, target, mean=5.0, std=2.0)
        self.assertEqual(result["rmse"], 0.0)


class PlanTests(unittest.TestCase):
    def test_repository_plan_is_valid(self):
        plan, budget = regression.load_plan(ROOT / regression.PLAN_PATH)
        self.assertEqual(plan["protocol"], "P33_COMPOSITION_REGRESSION_V1")
        self.assertEqual(budget["seeds"], [42, 43, 44])
        self.assertFalse(budget["test_scored"])

    def test_single_seed_plan_is_rejected(self):
        plan = json.loads((ROOT / regression.PLAN_PATH).read_text())
        plan["budget"]["seeds"] = [42]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                regression.load_plan(path)


class CellTests(unittest.TestCase):
    def test_regression_cell_is_deterministic(self):
        import torch
        from torch import nn
        rows, width = 48, 6
        rng = np.random.default_rng(5)
        features = {("x", "train"): rng.normal(size=(rows, width)).astype(np.float32),
                    ("x", "val"): rng.normal(size=(24, width)).astype(np.float32)}
        targets = {"train": rng.normal(size=(rows, 2)).astype(np.float64),
                   "val": rng.normal(size=(24, 2)).astype(np.float64)}
        scales = {"feo": (0.07, 0.05), "tio2": (0.01, 0.013)}
        budget = {"epochs": 3, "patience": 2, "batch_size": 8, "lr": 0.001, "weight_decay": 0.01,
                  "grad_clip": 1.0, "seeds": [42]}
        first = regression.train_cell(features, targets, scales, 42, budget, torch, nn)
        second = regression.train_cell(features, targets, scales, 42, budget, torch, nn)
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["parameters"], width * 2 + 2)


class LabelGuardTests(unittest.TestCase):
    def test_grid_constants_match_the_published_product(self):
        self.assertEqual(labels_mod.GRID, (23040, 46080))
        self.assertEqual(labels_mod.PATCH, 256)


if __name__ == "__main__":
    unittest.main()
