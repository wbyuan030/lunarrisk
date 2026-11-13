"""Synthetic checks for the latitude stratification and block gating."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p33_stratified", ROOT / "scripts/p33_fusion_stratified.py")
stratified = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stratified)


class PlanTests(unittest.TestCase):
    def test_repository_plan_is_valid(self):
        plan, budget = stratified.load_plan(ROOT / stratified.PLAN_PATH)
        self.assertEqual(plan["protocol"], "P33_FUSION_STRATIFIED_V1")
        self.assertEqual(budget["seeds"], [42, 43, 44])
        self.assertFalse(budget["test_scored"])

    def test_wrong_protocol_is_rejected(self):
        plan = json.loads((ROOT / stratified.PLAN_PATH).read_text())
        plan["protocol"] = "OTHER"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                stratified.load_plan(path)


class BandTests(unittest.TestCase):
    def test_bands_partition_the_val_rows(self):
        probabilities = np.full((6, 49), 1.0 / 49)
        targets = np.zeros(6, dtype=np.int64)
        latitudes = np.array([10.0, 20.0, 35.0, 45.0, 70.0, 85.0])
        rows = stratified.band_metrics(probabilities, targets, latitudes)
        for name, low, high in stratified.BANDS:
            members = (latitudes >= low) & (latitudes < high)
            self.assertEqual(rows[name]["rows"], int(members.sum()))
        self.assertEqual(rows["overall"]["rows"], 6)

    def test_empty_band_is_reported_as_none(self):
        probabilities = np.full((2, 49), 1.0 / 49)
        targets = np.zeros(2, dtype=np.int64)
        rows = stratified.band_metrics(probabilities, targets, np.array([10.0, 20.0]))
        self.assertIsNone(rows["polar"])


class GateTests(unittest.TestCase):
    def test_gate_starts_at_one_and_is_learnable(self):
        import torch
        from torch import nn
        model = stratified.make_model(1024, True, torch, nn)
        self.assertEqual([float(v) for v in model.weight.detach()], [1.0, 1.0])
        self.assertTrue(model.weight.requires_grad)
        plain = stratified.make_model(1024, False, torch, nn)
        self.assertFalse(hasattr(plain, "weight") and not isinstance(plain.weight, torch.Tensor))

    def test_gate_matches_plain_linear_at_unit_weight(self):
        import torch
        from torch import nn
        torch.manual_seed(0)
        plain = stratified.make_model(1024, False, torch, nn)
        gated = stratified.make_model(1024, True, torch, nn)
        gated.inner.load_state_dict(plain.state_dict())
        values = torch.randn(3, 1024)
        self.assertTrue(torch.equal(gated(values), plain(values)))


if __name__ == "__main__":
    unittest.main()
