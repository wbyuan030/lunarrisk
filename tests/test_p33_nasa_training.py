"""Synthetic CPU training/replay checks only; requires an available torch runtime."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from test_p33_nasa_torch import TinyBackbone
    from test_p33_nasa_runner import budget
    from lunarrisk.nasa_transfer.training import (make_arm, train_arm, replay_arm, evaluate,
                                                 state_digest, release_device_memory)

    class TinyTerrain:
        def __init__(self, split):
            self.split = split
        def __len__(self):
            return 4
        def __getitem__(self, index):
            rng = np.random.default_rng(index + (100 if self.split == "val" else 0))
            return {"dtm": rng.normal(size=(1, 4, 4)).astype(np.float32),
                    "slope": rng.normal(size=(1, 4, 4)).astype(np.float32),
                    "target": np.int64(index % 2), "sample_id": self.split + str(index)}


def _fake_loader(backbone):
    with torch.no_grad():
        backbone.model.encoder[0].weight.add_(1.)
    return {"fixture": "synthetic; not a NASA checkpoint"}


@unittest.skipIf(torch is None, "requires compatible torch; tiny synthetic fixture is not official NASA")
class TestNasaTrainingEngine(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = {split: TinyTerrain(split) for split in ("train", "val")}

    def make(self, arm="random", adaptation="frozen"):
        return make_arm(TinyBackbone, _fake_loader, seed=42, arm=arm, adaptation=adaptation)

    def run_arm(self, arm="random", adaptation="frozen"):
        config = budget()
        config["adaptation"] = adaptation
        model, coverage = self.make(arm, adaptation)
        identity = {"fixture": "not science", "seed": 42, "arm": arm}
        report = train_arm(model, self.data, budget=config, seed=42, arm=arm, device="cpu",
                           outdir=self.root / arm, identity=identity, coverage=coverage)
        return config, identity, report, model

    def test_paired_training_has_equal_initialization_order_and_frozen_backbones(self):
        reports = []
        for arm in ("random", "pretrained"):
            config, identity, report, model = self.run_arm(arm)
            reports.append(report)
            self.assertEqual(state_digest(model.backbone.state_dict()), report["backbone_sha256"])
            self.assertEqual(report["best_epoch"], max(range(len(report["history"])), key=lambda i: report["history"][i]["val"]["macro_f1_full"]) + 1)
            self.assertEqual(report["training_visits"], 4 * len(report["history"]))
            self.assertFalse(report["test_scored"])
            self.assertTrue((self.root / arm / "selected.pt").is_file())
        self.assertEqual(reports[0]["initial_task_sha256"], reports[1]["initial_task_sha256"])
        for left, right in zip(reports[0]["history"], reports[1]["history"]):
            self.assertEqual(left["training_order_sha256"], right["training_order_sha256"])

    def test_selected_state_reconstructs_identical_predictions(self):
        config, identity, report, _ = self.run_arm()
        fresh, _ = self.make()
        receipt = replay_arm(fresh, self.data, budget=config, seed=42, device="cpu", original=self.root / "random",
                             outdir=self.root / "replay", identity=identity)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["prediction_sha256"], report["prediction_sha256"])

    def test_affine_adapters_train_with_frozen_backbone(self):
        config, identity, report, _ = self.run_arm(adaptation="input_affine")
        self.assertNotEqual(report["initial_task_sha256"], report["selected_task_sha256"])
        fresh, _ = self.make(adaptation="input_affine")
        replay_arm(fresh, self.data, budget=config, seed=42, device="cpu", original=self.root / "random",
                   outdir=self.root / "replay", identity=identity)

    def test_changed_replay_identity_or_checkpoint_fails(self):
        config, identity, _, _ = self.run_arm()
        model, _ = self.make()
        with self.assertRaisesRegex(ValueError, "identity"):
            replay_arm(model, self.data, budget=config, seed=42, device="cpu", original=self.root / "random",
                       outdir=self.root / "replay", identity={"wrong": True})
        path = self.root / "random/selected.pt"
        path.write_bytes(path.read_bytes() + b"mutation")
        with self.assertRaisesRegex(ValueError, "checksum"):
            replay_arm(model, self.data, budget=config, seed=42, device="cpu", original=self.root / "random",
                       outdir=self.root / "replay", identity=identity)

    def test_test_split_is_not_evaluated_or_trained(self):
        model, coverage = self.make()
        with self.assertRaisesRegex(ValueError, "Test"):
            evaluate(model, self.data["val"], batch_size=2, device=torch.device("cpu"), split="test")
        with self.assertRaises(ValueError):
            train_arm(model, {"train": self.data["train"], "test": self.data["val"]}, budget=budget(), seed=42,
                      arm="random", device="cpu", outdir=self.root / "bad", identity={}, coverage=coverage)
        self.assertFalse((self.root / "bad").exists())

    def test_gpu_cache_released_before_replay_and_live_tensors_block(self):
        with patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "empty_cache") as empty, \
             patch.object(torch.cuda, "memory_allocated", return_value=0):
            release_device_memory("cuda", 0)
            empty.assert_called_once()
        with patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "empty_cache"), \
             patch.object(torch.cuda, "memory_allocated", return_value=1024):
            with self.assertRaisesRegex(RuntimeError, "remain live"):
                release_device_memory("cuda", 0)


if __name__ == "__main__":
    unittest.main()
