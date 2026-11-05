"""Pure orchestration tests; fake receipts/data never claim actual NASA execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p33_pair_test_module", ROOT / "scripts/p33_train_pair.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from lunarrisk.nasa_transfer.contracts import file_sha256


def budget():
    return {"epochs": 2, "patience": 1, "batch_size": 2, "lr": .001, "weight_decay": .01,
            "grad_clip": 1., "seeds": [42], "adaptation": "frozen", "precision": "fp32",
            "optimizer": "AdamW", "schedule": "constant", "selection": "val_macro_f1_full_earliest_tie",
            "test_scored": False}


def fake_preflight():
    return {"kind": "ENGINEERING_PREFLIGHT_NOT_TRANSFER_SCORE", "official_model_execution": "PASS",
            "input_contract": "PASS", "config_sha256": "a" * 64, "device": "cpu", "torch_version": "synthetic",
            "nasa_code_revision": runner.NASA_CODE_REVISION, "nasa_model_revision": runner.NASA_MODEL_REVISION,
            "adaptation": "frozen_probe", "checkpoint_coverage": {"checkpoint_sha256": runner.NASA_CHECKPOINT_SHA256,
                "checkpoint_bytes": runner.NASA_CHECKPOINT_BYTES, "loaded_inference_fraction": 1.0},
            "arms": {arm: {"forward_backward": "PASS"} for arm in ("random", "pretrained")},
            "authoritative_train_membership": {"train_rows": 5135, "train_plan_sha256": runner.TRAIN_PLAN_SHA256,
                                               "metadata_archive_sha256": runner.METADATA_ARCHIVE_SHA256}}


class TestNasaPairRunner(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.preflight = self.root / "preflight.json"
        self.preflight.write_text(json.dumps(fake_preflight()))
        self.budget = self.root / "budget.json"
        self.budget.write_text(json.dumps(budget()))
        self.common = []
        for name in ("data", "metadata", "range-report", "nasa-config", "nasa-checkpoint", "provenance", "upstream"):
            self.common += ["--" + name, str(self.root / name)]
        self.common += ["--preflight", str(self.preflight), "--manifest", str(self.manifest),
                        "--nasa-config-sha256", "a" * 64, "--device", "cpu"]

    def test_preflight_requires_actual_model_device_and_adaptation(self):
        runner.validate_preflight(fake_preflight(), "a" * 64, "frozen", "cpu")
        for key, value in (("official_model_execution", "NOT_RUN"), ("device", "cuda"), ("adaptation", "input_affine"),
                           ("authoritative_train_membership", {"train_rows": 5135})):
            report = fake_preflight()
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_preflight(report, "a" * 64, "frozen", "cpu")

    def test_lock_scans_before_writing_and_records_all_budget_fields(self):
        datasets = {key: MagicMock() for key in ("train", "val")}
        with patch.object(runner, "runtime_identity", return_value={"torch": "synthetic"}), \
             patch.object(runner, "sources", return_value={"fake": "source"}), \
             patch.object(runner, "asset_identity", return_value={"synthetic": True}), \
             patch.object(runner, "load_data", return_value=(datasets, {"synthetic": True})), \
             patch("lunarrisk.nasa_transfer.dataset.scan_datasets", return_value={"pixel_scan": "PASS"}) as scan:
            runner.main(["--mode", "lock", *self.common, "--budget", str(self.budget)])
            scan.assert_called_once_with(datasets)
            for dataset in datasets.values():
                dataset.close.assert_called_once()
            plan = json.loads(self.manifest.read_text())
            self.assertEqual(plan["budget"], budget())
            self.assertEqual(plan["status"], "LOCKED_BEFORE_PAIR_SCORES")
            self.assertFalse(plan["test_scored"])
            with self.assertRaises(FileExistsError):
                runner.main(["--mode", "lock", *self.common, "--budget", str(self.budget)])

    def test_failed_scan_or_source_drift_does_not_lock_manifest(self):
        for failure in ("scan", "source"):
            datasets = {key: MagicMock() for key in ("train", "val")}
            with patch.object(runner, "runtime_identity", return_value={"torch": "synthetic"}), \
                 patch.object(runner, "sources", side_effect=[{"a": 1}, {"a": 2}]), \
                 patch.object(runner, "asset_identity", return_value={}), \
                 patch.object(runner, "load_data", return_value=(datasets, {})), \
                 patch("lunarrisk.nasa_transfer.dataset.scan_datasets", side_effect=ValueError("scan failed") if failure == "scan" else None):
                with self.assertRaises(ValueError):
                    runner.main(["--mode", "lock", *self.common, "--budget", str(self.budget)])
                self.assertFalse(self.manifest.exists())
                for dataset in datasets.values():
                    dataset.close.assert_called_once()

    def test_run_requires_exact_manifest_digest_before_data_or_torch(self):
        self.manifest.write_text("{}")
        with patch.object(runner, "load_data") as load:
            for extra in ([], ["--manifest-sha256", "0" * 64]):
                with self.assertRaisesRegex(ValueError, "manifest SHA256"):
                    runner.main(["--mode", "run", *self.common, "--outdir", str(self.root / "run"), *extra])
            load.assert_not_called()

    def test_run_cannot_override_budget_or_pick_favorable_seed(self):
        for extra in (["--budget", str(self.budget)], ["--seed", "42"]):
            with self.assertRaises(ValueError):
                runner.main(["--mode", "run", *self.common, "--outdir", str(self.root / "run"), *extra])

    def test_replay_command_handles_equals_options_and_absolute_paths(self):
        args = runner.parser().parse_args(["--mode=run", *self.common, "--outdir=relative-result", "--manifest-sha256", "b" * 64])
        args.data = Path("relative-data")
        command = runner.replay_command(args, Path("relative-original"), Path("relative-replay"), 42, "pretrained")
        parsed = runner.parser().parse_args(command[2:])
        self.assertEqual(parsed.mode, "replay")
        self.assertEqual(parsed.seed, 42)
        self.assertEqual(parsed.arm, "pretrained")
        for name in ("data", "outdir", "run_dir", "metadata", "manifest"):
            self.assertTrue(getattr(parsed, name).is_absolute(), name)
        self.assertEqual(parsed.manifest_sha256, "b" * 64)


if __name__ == "__main__":
    unittest.main()
