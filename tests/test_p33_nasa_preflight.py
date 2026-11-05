"""Subprocess preflight tests with explicitly synthetic config and raw fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from test_p33_nasa_contracts import fixture
from lunarrisk.nasa_transfer.contracts import file_sha256

ROOT = Path(__file__).resolve().parents[1]


class TestNasaPreflightCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw, valid, domains, provenance = fixture()
        self.batch, self.config, self.provenance, self.output = [self.root / name for name in
            ("synthetic.npz", "synthetic.yaml", "provenance.json", "result.json")]
        np.savez(self.batch, **raw, **{"valid_" + key: value for key, value in valid.items()},
                 sample_ids=np.array(["0", "1"]), targets=np.array([43, 43]))
        self.config.write_text(json.dumps({"data": {"domains": domains}}))
        provenance.update(split="train", batch_sha256=file_sha256(self.batch))
        self.provenance.write_text(json.dumps(provenance))
        self.command = [sys.executable, str(ROOT / "scripts/p33_nasa_preflight.py"),
            "--batch", str(self.batch), "--config", str(self.config), "--provenance", str(self.provenance),
            "--config-sha256", file_sha256(self.config), "--output", str(self.output)]

    def run_command(self):
        return subprocess.run(self.command, text=True, capture_output=True, check=False)

    def test_input_audit_is_not_reported_as_model_execution(self):
        result = self.run_command()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["input_contract"], "PASS")
        self.assertEqual(report["official_model_execution"], "NOT_RUN")
        self.assertEqual(report["science_claim"], "NOT_ESTABLISHED")
        self.assertNotIn("arms", report)
        self.assertEqual(report["authoritative_train_membership"]["train_rows"], 5135)
        self.assertEqual(report["sample_ids"], ["0", "1"])

    def test_forged_val_test_ids_and_wrong_target_are_rejected(self):
        # These are synthetic pixels; real historical IDs only exercise the
        # membership gate and do not claim the pixels belong to those samples.
        import gzip
        import csv
        with gzip.open(ROOT / "reports/evidence/p30_20261103/metadata_rows.csv.gz", "rt") as handle:
            records = list(csv.DictReader(handle))
        excluded = [next(row["sample_id"] for row in records if row["stored_split"] == split)
                    for split in ("1", "2")]
        for ids, targets in ((["forged-001", "1"], [43, 43]),
                             ([excluded[0], "1"], [43, 43]),
                             ([excluded[1], "1"], [43, 43]),
                             (["0", "1"], [0, 43])):
            with self.subTest(ids=ids, targets=targets):
                with np.load(self.batch) as original:
                    batch = {key: np.array(original[key]) for key in original.files}
                batch.update(sample_ids=np.array(ids), targets=np.array(targets))
                np.savez(self.batch, **batch)
                provenance = json.loads(self.provenance.read_text())
                provenance["batch_sha256"] = file_sha256(self.batch)
                self.provenance.write_text(json.dumps(provenance))
                result = self.run_command()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("authoritative Train manifest", result.stderr)
                self.assertFalse(self.output.exists())

    def test_tampered_authority_files_are_rejected_without_override_hash(self):
        for flag, source in (("--train-plan", ROOT / "protocols/p31_local_prefix_eval_plan.json"),
                             ("--train-metadata", ROOT / "reports/evidence/p30_20261103/metadata_rows.csv.gz")):
            with self.subTest(flag=flag):
                altered = self.root / "tampered_authority"
                altered.write_bytes(source.read_bytes() + b" ")
                result = subprocess.run(self.command + [flag, str(altered)], text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("SHA256 mismatch", result.stderr)
                self.assertFalse(self.output.exists())

    def test_bytes_or_empty_ids_rejected_before_output_creation(self):
        for ids in (np.array([b"train-1", b"train-2"]), np.array(["", "train-2"])):
            with np.load(self.batch) as original:
                batch = {key: np.array(original[key]) for key in original.files}
            batch["sample_ids"] = ids
            np.savez(self.batch, **batch)
            provenance = json.loads(self.provenance.read_text())
            provenance["batch_sha256"] = file_sha256(self.batch)
            self.provenance.write_text(json.dumps(provenance))
            result = self.run_command()
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.output.exists())

    def test_output_is_not_overwritten(self):
        self.output.write_text("preserve original")
        result = self.run_command()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_text(), "preserve original")

    def test_test_split_and_unverified_batch_fail_closed(self):
        original = json.loads(self.provenance.read_text())
        for key, value in (("split", "test"), ("batch_sha256", "0" * 64)):
            with self.subTest(key=key):
                self.provenance.write_text(json.dumps(dict(original, **{key: value})))
                result = self.run_command()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_config_mutation_and_model_run_without_assets_fail(self):
        self.config.write_text(self.config.read_text() + "\n")
        result = self.run_command()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())
        self.command += ["--forward-backward"]
        result = self.run_command()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --upstream and --checkpoint", result.stderr)


if __name__ == "__main__":
    unittest.main()
