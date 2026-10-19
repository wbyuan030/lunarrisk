#!/usr/bin/env python3
"""Offline tests for the LROC NAC CDR I/O probe (synthetic PDS3 image only)."""
from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("probe_nac_cdr_io", ROOT / "scripts" / "probe_nac_cdr_io.py")
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def build_image(path: Path, lines: int = 64, samples: int = 48, record_bytes: int = 512,
                scale: float = 3.05185e-05, null: int = -32768, seed: int = 7) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    label = "\r\n".join([
        "PDS_VERSION_ID = PDS3",
        "RECORD_TYPE = FIXED_LENGTH",
        f"RECORD_BYTES = {record_bytes}",
        f"FILE_RECORDS = {(lines * samples * 2) // record_bytes + 2}",
        "LABEL_RECORDS = 1",
        "^IMAGE = 2",
        'DATA_SET_ID = "LRO-L-LROC-3-CDR-V1.0"',
        'PRODUCT_ID = "M153639872LC"',
        'MISSION_PHASE_NAME = "SCIENCE MISSION"',
        'UNIT = "Scaled I/F"',
        "ORBIT_NUMBER = 7776",
        "START_TIME = 2011-03-01T17:30:05.040",
        f"LINES = {lines}",
        f"LINE_SAMPLES = {samples}",
        "SAMPLE_BITS = 16",
        "SAMPLE_TYPE = LSB_INTEGER",
        f"SCALING_FACTOR = {scale!r}",
        f"NULL = {null}",
        "END",
        "",
    ]).encode("latin-1")
    # pad the label to exactly RECORD_BYTES so the image starts at the next record
    padding = record_bytes - (len(label) % record_bytes)
    payload = label + b" " * padding
    assert len(payload) % record_bytes == 0, len(payload)
    pixels = rng.integers(1000, 4000, size=(lines, samples), dtype=np.uint16)
    pixels[0, :4] = np.uint16(null & 0xFFFF)
    raw = payload + pixels.astype("<u2").tobytes()
    path.write_bytes(raw)
    return pixels, {"lines": lines, "samples": samples, "record_bytes": record_bytes}


class TestLabelParsing(unittest.TestCase):
    def test_label_pointer_and_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.IMG"
            build_image(path)
            pixels, label = probe.load_image(path)
            self.assertEqual(label["image_record"], 2)
            self.assertEqual(label["pixel_offset"], 512)
            self.assertEqual(label["lines"], 64)
            self.assertEqual(label["line_samples"], 48)
            self.assertEqual(label["sample_bits"], 16)
            self.assertEqual(label["null_dn"], 32768)  # -32768 wrapped to uint16
            self.assertAlmostEqual(label["scaling_factor"], 3.05185e-05, places=12)
            self.assertEqual(label["product_id"], "M153639872LC")
            self.assertEqual(pixels.shape, (64, 48))
            self.assertEqual(label["bytes_after_label"], 0)

    def test_missing_pointer_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y.IMG"
            build_image(path)
            raw = path.read_bytes().replace(b"^IMAGE = 2", b"^UNKNOWN = 9")
            path.write_bytes(raw)
            with self.assertRaises(KeyError):
                probe.load_image(path)

    def test_truncated_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "z.IMG"
            build_image(path)
            raw = path.read_bytes()
            path.write_bytes(raw[: len(raw) // 2])
            with self.assertRaises(ValueError):
                probe.load_image(path)


class TestStatistics(unittest.TestCase):
    def test_null_and_i_f_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.IMG"
            build_image(path)
            pixels, label = probe.load_image(path)
            stats = probe.statistics(pixels, label)
            self.assertAlmostEqual(stats["valid_fraction"], 1 - 4 / (64 * 48), places=5)
            self.assertEqual(stats["saturated_fraction"], 0.0)
            self.assertGreater(stats["if_percentiles"]["p99"], stats["if_percentiles"]["p50"])
            self.assertEqual(stats["shape"], [64, 48])

    def test_main_writes_report(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name in ("b.IMG", "c.IMG"):
                build_image(base / name, seed=11)
            out = base / "probe.json"
            code = probe.main([str(base / "b.IMG"), str(base / "c.IMG"), "--out", str(out)])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(report["images"]), 2)
            self.assertIn("stereo_overlap", report)
            self.assertIn("label_sha256", report["images"][0]["label"])


if __name__ == "__main__":
    unittest.main()
