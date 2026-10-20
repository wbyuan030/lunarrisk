#!/usr/bin/env python3
"""Offline tests for the P20 coverage QA (synthetic masks only)."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("coverage_qa_p20", ROOT / "scripts" / "coverage_qa_p20.py")
cov = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cov
spec.loader.exec_module(cov)


class TestLoad(unittest.TestCase):
    def test_fill_detected_as_most_frequent_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.raw"
            grid = np.full((4, 4), 0.5, dtype="<f4")   # fill, 12 of 16 pixels
            grid[0, 0] = 0.9
            grid[1, 1] = 0.8
            grid[2, 2] = 0.7
            grid[3, 3] = 0.6
            grid.tofile(path)
            image, fill = cov.load(path, 4, 4)
            self.assertEqual(fill[0], 0.5)
            self.assertAlmostEqual(fill[1], 12 / 16)
            self.assertEqual(int(np.isfinite(image).sum()), 4)
            self.assertTrue(np.isnan(image[0, 3]))

    def test_wrong_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.raw"
            np.zeros(10, dtype="<f4").tofile(path)
            with self.assertRaises(AssertionError):
                cov.load(path, 4, 4)


if __name__ == "__main__":
    unittest.main()
