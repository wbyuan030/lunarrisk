"""Diagnostic counts must not confuse observed zeros and missing terrain."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("p33_audit", Path(__file__).resolve().parents[1] / "scripts/p33_audit_terrain_cohort.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TerrainAuditTests(unittest.TestCase):
    def test_missing_zero_and_candidates_stay_distinct(self):
        values = np.ones((2, 256, 256), dtype=np.float32)
        values[0, 0, :6] = [0, np.nan, np.inf, -np.inf, -9999, 65535]
        before = values.copy()
        result = audit.summarize(values)
        self.assertEqual(result["elevation"]["finite"], 65533)
        for key in ("zero", "nan", "posinf", "neginf", "candidate_-9999", "candidate_65535"):
            self.assertEqual(result["elevation"][key], 1)
        self.assertEqual(result["slope"]["finite"], 65536)
        np.testing.assert_array_equal(values, before)

    def test_all_missing_has_no_fabricated_range(self):
        result = audit.summarize(np.full((2, 256, 256), np.nan, dtype=np.float32))
        self.assertIsNone(result["elevation"]["min"])
        self.assertIsNone(result["slope"]["max"])
        self.assertEqual(result["elevation"]["zero"], 0)

    def test_reject_wrong_geometry_and_dtype(self):
        for values in (np.zeros((2, 256, 256), dtype=np.float64), np.zeros((28, 256, 256), dtype=np.float32)):
            with self.assertRaises(ValueError):
                audit.summarize(values)


if __name__ == "__main__":
    unittest.main()
