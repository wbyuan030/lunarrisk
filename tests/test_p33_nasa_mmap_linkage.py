"""Synthetic checks for the published-mmap terrain linkage audit."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p33_mmap_linkage", ROOT / "scripts/p33_audit_mmap_linkage.py")
linkage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linkage)


def packed(mask_2d):
    return np.packbits(np.asarray(mask_2d, dtype=bool).ravel())


class ZScoreTests(unittest.TestCase):
    def test_float32_convention_roundtrip(self):
        mean = np.float32(-502.1981201171875)
        std = np.float32(2237.179443359375)
        raw = np.array([[mean, mean + std, np.float32(0.0)]], dtype=np.float32)
        z = linkage.zscore(raw, -502.1981201171875, 2237.179443359375)
        self.assertEqual(z.dtype, np.float32)
        self.assertEqual(float(z[0, 0]), 0.0)
        self.assertAlmostEqual(float(z[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(z[0, 2]), 0.224, places=3)

    def test_nonfinite_becomes_zero_after_norm(self):
        raw = np.array([[np.nan, np.inf, -np.inf, 5.0]], dtype=np.float32)
        z = linkage.zscore(raw, 1.0, 2.0)
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(float(z[0, 0]), 0.0)
        self.assertEqual(float(z[0, 1]), 0.0)
        self.assertEqual(float(z[0, 2]), 0.0)
        self.assertEqual(float(z[0, 3]), 2.0)

    def test_rejects_non_float32_and_bad_stats(self):
        for raw in (np.zeros((2, 2), dtype=np.float64), np.zeros((2, 2), dtype=np.int16)):
            with self.assertRaises(ValueError):
                linkage.zscore(raw, 0.0, 1.0)
        for mean, std in ((np.nan, 1.0), (0.0, 0.0), (0.0, -1.0)):
            with self.assertRaises(ValueError):
                linkage.zscore(np.zeros((2, 2), dtype=np.float32), mean, std)


class CompareTests(unittest.TestCase):
    def test_exact_and_tolerance_accounting(self):
        reference = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        within = np.array([[1.0, 2.0 + 1e-7, 3.0 - 1e-3]], dtype=np.float32)
        result = linkage.compare(within, reference, 1e-5, 1e-6)
        self.assertEqual(result["pixels"], 3)
        self.assertEqual(result["exact"], 2)
        self.assertEqual(result["bitwise_exact"], 2)
        self.assertEqual(result["within"], 2)
        self.assertEqual(result["beyond"], 1)
        self.assertAlmostEqual(result["max_abs_diff"], 1e-3, places=6)

    def test_nan_never_counts_as_match(self):
        reference = np.array([[np.nan, 1.0]], dtype=np.float32)
        derived = np.array([[np.nan, 1.0]], dtype=np.float32)
        result = linkage.compare(derived, reference, 1e-5, 1e-6)
        self.assertEqual(result["within"], 1)
        self.assertEqual(result["beyond"], 1)
        self.assertEqual(result["derived_nan"], 1)
        self.assertEqual(result["reference_nan"], 1)

    def test_rejects_shape_and_dimensionality(self):
        for derived, reference in ((np.zeros((2, 2)), np.zeros((2, 3))),
                                   (np.zeros((4,)), np.zeros((4,)))):
            with self.assertRaises(ValueError):
                linkage.compare(derived.astype(np.float32), reference.astype(np.float32), 1e-5, 1e-6)

    def test_signed_zero_is_numerical_not_bitwise_equality(self):
        derived = np.array([[0.0, -0.0, 1.0]], dtype=np.float32)
        reference = np.array([[-0.0, -0.0, 1.0]], dtype=np.float32)
        result = linkage.compare(derived, reference, 1e-5, 1e-6)
        self.assertEqual(result["exact"], 3)
        self.assertEqual(result["bitwise_exact"], 2)
        self.assertEqual(result["within"], 3)
        self.assertEqual(result["beyond"], 0)

    def test_representable_difference_can_be_within_tolerance(self):
        reference = np.array([[2.0]], dtype=np.float32)
        derived = np.nextafter(reference, np.float32(np.inf))
        result = linkage.compare(derived, reference, 1e-5, 1e-6)
        self.assertEqual(result["exact"], 0)
        self.assertEqual(result["bitwise_exact"], 0)
        self.assertEqual(result["within"], 1)
        self.assertEqual(result["beyond"], 0)

    def test_equal_infinities_are_not_valid_terrain_matches(self):
        values = np.array([[np.inf, -np.inf, np.nan, 1.0]], dtype=np.float32)
        result = linkage.compare(values, values, 1e-5, 1e-6)
        self.assertEqual(result["exact"], 1)
        self.assertEqual(result["bitwise_exact"], 1)
        self.assertEqual(result["within"], 1)
        self.assertEqual(result["beyond"], 3)

    def test_bitwise_accounting_handles_noncontiguous_windows(self):
        derived = np.array([[0.0, 8.0, 1.0], [2.0, 9.0, 3.0]], dtype=np.float32)[:, ::2]
        reference = derived.copy()
        reference[0, 0] = -0.0
        result = linkage.compare(derived, reference, 1e-5, 1e-6)
        self.assertEqual(result["exact"], 4)
        self.assertEqual(result["bitwise_exact"], 3)


class WindowTests(unittest.TestCase):
    def test_tile_bounds(self):
        self.assertEqual(linkage.window(3, 7), (768, 1024, 1792, 2048))
        self.assertEqual(linkage.window(89, 179), (22784, 23040, 45824, 46080))

    def test_rejects_out_of_grid_and_non_integer(self):
        for row, col in ((90, 0), (0, 180), (-1, 0), (0, -1), (True, 0), (0, 1.5), ("0", 0)):
            with self.assertRaises(ValueError):
                linkage.window(row, col)


class MaskTests(unittest.TestCase):
    def test_counts_set_bits_not_nonzero_bytes(self):
        rows, columns = 512, 1024
        mask = np.zeros((rows, columns), dtype=bool)
        mask[300, 696:704] = True
        mask[301, 704:707] = True
        self.assertEqual(linkage.mask_nonzero(packed(mask), 1, 2, rows=rows, row_bytes=columns // 8), 11)

    def test_counts_tile_edge_bits_without_neighbor_leakage(self):
        rows, columns = 512, 1024
        mask = np.zeros((rows, columns), dtype=bool)
        mask[256, 512] = True
        mask[511, 767] = True
        mask[255, 512] = True
        mask[256, 511] = True
        mask[511, 768] = True
        self.assertEqual(linkage.mask_nonzero(packed(mask), 1, 2, rows=rows, row_bytes=columns // 8), 2)

    def test_bit_offsets_match_packbits_rows(self):
        rows, columns = 512, 1024
        mask = np.zeros((rows, columns), dtype=bool)
        mask[300, 700] = True
        packed_mask = packed(mask)
        self.assertEqual(linkage.mask_nonzero(packed_mask, 1, 2, rows=rows, row_bytes=columns // 8), 1)
        self.assertEqual(linkage.mask_nonzero(packed_mask, 1, 1, rows=rows, row_bytes=columns // 8), 0)
        self.assertEqual(linkage.mask_nonzero(packed_mask, 0, 0, rows=rows, row_bytes=columns // 8), 0)

    def test_rejects_wrong_payload_size(self):
        with self.assertRaises(ValueError):
            linkage.mask_nonzero(np.zeros(10, dtype=np.uint8), 0, 0, rows=512, row_bytes=64)

    def test_all_zero_payload_reports_no_declared_nan(self):
        packed_mask = np.zeros(512 * 128, dtype=np.uint8)
        for row, col in ((0, 0), (1, 2), (1, 3)):
            self.assertEqual(linkage.mask_nonzero(packed_mask, row, col, rows=512, row_bytes=128), 0)


class PlanTests(unittest.TestCase):
    def test_repository_plan_is_geometry_and_tolerance_consistent(self):
        plan = linkage.load_linkage_plan(ROOT / linkage.PLAN_PATH)
        self.assertEqual(plan["protocol"], "P33_MMAP_TERRAIN_LINKAGE_V1")
        self.assertEqual(tuple(plan["terrain_channels"]), linkage.TERRAIN)
        self.assertEqual(plan["test_pixels_requested"], 0)
        self.assertEqual(plan["training"], "NOT_RUN")
        self.assertIn("claims_not_permitted", plan)
        for entry in plan["assets"]:
            self.assertGreater(entry["bytes"], 0)
            digest = entry.get("lfs_sha256") or entry.get("git_blob_sha")
            self.assertIn(len(digest), (40, 64))

    def test_rejects_plan_with_altered_geometry_or_tolerance(self):
        plan = json.loads((ROOT / linkage.PLAN_PATH).read_text())
        plan["grid"]["shape"] = [23041, 46080]
        with self.assertRaises(ValueError):
            linkage.load_linkage_plan_object(plan)
        plan = json.loads((ROOT / linkage.PLAN_PATH).read_text())
        plan["comparison"]["rtol"] = 1e-3
        with self.assertRaises(ValueError):
            linkage.load_linkage_plan_object(plan)
        plan = json.loads((ROOT / linkage.PLAN_PATH).read_text())
        plan["terrain_channels"] = ["elevation"]
        with self.assertRaises(ValueError):
            linkage.load_linkage_plan_object(plan)


if __name__ == "__main__":
    unittest.main()
