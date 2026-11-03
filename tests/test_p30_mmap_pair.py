"""Synthetic byte-layout and comparison tests; no lunar data or network access."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from lunarrisk.benchmarks.mmap_pair import DATA_REVISION, FIXED_IDS, SHAPE, build_plan, compare_row
from lunarrisk.benchmarks.moonstone import CHANNELS


class TestMmapProbePlan(unittest.TestCase):
    def fixture(self):
        records = [{"sample_id": sample, "row_idx": 0, "col_idx": int(sample),
                    "stored_split": int(int(sample) >= 20)} for sample in FIXED_IDS]
        listing = {"repository": "ayushprd/Moonstone", "revision": DATA_REVISION,
                   "files": [{"rfilename": f"pretraining/mmap/{channel}.npy",
                              "size": SHAPE[0]*SHAPE[1]*4} for channel in CHANNELS]}
        return records, listing

    def test_exact_selected_rows_merge_adjacency_without_gap_overfetch(self):
        plan = build_plan(*self.fixture())
        self.assertEqual(len(plan["requests"]), 168)
        self.assertEqual(plan["new_body_bytes"], 688128)
        self.assertEqual(plan["full_crops_body_bytes"], 58720256)
        self.assertEqual(plan["full_crops_requests_after_merging"], 14336)
        self.assertEqual(plan["requests"][0]["start"], 0)
        self.assertEqual(plan["requests"][1]["start"], 128*46080*4)
        self.assertEqual(plan["requests"][3]["start"], 20*256*4)
        self.assertEqual(plan["requests"][3]["ids"], ["20", "21", "22", "23"])
        self.assertTrue(all(r["length"] == 4096 for r in plan["requests"]))
        self.assertFalse(plan["full_pixel_equivalence_validated"])
        self.assertFalse(plan["full_channel_group_masks_validated"])

    def test_test_split_changed_ids_duplicate_tiles_and_wrong_sizes_are_rejected(self):
        for key, value in (("stored_split", 2), ("sample_id", "900"), ("col_idx", 1)):
            records, listing = self.fixture()
            records[0][key] = value
            with self.assertRaises(ValueError):
                build_plan(records, listing)
        records, listing = self.fixture()
        listing["files"][0]["size"] += 16  # A real .npy header must not shift the assumed offsets.
        with self.assertRaises(ValueError):
            build_plan(records, listing)

    def test_unpinned_or_wrong_source_is_rejected(self):
        for field, value in (("revision", "main"), ("repository", "other/dataset")):
            records, listing = self.fixture()
            listing[field] = value
            with self.assertRaises(ValueError):
                build_plan(records, listing)


class TestMmapRowComparison(unittest.TestCase):
    def test_upstream_nonfinite_zero_mask_and_reference_tolerance(self):
        derived = np.array([0, 0, 2, 1e-7, 4], dtype=np.float32)
        remote = np.array([np.nan, np.inf, 2+1e-6, 0, 5], dtype=np.float32)
        result = compare_row(derived, remote)
        self.assertEqual(result["compared_pixels"], 5)
        self.assertEqual(result["not_close_pixels"], 1)
        self.assertEqual(result["pixel_mask_disagreements"], 1)
        self.assertEqual(result["remote_nonfinite_pixels"], 2)
        self.assertEqual(result["max_abs_error"], 1)
        self.assertEqual(result["examples"][0]["x"], 4)

    def test_nonfinite_derived_and_unpaired_shapes_fail(self):
        for a, b in (([np.nan], [0]), ([1], [1, 2]), ([[1]], [[1]])):
            with self.assertRaises(ValueError):
                compare_row(a, b)


if __name__ == "__main__":
    unittest.main()
