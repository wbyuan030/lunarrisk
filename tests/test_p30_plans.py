"""Offline allowlist and sample-selection tests; public assets are never downloaded here."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


asset = load("p30_asset_plan")
pilot = load("p30_extract_pilot")
validation = load("p30_validate_metadata")


def info(repo, large, size):
    return {"id": repo, "sha": "a"*40, "siblings": [
        *[{"rfilename": name, "size": 100} for name in asset.SMALL_FILES[repo]],
        {"rfilename": large, "size": size, "lfs": {"sha256": "b"*64}}]}


class TestAssetPlan(unittest.TestCase):
    def setUp(self):
        self.dataset = info("ayushprd/Moonstone", "benchmark/lunar_patches_v4.h5", 74449119347)
        self.model = info("nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", "backbone/checkpoint.pt", 2425755440)

    def test_exact_allowlist_is_dry_run_and_never_authorizes_large_download(self):
        plan = asset.build_plan(self.dataset, self.model)
        self.assertEqual(plan["network_requests"], 0)
        self.assertEqual(plan["mode"], "dry_run")
        self.assertEqual(plan["small_allowlist_bytes"], 700)
        self.assertEqual(len(plan["assets"]), 9)
        for row in plan["assets"]:
            self.assertIn("a"*40, row["url"])
            if row["bytes"] > 1024**3:
                self.assertFalse(row["complete_download_allowed"])
                self.assertEqual(row["operation"], "strict_range_only")

    def test_unpinned_or_wrong_repository_is_rejected(self):
        for key, value in (("sha", "main"), ("id", "unrelated/model")):
            broken = {**self.dataset, key: value}
            with self.assertRaises(ValueError):
                asset.build_plan(broken, self.model)

    def test_small_file_size_cannot_grow_into_a_bulk_download(self):
        self.dataset["siblings"][0]["size"] = 2*1024**3
        with self.assertRaises(ValueError):
            asset.build_plan(self.dataset, self.model)

    def test_initial_cap_cannot_be_silently_raised(self):
        with self.assertRaises(ValueError):
            asset.build_plan(self.dataset, self.model, cap=2*1024**3)


class TestSamplePlan(unittest.TestCase):
    def setUp(self):
        self.metadata = {"status": "REAL_HDF5_METADATA_READ_NOT_LOADER_EQUIVALENCE", "metadata": {
            "metadata/split": [2, 0, 1, 0, 1, 2], "metadata/patch_id": [8, 7, 6, 5, 4, 3],
            "metadata/geology/dominant_unit_name": ["Cc", "unknown", "Cc", "Nc", "EIp", "Cc"]}}

    def test_selects_first_valid_original_indices_without_test_pixels(self):
        plan = pilot.sample_plan(self.metadata, 1)
        self.assertEqual(plan["indices"], [3, 2])
        self.assertEqual(plan["patch_ids"], [5, 6])
        self.assertEqual(plan["stored_splits"], [0, 1])
        self.assertEqual(plan["test_pixels_requested"], 0)

    def test_does_not_fill_missing_train_samples_from_other_splits(self):
        with self.assertRaisesRegex(ValueError, "not enough"):
            pilot.sample_plan(self.metadata, 2)

    def test_failed_metadata_or_unbounded_sample_count_is_rejected(self):
        with self.assertRaises(ValueError):
            pilot.sample_plan({**self.metadata, "status": "BLOCKED"}, 1)
        with self.assertRaises(ValueError):
            pilot.sample_plan(self.metadata, 1000)


class TestMetadataAudit(unittest.TestCase):
    def fixture(self):
        return {"status": "REAL_HDF5_METADATA_READ_NOT_LOADER_EQUIVALENCE", "metadata": {
            "metadata/split": [0, 1, 2], "metadata/patch_id": [5, 3, 1],
            "metadata/row_idx": [0, 1, 2], "metadata/col_idx": [0, 1, 2],
            "metadata/center_lat": [89, 87, 85], "metadata/center_lon": [-179, -177, -175],
            "metadata/geology/dominant_unit_id": [1, 2, 3],
            "metadata/geology/dominant_unit_name": ["C1", "C2", "C3"]}}

    def test_histograms_ids_and_footprint_follow_stored_metadata(self):
        summary, records = validation.audit(self.fixture(), {f"C{i}": i for i in range(1, 50)})
        self.assertEqual(summary["geographic_excluded_rows"], 3)
        self.assertEqual(summary["tracks"]["official_stored"]["1"]["age_histogram"], [0, 0, 0, 0, 1])
        self.assertEqual([row["sample_id"] for row in records], ["5", "3", "1"])
        self.assertEqual(records[0]["lat_max"], 90)
        self.assertEqual(records[0]["lon_min"], -180)

    def test_duplicate_ids_and_changed_centers_fail_closed(self):
        for field, values in (("patch_id", [1, 1, 2]), ("center_lat", [89, 86, 85])):
            data = self.fixture()
            data["metadata"][f"metadata/{field}"] = values
            with self.assertRaises(ValueError):
                validation.audit(data, {f"C{i}": i for i in range(1, 50)})


if __name__ == "__main__":
    unittest.main()
