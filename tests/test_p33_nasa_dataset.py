"""Numpy/mock contract tests; synthetic HDF5 test skips if h5py is absent.

These do not establish actual retained-prefix eligibility or real NASA execution.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunarrisk.benchmarks.moonstone import CHANNELS
from lunarrisk.benchmarks.prefix import PrefixClassifierDataset
from lunarrisk.nasa_transfer import dataset as module
from lunarrisk.nasa_transfer.cohort import (
    load_authoritative_cohorts, load_authoritative_train, TRAIN_PLAN_PATH, METADATA_ARCHIVE_PATH)
from test_p33_nasa_contracts import fixture

ROOT = Path(__file__).resolve().parents[1]


class RawSource:
    def __init__(self, split=0):
        self.split = split
        self.indices = np.array([1, 0])
        self.ids = np.array(["b", "a"])
        self.targets = np.array([2, 3])
        self.ordered_ids_hash = hashlib.sha256(b"a\nb").hexdigest()
        self.pixels = np.zeros((2, 28, 4, 4), np.float32)
        self.pixels[:, CHANNELS.index("elevation")] = 14
        self.pixels[:, CHANNELS.index("slope")] = 30
        # Nonterrain missingness must not be read or normalized.
        self.pixels[:, 0] = np.nan
        self.reads = []
        self.closed = False

    def _file(self):
        outer = self
        class Pixels:
            def __getitem__(self, key):
                outer.reads.append(key)
                return outer.pixels[key]
        return {"patches": Pixels()}

    def __getitem__(self, index):
        raise AssertionError("normalized reader must never be called")

    def close(self):
        self.closed = True


def wrapped(split=0):
    _, _, domains, provenance = fixture()
    provenance["nodata_policy"] = module.NODATA_POLICY
    source = RawSource(split)
    return module.TerrainDataset(source, {"a": 2, "b": 3}, domains, provenance), source


class TestAuthoritativeCohorts(unittest.TestCase):
    def test_historical_train_val_ids_targets_and_train_compatibility(self):
        cohorts, receipt = load_authoritative_cohorts(ROOT / TRAIN_PLAN_PATH, ROOT / METADATA_ARCHIVE_PATH)
        self.assertEqual([len(cohorts[x]) for x in ("train", "val")], [5135, 1155])
        self.assertFalse(set(cohorts["train"]) & set(cohorts["val"]))
        train, old_receipt = load_authoritative_train(ROOT / TRAIN_PLAN_PATH, ROOT / METADATA_ARCHIVE_PATH)
        self.assertEqual(train, cohorts["train"])
        self.assertEqual(old_receipt, {k: v for k, v in receipt.items() if not k.startswith("val_")})
        self.assertNotIn("test", cohorts)


class TestTerrainDataset(unittest.TestCase):
    def test_raw_channel_only_normalization_and_loader_contract(self):
        ds, source = wrapped()
        item = ds[0]
        self.assertEqual(set(item), {"dtm", "slope", "target", "sample_id"})
        self.assertEqual(item["dtm"].shape, (1, 4, 4))
        self.assertEqual(item["dtm"].dtype, np.float32)
        self.assertEqual(item["target"], np.int64(2))
        self.assertEqual(item["sample_id"], "a")
        np.testing.assert_array_equal(item["dtm"], 2)
        np.testing.assert_array_equal(item["slope"], 10)
        self.assertEqual([k[1] for k in source.reads], [slice(1, 2), slice(2, 3)])
        self.assertTrue(all(k[0] == 1 for k in source.reads))
        ds.close()
        self.assertTrue(source.closed)

    def test_zero_remains_observed(self):
        ds, source = wrapped()
        source.pixels[:, 1:3] = 0
        np.testing.assert_array_equal(ds[0]["dtm"], -5)

    def test_missing_pixels_and_geometry_fail_without_filtering(self):
        for value in (np.nan, np.inf, -np.inf):
            ds, source = wrapped()
            source.pixels[1, 1, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "missing/nonfinite"):
                ds[0]
            self.assertEqual(len(ds), 2)
        ds, source = wrapped()
        source.pixels = source.pixels[:, :, :3]
        with self.assertRaisesRegex(ValueError, "do not silently resize"):
            ds[0]

    def test_no_finite_sentinel_assertion_is_mandatory(self):
        _, _, domains, provenance = fixture()
        for flag in (None, "finite_only", True):
            provenance["nodata_policy"] = flag
            with self.assertRaisesRegex(ValueError, "NoData policy"):
                module.TerrainDataset(RawSource(), {"a": 2, "b": 3}, domains, provenance)

    def test_cohort_order_targets_and_test_rejected(self):
        _, _, domains, provenance = fixture()
        provenance["nodata_policy"] = module.NODATA_POLICY
        for authority in ({"a": 2}, {"b": 3, "a": 2}, {"a": 4, "b": 3}):
            with self.assertRaisesRegex(ValueError, "full authoritative cohort"):
                module.TerrainDataset(RawSource(), authority, domains, provenance)
        with self.assertRaisesRegex(ValueError, "Test is forbidden"):
            module.TerrainDataset(RawSource(2), {"a": 2, "b": 3}, domains, provenance)

    def test_config_and_indices_are_snapshots_and_non_scalar_indices_blocked(self):
        ds, source = wrapped()
        source.indices[:] = 0
        source.targets[:] = 40
        source.ids[:] = "z"
        self.assertEqual(ds[0]["sample_id"], "a")
        self.assertEqual(ds[0]["target"], 2)
        for index in (slice(None), True, [0]):
            with self.assertRaises(TypeError):
                ds[index]

    def test_scan_reads_all_rows_and_fails_on_any_bad_row(self):
        train, t = wrapped()
        val, v = wrapped(1)
        report = module.scan_datasets({"train": train, "val": val})
        self.assertEqual(report["pixel_scan"], "PASS")
        self.assertEqual(len(t.reads), 4)
        self.assertEqual(len(v.reads), 4)
        v.pixels[0, 2, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            module.scan_datasets({"train": train, "val": val})
        with self.assertRaises(ValueError):
            module.scan_datasets({"test": val})


class TestOpenAuthentication(unittest.TestCase):
    def test_historical_hash_mismatch_precedes_dataset_construction(self):
        _, _, domains, provenance = fixture()
        provenance["nodata_policy"] = module.NODATA_POLICY
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "bad"
            bad.write_bytes(b"not a retained prefix")
            with patch.object(module, "PrefixClassifierDataset") as constructor:
                with self.assertRaisesRegex(ValueError, "historical asset sha256 SHA256 mismatch"):
                    module.open_datasets(data_path=bad, stats_path=bad, class_map_path=bad,
                        metadata_path=bad, range_report_path=bad, plan_path=ROOT / TRAIN_PLAN_PATH,
                        metadata_archive_path=ROOT / METADATA_ARCHIVE_PATH,
                        domains=domains, provenance=provenance)
                constructor.assert_not_called()

    def test_every_external_asset_hash_is_mandatory(self):
        _, _, domains, provenance = fixture()
        provenance["nodata_policy"] = module.NODATA_POLICY
        fields = ("sha256", "stats_sha256", "class_map_sha256", "metadata_sha256", "range_report_sha256")
        with tempfile.TemporaryDirectory() as temporary:
            asset = Path(temporary) / "asset"
            asset.write_text("{}")
            plan = {"data": {"bytes": 2, **{key: "a" * 64 for key in fields}}}
            for position, field in enumerate(fields):
                hashes = ["a" * 64] * position + ["b" * 64]
                with self.subTest(field=field), \
                     patch.object(module, "load_authoritative_cohorts", return_value=({}, {})), \
                     patch("lunarrisk.nasa_transfer.cohort._read_pinned", return_value=json.dumps(plan)), \
                     patch.object(module, "file_sha256", side_effect=hashes), \
                     patch.object(module, "PrefixClassifierDataset") as constructor:
                    with self.assertRaisesRegex(ValueError, f"historical asset {field} SHA256 mismatch"):
                        module.open_datasets(data_path=asset, stats_path=asset, class_map_path=asset,
                            metadata_path=asset, range_report_path=asset, plan_path="unused",
                            metadata_archive_path="unused", domains=domains, provenance=provenance)
                    constructor.assert_not_called()

    def test_mock_authenticated_factory_hashes_each_asset_once_and_receipt_is_path_free(self):
        _, _, domains, provenance = fixture()
        provenance["nodata_policy"] = module.NODATA_POLICY
        with tempfile.TemporaryDirectory() as temporary:
            paths = {}
            for key in ("data", "stats", "class_map", "metadata", "range_report"):
                paths[key + "_path"] = Path(temporary) / key
                paths[key + "_path"].write_text("{}")
            fake_plan = {"data": {"bytes": 2, **{key: "a" * 64 for key in
                ("sha256", "stats_sha256", "class_map_sha256", "metadata_sha256", "range_report_sha256")}}}
            def construct(*args, **kwargs):
                return RawSource(kwargs["split"])
            with patch.object(module, "load_authoritative_cohorts", return_value=(
                    {"train": {"a": 2, "b": 3}, "val": {"a": 2, "b": 3}}, {})), \
                 patch("lunarrisk.nasa_transfer.cohort._read_pinned", return_value=json.dumps(fake_plan)), \
                 patch.object(module, "file_sha256", return_value="a" * 64) as hasher, \
                 patch.object(module, "PrefixClassifierDataset", side_effect=construct):
                datasets, receipt = module.open_datasets(**paths, plan_path="unused",
                    metadata_archive_path="unused", domains=domains, provenance=provenance)
                self.assertEqual(hasher.call_count, 5)
                datasets["train"][0]
                datasets["train"][0]
                self.assertEqual(hasher.call_count, 5)
                self.assertNotIn(temporary, json.dumps(receipt))
                self.assertEqual(receipt["pixel_scan"], "NOT_RUN")
                self.assertEqual(receipt["test_access"], "FORBIDDEN")


@unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py absent: real synthetic HDF5 slicing not executed")
class TestSyntheticHDF5(unittest.TestCase):
    def test_existing_prefix_adapter_with_raw_channel_slices(self):
        import h5py
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "fixture.h5"
            with h5py.File(path, "w") as handle:
                pixels = handle.create_dataset("patches", shape=(2, 28, 256, 256), dtype="float32",
                                               chunks=(1, 28, 256, 256), compression="gzip")
                pixels[:] = 20
                records = []
                for i in range(2):
                    chunk = pixels.id.get_chunk_info_by_coord((i, 0, 0, 0))
                    records.append({"index": i, "sample_id": str(i), "byte_offset": chunk.byte_offset,
                                    "compressed_bytes": chunk.size})
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"attributes": {"channel_names": CHANNELS}, "metadata": {
                "metadata/patch_id": ["0", "1"], "metadata/geology/dominant_unit_name": ["C1", "C1"],
                "metadata/geology/dominant_unit_id": [1, 1], "metadata/split": [0, 1]}}))
            report = root / "ranges.json"
            report.write_text('{"hdf5_ranges": []}')
            plan = {"data": {"bytes": path.stat().st_size, "upstream_bytes": path.stat().st_size},
                    "complete_chunks": records, "splits": {split: {"rows": 1, "indices": [i],
                    "ordered_ids_sha256": hashlib.sha256(str(i).encode()).hexdigest()}
                    for i, split in enumerate(("train", "val"))}}
            source = PrefixClassifierDataset(path, {c: [999, 2] for c in CHANNELS},
                {f"C{i}": i for i in range(1, 50)}, metadata, report, plan, task="geology", split=0)
            self.addCleanup(source.close)
            _, _, domains, provenance = fixture()
            for domain in domains.values():
                domain["input_size"] = 256
            provenance["nodata_policy"] = module.NODATA_POLICY
            terrain = module.TerrainDataset(source, {"0": 0}, domains, provenance)
            with patch.object(PrefixClassifierDataset, "__getitem__", side_effect=AssertionError):
                np.testing.assert_array_equal(terrain[0]["dtm"], 5)


if __name__ == "__main__":
    unittest.main()
