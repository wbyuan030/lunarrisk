"""Synthetic tests of the pinned Moonstone contract; these do not establish real equivalence."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunarrisk.benchmarks.moonstone import (
    CHANNELS, GROUP_CHANNELS, MoonstoneHDF5Pilot, age_target, geology_targets,
    normalize_hdf5, select_split,
)

MAPPING = {f"C{i}": i for i in range(1, 50)}
STATS = {channel: {"mean": 1.0, "std": 2.0} for channel in CHANNELS}


class TestMoonstoneContract(unittest.TestCase):
    def test_age_prefix_priority_case_and_unknowns(self):
        values = ["pNc", "Nc", "EIp", "Im", "Ec", "Cc", "eIc", "lIc", "none", " Cc"]
        self.assertEqual([age_target(value) for value in values], [0, 1, 2, 2, 3, 4, -1, -1, -1, -1])
        self.assertEqual(age_target(b"EIp"), 2)

    def test_geographic_boundaries_and_exclusions(self):
        latitude = np.array([-90, -80, -70, -60, 0, 60, 60.1, 70, 70.1, 80, 80.1, np.nan])
        stored = np.zeros(len(latitude))
        expected = [[3, 4, 5], [2, 6, 7], [1, 8, 9]]
        for split in range(3):
            self.assertEqual(np.flatnonzero(select_split(stored, latitude, split, "official_geographic")).tolist(), expected[split])

    def test_stored_membership_and_order_are_never_regenerated(self):
        stored = np.array([2, 0, 1, 0, 2])
        self.assertEqual(np.flatnonzero(select_split(stored, [0]*5, 0)).tolist(), [1, 3])
        with self.assertRaises(ValueError):
            select_split([0, 3], [0, 0], 0)
        with self.assertRaises(ValueError):
            select_split(stored, [0]*5, 3)

    def test_geology_id_map_is_bijective_and_not_silently_filtered(self):
        np.testing.assert_array_equal(geology_targets([1, 49], ["C1", "C49"], MAPPING), [0, 48])
        for ids, names in [([0], ["none"]), ([50], ["C50"]), ([1], ["C2"])]:
            with self.assertRaises(ValueError):
                geology_targets(ids, names, MAPPING)
        with self.assertRaises(ValueError):
            geology_targets([1], ["C1"], {"C1": 1})

    def test_channel_order_no_target_channels(self):
        self.assertEqual(len(CHANNELS), 28)
        self.assertEqual(len(GROUP_CHANNELS), 7)
        self.assertFalse(set(CHANNELS) & {"geology", "dominant_unit_name", "dominant_unit_id", "age"})
        with self.assertRaises(ValueError):
            normalize_hdf5(np.zeros((28, 2, 2)), STATS, CHANNELS[::-1])

    def test_normalization_preserves_raw_zero_and_restores_nonfinite(self):
        raw = np.ones((28, 2, 2), dtype=np.float32)
        raw[0] = [[0, np.nan], [np.inf, 3]]
        pixels, channels, groups = normalize_hdf5(raw, STATS)
        np.testing.assert_array_equal(pixels[0], [[-0.5, 0], [0, 1]])
        self.assertEqual(channels[0], 1)
        self.assertEqual(channels[1], 0)  # finite at the global mean becomes normalized zero
        self.assertEqual(groups[0], 1)  # ANY member, not ALL
        self.assertEqual(groups[1:].sum(), 0)
        self.assertTrue(np.isnan(raw[0, 0, 1]))  # input never mutated
        public_pairs = {channel: [1.0, 2.0] for channel in CHANNELS}
        np.testing.assert_array_equal(normalize_hdf5(raw, public_pairs)[0], pixels)
        with self.assertRaises(ValueError):
            normalize_hdf5(raw, {})


class TestMoonstoneHDF5(unittest.TestCase):
    def setUp(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py required for the synthetic HDF5 fixture")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "pilot.h5"
        with h5py.File(self.path, "w") as handle:
            handle.attrs["channel_names"] = json.dumps(CHANNELS)
            patches = handle.create_dataset("patches", shape=(4, 28, 256, 256), dtype="float32",
                                            chunks=(1, 28, 256, 256), compression="gzip")
            patches[0, 0, :2, :2] = 7.0
            meta = handle.create_group("metadata")
            for key, values in {"patch_id": [8, 4, 11, 2], "row_idx": [1, 2, 3, 4],
                                "col_idx": [5, 6, 7, 8], "center_lat": [0, 65, 75, 5],
                                "split": [0, 1, 2, 0]}.items():
                meta[key] = values
            meta["geology/dominant_unit_id"] = [1, 2, 3, 0]
            meta["geology/dominant_unit_name"] = np.array([b"C1", b"C2", b"C3", b"none"])

    def test_age_unknown_filter_split_and_no_target_leakage(self):
        with_dataset = MoonstoneHDF5Pilot(self.path, STATS, MAPPING, split=0)
        self.addCleanup(with_dataset.close)
        self.assertEqual(with_dataset.protocol, "HDF5_DERIVED_PILOT")
        self.assertEqual(with_dataset.ids[with_dataset.indices].tolist(), [8])
        self.assertEqual(with_dataset.unknown_age_count, 1)
        sample = with_dataset[0]
        self.assertEqual(sample["target"], 4)
        self.assertEqual(sample["sample_id"], "8")
        self.assertEqual(sample["pixels"].shape, (28, 256, 256))
        self.assertEqual(set(sample), {"pixels", "has_channel", "has_group", "target", "sample_id"})
        self.assertEqual(with_dataset.class_counts.tolist(), [0, 0, 0, 0, 1])
        # Target metadata is not numerically concatenated to the pixels.
        self.assertEqual(sample["pixels"][0, 0, 0], 3.0)

    def test_geology_refuses_upstream_invalid_id_instead_of_relabeling(self):
        with self.assertRaisesRegex(ValueError, "INVALID_GEOLOGY_ID"):
            MoonstoneHDF5Pilot(self.path, STATS, MAPPING, task="geology")

    def test_worker_pickle_has_no_live_handle_and_reads_same_sample(self):
        dataset = MoonstoneHDF5Pilot(self.path, STATS, MAPPING)
        self.addCleanup(dataset.close)
        original = dataset[0]
        restored = pickle.loads(pickle.dumps(dataset))
        self.addCleanup(restored.close)
        self.assertIsNone(restored._handle)
        np.testing.assert_array_equal(restored[0]["pixels"], original["pixels"])
        self.assertEqual(restored[0]["sample_id"], original["sample_id"])

    def test_duplicate_ids_and_channel_permutation_are_refused(self):
        import h5py
        with h5py.File(self.path, "r+") as handle:
            handle["metadata/patch_id"][1] = 8
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MoonstoneHDF5Pilot(self.path, STATS, MAPPING)
        with h5py.File(self.path, "r+") as handle:
            handle["metadata/patch_id"][1] = 4
            handle.attrs["channel_names"] = json.dumps(CHANNELS[::-1])
        with self.assertRaisesRegex(ValueError, "channel order"):
            MoonstoneHDF5Pilot(self.path, STATS, MAPPING)


if __name__ == "__main__":
    unittest.main()
