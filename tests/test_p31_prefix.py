"""Synthetic retained-prefix fixtures; no network, real partials or GPU execution."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunarrisk.benchmarks.moonstone import CHANNELS
from lunarrisk.benchmarks.prefix import (
    MissingLocalBytes, PrefixClassifierDataset, PrefixFile, cached_blocks)

try:
    import h5py
    import torch
except ImportError:
    h5py = torch = None


def ids_hash(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


class TemporaryFixture(unittest.TestCase):
    def setUp(self):
        job = os.environ.get("CLAUDE_JOB_DIR")
        root = Path(job) / "tmp" if job else None
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)


class TestPrefixFile(TemporaryFixture):
    def setUp(self):
        super().setUp()
        self.path = self.root / "prefix.bin"
        self.path.write_bytes(b"abcd")

    def test_prefix_cached_ranges_readinto_seek_and_logical_eof(self):
        with PrefixFile(self.path, 12, [(4, b"ef"), (6, b"ghijkl")], 4) as stream:
            self.assertTrue(stream.readable())
            self.assertTrue(stream.seekable())
            self.assertFalse(stream.writable())
            self.assertEqual(stream.read(6), b"abcdef")
            self.assertEqual(stream.seek(-2, 1), 4)
            target = bytearray(10)
            self.assertEqual(stream.readinto(target), 8)
            self.assertEqual(target, b"efghijkl\0\0")
            self.assertEqual(stream.tell(), 12)
            self.assertEqual(stream.read(5), b"")
            self.assertEqual(stream.seek(-2, 2), 10)
            self.assertEqual(stream.read(100), b"kl")
            self.assertEqual(stream.read(0), b"")
            with self.assertRaises(io.UnsupportedOperation):
                stream.write(b"x")
        for operation in (stream.tell, stream.readable, stream.seekable,
                          lambda: stream.read(0), lambda: stream.seek(0)):
            with self.assertRaises(ValueError):
                operation()
        stream.close()
        self.assertEqual(self.path.read_bytes(), b"abcd")

    def test_bounds_invalid_sizes_and_readonly_buffer(self):
        with PrefixFile(self.path, 12, []) as stream:
            for offset, whence in ((-1, 0), (13, 0), (-13, 2), (0, 3)):
                with self.subTest(offset=offset, whence=whence), self.assertRaises(ValueError):
                    stream.seek(offset, whence)
            for operation in (stream.read, lambda: stream.read(-2)):
                with self.assertRaises(ValueError):
                    operation()
            for operation in (lambda: stream.read(1.5), lambda: stream.seek(0.5)):
                with self.assertRaises(TypeError):
                    operation()
            with self.assertRaises(TypeError):
                stream.readinto(b"immutable")

    def test_missing_hole_never_returns_zeros_or_skips_to_cached_bytes(self):
        with PrefixFile(self.path, 12, [(8, b"ijkl")]) as stream:
            stream.seek(4)
            with self.assertRaises(MissingLocalBytes) as error:
                stream.read(8)
            self.assertEqual((error.exception.start, error.exception.length), (4, 8))
            self.assertEqual(stream.tell(), 4)
            stream.seek(8)
            self.assertEqual(stream.read(4), b"ijkl")

    def test_consistent_overlaps_accepted_and_contradictions_rejected(self):
        with PrefixFile(self.path, 12, [(6, b"ghij"), (2, b"cdefgh"), (8, b"ijkl")]) as stream:
            self.assertEqual(stream.read(12), b"abcdefghijkl")
        for blocks in ([(2, b"XX")], [(4, b"efgh"), (6, b"XX")], [(12, b"x")],
                       [(-1, b"x")], [(4, b"")], [(4, bytearray(b"x"))]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                PrefixFile(self.path, 12, blocks)

    def test_constructor_closes_source_on_post_open_validation_failure(self):
        real_open = Path.open
        handles = []
        def tracked(path, *args, **kwargs):
            handle = real_open(path, *args, **kwargs)
            handles.append(handle)
            return handle
        with patch.object(Path, "open", tracked):
            for logical, expected, blocks in ((3, None, []), (12, 5, []), (12, 4, [(0, b"z")])):
                with self.assertRaises(ValueError):
                    PrefixFile(self.path, logical, blocks, expected)
        self.assertEqual(len(handles), 3)
        self.assertTrue(all(handle.closed for handle in handles))
        for value in (0, -1, True, 12.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PrefixFile(self.path, value, [])

    def test_mutation_growth_truncation_and_same_size_write_rejected(self):
        for body in (b"abcdefgh", b"a", b"wxyz"):
            with self.subTest(body=body):
                self.path.write_bytes(b"abcd")
                with PrefixFile(self.path, 12, [], 4) as stream:
                    timestamp = self.path.stat().st_mtime_ns
                    self.path.write_bytes(body)
                    # Some filesystems coalesce rapid same-size write timestamps.
                    os.utime(self.path, ns=(timestamp + 10**9, timestamp + 10**9))
                    with self.assertRaisesRegex(ValueError, "mutated"):
                        stream.read(0)


class TestCachedBlocks(TemporaryFixture):
    def setUp(self):
        super().setUp()
        self.report = self.root / "report.json"
        self.cache = self.root / "cache.bin"
        self.cache.write_bytes(b"abcdef")
        self.descriptor = {"start": 10, "length": 6, "cache": "cache.bin",
                           "sha256": hashlib.sha256(b"abcdef").hexdigest()}

    def save(self, rows=None):
        self.report.write_text(json.dumps({"hdf5_ranges": rows or [self.descriptor]}))

    def test_checksums_and_report_mutation(self):
        self.save()
        digest = hashlib.sha256(self.report.read_bytes()).hexdigest()
        self.assertEqual(cached_blocks(self.report, digest), ((10, b"abcdef"),))
        self.report.write_text(self.report.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "report changed"):
            cached_blocks(self.report, digest)
        for field, value in (("sha256", "0" * 64), ("length", 7), ("length", 0),
                             ("length", True), ("start", -1), ("start", 1.5)):
            bad = {**self.descriptor, field: value}
            self.save([bad])
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                cached_blocks(self.report)

    def test_relative_absolute_and_symlink_path_escape(self):
        nested = self.root / "nested"
        nested.mkdir()
        report = nested / "report.json"
        (nested / "escape").symlink_to(self.cache)
        for name in ("../cache.bin", str(self.cache), "escape"):
            report.write_text(json.dumps({"hdf5_ranges": [{**self.descriptor, "cache": name}]}))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "escapes root"):
                cached_blocks(report)

    def test_contradictory_overlaps_even_with_valid_individual_checksums(self):
        self.save([self.descriptor, {**self.descriptor, "start": 12}])
        with self.assertRaisesRegex(ValueError, "contradictory"):
            cached_blocks(self.report)
        self.save([self.descriptor, self.descriptor])
        self.assertEqual(cached_blocks(self.report), ((10, b"abcdef"),))


@unittest.skipIf(h5py is None or torch is None, "h5py and torch required for synthetic HDF5/DataLoader")
class TestPrefixDataset(TemporaryFixture):
    def setUp(self):
        super().setUp()
        self.path = self.root / "retained.partial"
        full = self.root / "fixture.h5"
        n = 6
        with h5py.File(full, "w") as handle:
            patches = handle.create_dataset("patches", shape=(n, 28, 256, 256), dtype="float32",
                                            chunks=(1, 28, 256, 256), compression="gzip")
            for index in range(n):
                raw = np.full((28, 256, 256), index + 1, dtype=np.float32)
                raw[0, 0, 0] = 0
                raw[0, 0, 1] = np.nan
                patches[index] = raw
            records = []
            for index in range(n):
                chunk = patches.id.get_chunk_info_by_coord((index, 0, 0, 0))
                records.append({"index": index, "sample_id": str(100 + index),
                                "byte_offset": chunk.byte_offset, "compressed_bytes": chunk.size})
        # No Test chunk bytes are retained or cached. HDF5's logical EOF stays intact.
        body = full.read_bytes()
        prefix_size = records[5]["byte_offset"]
        self.path.write_bytes(body[:prefix_size])
        self.full_size = len(body)
        full.unlink()
        self.metadata_path = self.root / "metadata.json"
        self.metadata = {"attributes": {"channel_names": list(CHANNELS)}, "metadata": {
            "metadata/patch_id": list(range(100, 106)),
            "metadata/geology/dominant_unit_name": ["C1"] * n,
            "metadata/geology/dominant_unit_id": [1] * n,
            "metadata/split": [0, 0, 0, 1, 1, 2]}}
        self.save_metadata()
        self.report = self.root / "report.json"
        self.report.write_text(json.dumps({"hdf5_ranges": []}))
        self.stats = {name: [1, 2] for name in CHANNELS}
        self.mapping = {f"C{i}": i for i in range(1, 50)}
        self.plan = {"data": {"bytes": prefix_size, "upstream_bytes": len(body)},
                     "splits": {"train": {"indices": [2, 0, 1], "rows": 3,
                                           "ordered_ids_sha256": ids_hash([102, 100, 101])},
                                "val": {"indices": [4, 3], "rows": 2,
                                         "ordered_ids_sha256": ids_hash([104, 103])}},
                     "complete_chunks": records[:5]}

    def save_metadata(self):
        self.metadata_path.write_text(json.dumps(self.metadata))

    def dataset(self, **kwargs):
        dataset = PrefixClassifierDataset(self.path, self.stats, self.mapping,
                                          self.metadata_path, self.report,
                                          kwargs.pop("plan", self.plan), **kwargs)
        self.addCleanup(dataset.close)
        return dataset

    def test_final_short_batches_train_val_order_no_test_pixels_and_raw_labels_outside_pixels(self):
        from lunarrisk.benchmarks.classifier import make_loader
        read = h5py.Dataset.__getitem__
        seen = []
        def guarded(source, index):
            if source.name == "/patches":
                self.assertIsInstance(index, int)
                self.assertLess(index, 5)
                seen.append(index)
            return read(source, index)
        with patch.object(h5py.Dataset, "__getitem__", guarded):
            train = self.dataset()
            val = self.dataset(task="geology", split=1)
            self.assertEqual(seen, [])
            self.assertIsNone(train._handle)
            batches = list(make_loader(train, 2, False, torch.Generator()))
            self.assertEqual([len(row["target"]) for row in batches], [2, 1])
            self.assertEqual([sample for row in batches for sample in row["sample_id"]], ["102", "100", "101"])
            self.assertEqual(batches[0]["pixels"].shape, (2, 28, 256, 256))
            self.assertEqual(set(batches[0]), {"pixels", "target", "sample_id"})
            self.assertEqual(batches[0]["target"].tolist(), [4, 4])
            self.assertEqual(float(batches[0]["pixels"][0, 0, 0, 0]), -0.5)
            self.assertEqual(float(batches[0]["pixels"][0, 0, 0, 1]), 0)
            self.assertEqual(float(batches[0]["pixels"][0, 1, 0, 0]), 1)
            self.assertEqual([row["sample_id"] for row in val], ["104", "103"])
            self.assertEqual(val.class_counts[0], 2)
        self.assertEqual(seen, [2, 0, 1, 4, 3])

    def test_prefix_never_hashed_or_read_whole_and_constructor_never_opens_hdf5(self):
        read_bytes = Path.read_bytes
        def bounded(path):
            if path == self.path:
                raise AssertionError("whole retained prefix read")
            return read_bytes(path)
        with patch.object(Path, "read_bytes", bounded):
            with patch.object(h5py, "File", side_effect=AssertionError("constructor opened HDF5")):
                dataset = self.dataset()
            self.assertEqual(dataset[0]["sample_id"], "102")
            dataset.close()
            self.assertEqual(dataset[0]["sample_id"], "102")

    def test_invalid_metadata_shape_ids_channel_split_and_mapping(self):
        variants = [("metadata/patch_id", [100] * 6), ("metadata/patch_id", [[i] for i in range(6)]),
                    ("metadata/patch_id", [1.5] * 6), ("metadata/split", [0, 0, 0, 1, 1, 9]),
                    ("metadata/split", [0., 0., 0., 1., 1., 2.]),
                    ("metadata/geology/dominant_unit_id", [0] + [1] * 5),
                    ("metadata/geology/dominant_unit_name", ["C2"] + ["C1"] * 5),
                    ("metadata/geology/dominant_unit_name", ["C1"] * 5)]
        original = copy.deepcopy(self.metadata)
        for key, value in variants:
            self.metadata = copy.deepcopy(original)
            self.metadata["metadata"][key] = value
            self.save_metadata()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.dataset()
        self.metadata = copy.deepcopy(original)
        self.metadata["attributes"]["channel_names"].reverse()
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "channel order"):
            self.dataset()

    def test_invalid_frozen_indices_rows_hashes_and_wrong_other_split(self):
        for indices in ([], [0, 0, 1], [-1, 0, 1], [0, 1, 6], [0, 1, 5], [0, 1, 3],
                        [True, 0, 1], [0., 1., 2.]):
            plan = copy.deepcopy(self.plan)
            plan["splits"]["train"]["indices"] = indices
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.dataset(plan=plan)
        for key, field, value in (("train", "rows", 2), ("train", "rows", 3.),
                                  ("train", "ordered_ids_sha256", "0" * 64),
                                  ("val", "indices", [5])):
            plan = copy.deepcopy(self.plan)
            plan["splits"][key][field] = value
            with self.subTest(key=key, field=field), self.assertRaises(ValueError):
                self.dataset(plan=plan)
        for kwargs in ({"task": "other"}, {"split": 2}, {"split": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.dataset(**kwargs)

    def test_frozen_subset_does_not_silently_filter_unknown_age(self):
        self.mapping["Unknown"] = self.mapping.pop("C1")
        self.metadata["metadata"]["metadata/geology/dominant_unit_name"] = ["Unknown"] * 6
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "task filtering is forbidden"):
            self.dataset()
        self.assertEqual(self.dataset(task="geology").targets.tolist(), [0, 0, 0])

    def test_invalid_incomplete_missing_duplicate_and_mismatched_chunk_records(self):
        for field, value in (("index", -1), ("index", 6), ("sample_id", "wrong"),
                             ("byte_offset", -1), ("compressed_bytes", 0),
                             ("compressed_bytes", self.plan["data"]["bytes"])):
            plan = copy.deepcopy(self.plan)
            plan["complete_chunks"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.dataset(plan=plan)
        for rows in (self.plan["complete_chunks"][1:], self.plan["complete_chunks"] * 2):
            plan = copy.deepcopy(self.plan)
            plan["complete_chunks"] = rows
            with self.assertRaises(ValueError):
                self.dataset(plan=plan)

    def test_physical_chunk_descriptors_checked_before_any_pixels_and_failure_closes(self):
        for field in ("byte_offset", "compressed_bytes"):
            plan = copy.deepcopy(self.plan)
            plan["complete_chunks"][0][field] += 1
            dataset = self.dataset(plan=plan)
            adapters, handles = [], []
            real_adapter, real_file = PrefixFile, h5py.File
            def tracked_adapter(*args, **kwargs):
                adapter = real_adapter(*args, **kwargs)
                adapters.append(adapter)
                return adapter
            def tracked_file(*args, **kwargs):
                handle = real_file(*args, **kwargs)
                handles.append(handle)
                return handle
            with patch("lunarrisk.benchmarks.prefix.PrefixFile", tracked_adapter), \
                    patch.object(h5py, "File", tracked_file), \
                    patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("pixels read too soon")):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "physical chunk"):
                    dataset[0]
            self.assertTrue(all(adapter.closed for adapter in adapters))
            self.assertTrue(all(not handle.id.valid for handle in handles))
            self.assertIsNone(dataset._handle)
            self.assertIsNone(dataset._adapter)

    def test_hdf5_open_failure_closes_adapter(self):
        dataset = self.dataset()
        adapters = []
        def tracked(*args, **kwargs):
            adapter = PrefixFile(*args, **kwargs)
            adapters.append(adapter)
            return adapter
        with patch("lunarrisk.benchmarks.prefix.PrefixFile", tracked), \
                patch.object(h5py, "File", side_effect=OSError("bad fixture")):
            with self.assertRaises(OSError):
                dataset[0]
        self.assertEqual(len(adapters), 1)
        self.assertTrue(adapters[0].closed)

    def test_reject_pixel_shape_chunks_dtype_and_external_link_before_read(self):
        # Make separate, complete tiny HDF5 schemas; no real retained asset is modified.
        for kind in ("shape", "chunks", "dtype", "external", "softlink", "unallocated"):
            path = self.root / f"{kind}.h5"
            with h5py.File(path, "w") as handle:
                if kind == "external":
                    handle["patches"] = h5py.ExternalLink("never-open.h5", "/patches")
                elif kind == "softlink":
                    handle["patches"] = h5py.SoftLink("/elsewhere")
                else:
                    shape = (6, 28, 256, 256) if kind != "shape" else (7, 28, 256, 256)
                    chunks = (1, 28, 256, 256) if kind != "chunks" else (1, 28, 128, 256)
                    handle.create_dataset("patches", shape=shape, chunks=chunks,
                                          dtype="float64" if kind == "dtype" else "float32")
                handle.create_dataset("padding", data=np.zeros(65536, dtype="uint8"))
            plan = copy.deepcopy(self.plan)
            plan["data"] = {"bytes": path.stat().st_size, "upstream_bytes": path.stat().st_size}
            dataset = PrefixClassifierDataset(path, self.stats, self.mapping,
                                              self.metadata_path, self.report, plan)
            self.addCleanup(dataset.close)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "schema|local dataset|physical chunk"):
                dataset[0]
            self.assertIsNone(dataset._handle)
            self.assertIsNone(dataset._adapter)

    def test_prefix_and_range_report_mutation_between_init_and_open(self):
        dataset = self.dataset()
        with self.path.open("ab") as output:
            output.write(b"x")
        with self.assertRaisesRegex(ValueError, "size changed"):
            dataset[0]
        with self.assertRaisesRegex(ValueError, "size differs"):
            self.dataset()
        with self.path.open("r+b") as output:
            output.truncate(self.plan["data"]["bytes"])
        dataset = self.dataset()
        self.report.write_text(self.report.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "report changed"):
            dataset[0]

    def test_prefix_mutation_even_when_hdf5_pixel_chunk_is_cached(self):
        dataset = self.dataset()
        dataset[0]
        with self.path.open("ab") as output:
            output.write(b"x")
        with self.assertRaisesRegex(ValueError, "mutated"):
            dataset[0]

    def test_pickle_drops_handles_and_pid_change_reopens(self):
        dataset = self.dataset()
        parent = dataset._file()
        adapter = dataset._adapter
        clone = pickle.loads(pickle.dumps(dataset))
        self.addCleanup(clone.close)
        self.assertIsNone(clone._handle)
        self.assertIsNone(clone._adapter)
        self.assertEqual(clone[0]["sample_id"], "102")
        self.assertTrue(parent.id.valid)
        dataset._pid = -1
        self.assertEqual(dataset[0]["sample_id"], "102")
        self.assertFalse(parent.id.valid)
        self.assertTrue(adapter.closed)
        self.assertEqual(dataset._pid, os.getpid())

    def test_two_workers_and_spawn_keep_parent_handle_and_final_batch(self):
        dataset = self.dataset()
        parent = dataset._file()
        for context in ("fork", "spawn"):
            with self.subTest(context=context):
                loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2,
                                                     multiprocessing_context=context, drop_last=False)
                iterator = iter(loader)
                workers = list(iterator._workers)
                batches = list(iterator)
                self.assertEqual([worker.exitcode for worker in workers], [0, 0])
                self.assertEqual([len(batch["target"]) for batch in batches], [2, 1])
                self.assertEqual([sample for batch in batches for sample in batch["sample_id"]],
                                 ["102", "100", "101"])
                self.assertTrue(parent.id.valid)
                self.assertEqual(dataset._pid, os.getpid())

    def test_non_scalar_out_of_range_indices_do_not_open_file(self):
        dataset = self.dataset()
        for index in (3, -4):
            with self.assertRaises(IndexError):
                dataset[index]
        for index in (slice(None), [0, 1], 0.5):
            with self.assertRaises(TypeError):
                dataset[index]
        self.assertIsNone(dataset._handle)


if __name__ == "__main__":
    unittest.main()
