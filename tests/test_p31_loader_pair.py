"""Pure synthetic offline P31-02 tests: coverage, complete crops, exact I/O and reservations."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from lunarrisk.benchmarks.http_range import ByteBudget, range_get
from lunarrisk.benchmarks.mmap_pair import DATA_REVISION, SHAPE
from lunarrisk.benchmarks.moonstone import CHANNELS
from lunarrisk.benchmarks.p31_pair import (
    CAP, HDF5_BYTES, HDF5_LIMIT, PRIOR_CONSERVATIVE, ExactRangeFile, PersistentOpener,
    build_plan, canonical_hash, check_lock, compare_crop, digest, ranges, reserved_workers,
    run_workers, select_records)


def fixture():
    present = [True]*28
    present[3] = False
    m = {"metadata/patch_id": [10, 11, 12, 13], "metadata/split": [0, 1, 2, 0],
         "metadata/row_idx": [15, 15, 15, 15], "metadata/col_idx": [0, 1, 2, 3],
         "metadata/center_lat": [59., 59., 59., 59.],
         "metadata/has_channel": [present.copy() for _ in range(4)],
         "metadata/m3/m3_valid_fraction": [.5]*4, "metadata/valid_fraction": [0.]*4,
         "metadata/geology/dominant_unit_id": [1]*4,
         "metadata/geology/dominant_unit_name": ["pNsynthetic"]*4}
    inspection = {"status": "REAL_HDF5_METADATA_READ_NOT_LOADER_EQUIVALENCE",
                  "attributes": {"channel_names": list(CHANNELS)}, "metadata": m}
    listing = {"repository": "ayushprd/Moonstone", "revision": DATA_REVISION,
               "files": [{"rfilename": "benchmark/lunar_patches_v4.h5", "size": HDF5_BYTES}]+[
                   {"rfilename": f"pretraining/mmap/{c}.npy", "size": SHAPE[0]*SHAPE[1]*4}
                   for c in CHANNELS]}
    stats = {c: [2., 2.] for c in CHANNELS}
    mapping = {"pNsynthetic": 1, **{f"unit{i}": i for i in range(2, 50)}}
    return inspection, listing, stats, mapping


class Response(io.BytesIO):
    def __init__(self, raw, start=0, total=None, status=206):
        super().__init__(raw)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{start+len(raw)-1}/{total or len(raw)}",
                        "Content-Length": str(len(raw))}
        self.read_calls = 0
    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)


class TestSelectionAndPlan(unittest.TestCase):
    def test_minimum_both_splits_all_available_channels_and_partial_nonpolar(self):
        inspection, _, _, mapping = fixture()
        records, missing = select_records(inspection, mapping)
        self.assertEqual([r["sample_id"] for r in records], ["10", "11"])
        self.assertEqual(missing, ["roughness"])
        self.assertEqual([r["age_target"] for r in records], [0, 0])
        # Targets and model performance never determine the minimum numeric ID.
        inspection["metadata"]["metadata/geology/dominant_unit_name"][0] = "unit2"
        inspection["metadata"]["metadata/geology/dominant_unit_id"][0] = 2
        self.assertEqual(select_records(inspection, mapping)[0][0]["sample_id"], "10")

    def test_test_only_channel_never_used_to_claim_train_val_coverage(self):
        inspection, _, _, mapping = fixture()
        inspection["metadata"]["metadata/has_channel"][2][3] = True
        self.assertEqual(select_records(inspection, mapping)[1], ["roughness"])

    def test_missing_group_or_split_cannot_silently_borrow_test(self):
        for change in ("group", "split", "partial", "polar", "duplicate", "label"):
            inspection, _, _, mapping = fixture()
            m = inspection["metadata"]
            if change == "group":
                for row in m["metadata/has_channel"]: row[19:21] = [False, False]
            if change == "split": m["metadata/split"][1] = 2
            if change == "partial": m["metadata/m3/m3_valid_fraction"][1] = 1.
            if change == "polar": m["metadata/center_lat"][1] = 61.
            if change == "duplicate": m["metadata/patch_id"][1] = 10
            if change == "label": m["metadata/geology/dominant_unit_id"][0] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_records(inspection, mapping)

    def test_locked_plan_exact_requests_and_budget_no_reset(self):
        inspection, listing, stats, mapping = fixture()
        plan = build_plan(inspection, listing, stats, mapping, {"metadata": "synthetic"})
        check_lock(plan)
        self.assertEqual(plan["transfer"]["mmap_requests"], 14336)
        self.assertEqual(plan["transfer"]["mmap_body_bytes"], 14680064)
        self.assertEqual(plan["transfer"]["new_body_upper_bound"], HDF5_LIMIT+14680064)
        self.assertEqual(plan["transfer"]["cumulative_conservative_upper_bound"], PRIOR_CONSERVATIVE+HDF5_LIMIT+14680064)
        self.assertLess(plan["transfer"]["cumulative_conservative_upper_bound"], CAP)
        plan["rtol"] = .1
        with self.assertRaises(ValueError): check_lock(plan)

    def test_checked_in_plan_lock_expands_to_exact_full_crop_requests(self):
        plan = json.loads((ROOT/"protocols/p31_loader_pair_plan.json").read_text())
        check_lock(plan)
        expanded = list(ranges(plan["records"]))
        self.assertEqual(canonical_hash(expanded), plan["mmap_ranges"]["expanded_requests_sha256"])
        self.assertEqual(len(expanded), plan["transfer"]["mmap_requests"])
        self.assertEqual(sum(r["length"] for r in expanded), plan["transfer"]["mmap_body_bytes"])
        self.assertEqual([r["sample_id"] for r in plan["records"]], ["2705", "2700"])
        self.assertEqual({r["stored_split"] for r in plan["records"]}, {0, 1})

    def test_source_revision_size_and_stats_fail_before_network(self):
        for change in ("revision", "size", "stats"):
            inspection, listing, stats, mapping = fixture()
            if change == "revision": listing["revision"] = "main"
            if change == "size": listing["files"][1]["size"] += 16
            if change == "stats": stats[CHANNELS[0]] = [0, 0]
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_plan(inspection, listing, stats, mapping, {})

    def test_exact_offsets_full_rows_no_gap_overfetch_and_last_grid_tile(self):
        inspection, _, _, mapping = fixture()
        records, _ = select_records(inspection, mapping)
        records[1].update(row_idx=89, col_idx=179)
        requests = list(ranges(records))
        self.assertEqual(len(requests), 2*28*256)
        self.assertTrue(all(r["length"] == 1024 for r in requests))
        self.assertEqual(requests[0]["start"], 15*256*46080*4)
        self.assertEqual(requests[-1]["start"]+1024, SHAPE[0]*SHAPE[1]*4)
        self.assertEqual(requests[1]["start"]-requests[0]["start"], 46080*4)
        self.assertEqual({r["sample_id"] for r in requests}, {"10", "11"})

    def test_offset_bounds_test_duplicate_id_and_duplicate_tile_rejected(self):
        for change in ("row", "col", "split", "id", "tile"):
            inspection, _, _, mapping = fixture()
            records, _ = select_records(inspection, mapping)
            if change == "row": records[1]["row_idx"] = 90
            if change == "col": records[1]["col_idx"] = -1
            if change == "split": records[1]["stored_split"] = 2
            if change == "id": records[1]["sample_id"] = records[0]["sample_id"]
            if change == "tile": records[1]["col_idx"] = 0
            with self.subTest(change=change), self.assertRaises(ValueError): list(ranges(records))


class TestFullCropComparison(unittest.TestCase):
    def setUp(self):
        self.stats = {c: [2., 2.] for c in CHANNELS}
        self.raw = np.full((28, 256, 256), 4., dtype=np.float32)
        self.raw[3] = np.nan
        self.raw[8, 20, :4] = np.nan
        self.raw[19, 5, 6] = 0.  # Finite raw zero is a valid normalized -1.
        self.remote = (self.raw-2)/2
        self.remote[~np.isfinite(self.remote)] = 0

    def test_full_counts_zero_partial_radar_and_presence(self):
        result = compare_crop(self.raw, self.remote, self.stats)
        self.assertTrue(result["numerical_and_masks_pass"])
        self.assertEqual(sum(c["compared_pixels"] for c in result["channels"]), 28*65536)
        self.assertEqual(result["channels"][3]["raw_missing_pixels"], 65536)
        self.assertEqual(result["channels"][8]["raw_missing_pixels"], 4)
        self.assertEqual(result["channels"][19]["raw_finite_zero_pixels"], 1)
        self.assertEqual(result["channels"][19]["derived_valid_pixels"], 65536)
        self.assertTrue(all(result["mmap_group_presence"]))

    def test_mismatch_outside_old_three_rows_caught(self):
        self.remote[27, 72, 191] += .01
        result = compare_crop(self.raw, self.remote, self.stats)
        self.assertFalse(result["numerical_and_masks_pass"])
        self.assertEqual(result["channels"][27]["not_close_pixels"], 1)

    def test_numeric_close_does_not_hide_exact_mask_disagreement(self):
        self.raw[0, 42, 8] = 2+np.float32(2e-7)
        self.remote[0, 42, 8] = 0
        result = compare_crop(self.raw, self.remote, self.stats)
        self.assertEqual(result["channels"][0]["not_close_pixels"], 0)
        self.assertEqual(result["channels"][0]["mask_disagreements"], 1)
        self.assertFalse(result["numerical_and_masks_pass"])

    def test_nonfinite_mmap_masks_and_wrong_shape(self):
        self.remote[3] = np.inf
        self.assertTrue(compare_crop(self.raw, self.remote, self.stats)["numerical_and_masks_pass"])
        with self.assertRaises(ValueError): compare_crop(self.raw[:, :3], self.remote[:, :3], self.stats)


class TestBoundedIO(unittest.TestCase):
    def test_streaming_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"large.bin"
            value = b"abc"*(1 << 20)
            path.write_bytes(value)
            self.assertEqual(digest(path), hashlib.sha256(value).hexdigest())

    def test_hdf_exact_reads_and_request_ceiling_no_aligned_neighbor_body(self):
        seen = []
        def opener(request, timeout):
            span = request.get_header("Range")
            seen.append(span)
            return Response(b"abc", 11, 100)
        saved = []
        stream = ExactRangeFile({"url": "https://example.test/file", "bytes": 100}, ByteBudget(3),
                                opener, lambda r, b: saved.append((r, b)), request_limit=1)
        stream.seek(11)
        self.assertEqual(stream.read(3), b"abc")
        self.assertEqual(seen, ["bytes=11-13"])
        self.assertEqual(saved[0][0], {"start": 11, "length": 3})
        with self.assertRaises(ValueError): stream.read(1)
        with self.assertRaises(ValueError): stream.read()

    def test_206_and_headers_validated_before_any_body(self):
        for change in ("status", "range", "length", "encoding"):
            response = Response(b"abc")
            if change == "status": response.status = 200
            if change == "range": response.headers["Content-Range"] = "bytes 1-3/3"
            if change == "length": response.headers["Content-Length"] = "4"
            if change == "encoding": response.headers["Content-Encoding"] = "gzip"
            budget = ByteBudget(3)
            with self.subTest(change=change), self.assertRaises(ValueError):
                range_get("https://example.test/x", 0, 3, 3, budget, lambda *a, **k: response)
            self.assertEqual(response.read_calls, 0)
            self.assertEqual(budget.received, 0)

    def test_reservation_concurrency_every_byte_reserved_before_start(self):
        jobs = [{"length": 3} for _ in range(24)]
        with self.assertRaises(ValueError): reserved_workers(jobs, 4, 71)
        reservations = reserved_workers(jobs, 4, 72)
        barrier = threading.Barrier(4)
        def worker(i, lane, budget):
            barrier.wait(timeout=5)
            for job in lane: self.assertEqual(budget.read(io.BytesIO(b"abc"), job["length"]), b"abc")
            with self.assertRaises(ValueError): budget.check(1)
            return budget.received
        self.assertEqual(run_workers(reservations, worker), [18]*4)
        self.assertEqual(sum(b.received for _, b in reservations), 72)

    def test_failed_truncated_body_charged_no_retry(self):
        response = Response(b"ab", total=3)
        response.headers.update({"Content-Range": "bytes 0-2/3", "Content-Length": "3"})
        budget = ByteBudget(3)
        opener = Mock(return_value=response)
        with self.assertRaises(ValueError): range_get("https://example.test/x", 0, 3, 3, budget, opener)
        self.assertEqual(budget.received, 2)
        self.assertEqual(opener.call_count, 1)

    def test_redirect_bodies_not_consumed_and_streamed_response(self):
        # No requests/network dependency: construct the adapter around a fake session.
        opener = PersistentOpener.__new__(PersistentOpener)
        redirect = Mock(status_code=302, headers={"Location": "https://cdn.test/x"})
        final = Mock(status_code=206, headers={"Content-Range": "bytes 0-2/3", "Content-Length": "3"})
        final.raw.read.return_value = b"abc"
        opener.session = Mock()
        opener.session.get.side_effect = [redirect, final]
        budget = ByteBudget(3)
        self.assertEqual(range_get("https://example.test/x", 0, 3, 3, budget, opener), b"abc")
        redirect.close.assert_called_once()
        redirect.raw.read.assert_not_called()
        for call in opener.session.get.call_args_list:
            self.assertFalse(call.kwargs["allow_redirects"])
            self.assertTrue(call.kwargs["stream"])
        final.raw.read.assert_called_once_with(3, decode_content=False)


class TestHDF5AdapterIntegration(unittest.TestCase):
    def test_real_h5py_selected_chunks_never_overread_test_pixel_chunk(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("host lacks h5py; run same synthetic test in existing ROCm container")
        local = io.BytesIO()
        with h5py.File(local, "w") as handle:
            patches = handle.create_dataset("patches", (4, 28, 256, 256), dtype="float32",
                                            chunks=(1, 28, 256, 256), compression="gzip")
            for i in range(4): patches[i] = i+1
        blob = local.getvalue()
        with h5py.File(io.BytesIO(blob), "r") as handle:
            test_chunk = handle["patches"].id.get_chunk_info_by_coord((2, 0, 0, 0))
        seen = []
        def opener(request, timeout):
            start, end = map(int, request.get_header("Range")[6:].split("-"))
            seen.append((start, end+1))
            return Response(blob[start:end+1], start, len(blob))
        budget = ByteBudget(len(blob)*10)
        stream = ExactRangeFile({"url": "https://example.test/synthetic.h5", "bytes": len(blob)},
                                budget, opener, lambda r, b: None)
        with h5py.File(stream, "r") as handle:
            self.assertTrue(np.all(handle["patches"][0] == 1))
            self.assertTrue(np.all(handle["patches"][1] == 2))
        a, b = test_chunk.byte_offset, test_chunk.byte_offset+test_chunk.size
        self.assertFalse(any(start < b and end > a for start, end in seen))
        self.assertEqual(budget.received, sum(end-start for start, end in seen))


if __name__ == "__main__":
    unittest.main()
