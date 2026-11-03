"""Offline tests of strict download limits; no public requests or lunar measurements."""
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunarrisk.benchmarks.http_range import ByteBudget, HTTPRangeFile, range_get, small_get


class Response(io.BytesIO):
    def __init__(self, body, status=206, headers=None):
        super().__init__(body)
        self.status, self.headers, self.read_calls = status, headers or {}, 0

    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)


class TestRangeBudget(unittest.TestCase):
    def test_ignored_range_never_reads_body(self):
        response = Response(b"enormous-file", status=200)
        with self.assertRaisesRegex(ValueError, "body not read"):
            range_get("https://example.org/data", 0, 4, 1000000, ByteBudget(4),
                      opener=lambda *a, **k: response)
        self.assertEqual(response.read_calls, 0)

    def test_mismatched_range_or_total_never_reads_body(self):
        for value in ("bytes 1-4/20", "bytes 0-3/21", "garbage"):
            response = Response(b"test", headers={"Content-Range": value})
            with self.assertRaisesRegex(ValueError, "Content-Range"):
                range_get("https://example.org/data", 0, 4, 20, ByteBudget(4),
                          opener=lambda *a, **k: response)
            self.assertEqual(response.read_calls, 0)

    def test_size_and_encoding_checked_before_body(self):
        for extra in ({"Content-Length": "9"}, {"Content-Encoding": "gzip"}):
            response = Response(b"test", headers={"Content-Range": "bytes 0-3/20", **extra})
            with self.assertRaises(ValueError):
                range_get("https://example.org/data", 0, 4, 20, ByteBudget(4),
                          opener=lambda *a, **k: response)
            self.assertEqual(response.read_calls, 0)

    def test_transfer_exhaustion_precedes_network(self):
        def forbidden(*args, **kwargs):
            self.fail("network called after budget exhausted")
        with self.assertRaisesRegex(ValueError, "BUDGET"):
            range_get("https://example.org/data", 0, 4, 20, ByteBudget(7, 4), opener=forbidden)

    def test_short_response_counts_received_bytes_and_fails(self):
        budget = ByteBudget(4)
        response = Response(b"abc", headers={"Content-Range": "bytes 0-3/20"})
        with self.assertRaisesRegex(ValueError, "truncated"):
            range_get("https://example.org/data", 0, 4, 20, budget, opener=lambda *a, **k: response)
        self.assertEqual(budget.received, 3)

    def test_small_metadata_limit_counts_rejected_body(self):
        budget = ByteBudget(5)
        with self.assertRaisesRegex(ValueError, "limit"):
            small_get("https://example.org/meta", budget, max_bytes=4,
                      opener=lambda *a, **k: Response(b"abcdef", 200))
        self.assertEqual(budget.received, 5)

    def test_seek_block_crossing_and_cache_reuse(self):
        content = b"abcdefghijklmnopqrstuvwxyz"
        calls = []
        def opener(request, **kwargs):
            interval = request.headers["Range"].split("=")[1]
            start, end = map(int, interval.split("-"))
            calls.append((start, end))
            return Response(content[start:end+1], headers={
                "Content-Range": f"bytes {start}-{end}/{len(content)}"})
        budget = ByteBudget(64)
        with HTTPRangeFile("https://example.org/data", len(content), budget,
                           block_size=4, cache_blocks=2, opener=opener) as stream:
            stream.seek(2)
            self.assertEqual(stream.read(5), b"cdefg")
            stream.seek(-3, io.SEEK_CUR)
            self.assertEqual(stream.read(2), b"ef")
            self.assertEqual(len(calls), 2)
            stream.seek(-2, io.SEEK_END)
            target = bytearray(5)
            self.assertEqual(stream.readinto(target), 2)
            self.assertEqual(target[:2], b"yz")
            self.assertEqual(stream.read(), b"")
            with self.assertRaises(ValueError):
                stream.seek(-1)
        self.assertEqual(budget.received, 10)

    def test_hdf5_reads_real_file_object_with_strict_ranges(self):
        try:
            import h5py
            import numpy as np
        except ImportError:
            self.skipTest("h5py required for the synthetic HDF5 range fixture")
        fixture = io.BytesIO()
        with h5py.File(fixture, "w") as handle:
            handle.create_dataset("patches", data=np.arange(120).reshape(3, 4, 10), chunks=(1, 4, 10))
        body = fixture.getvalue()
        def opener(request, **kwargs):
            start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
            return Response(body[start:end+1], headers={"Content-Range": f"bytes {start}-{end}/{len(body)}"})
        with HTTPRangeFile("https://example.org/fixture", len(body), ByteBudget(1024*1024),
                           block_size=256, opener=opener) as stream:
            with h5py.File(stream, "r") as handle:
                np.testing.assert_array_equal(handle["patches"][2], np.arange(80, 120).reshape(4, 10))


if __name__ == "__main__":
    unittest.main()
