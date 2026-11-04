"""Read-only retained HDF5 prefix and checksum-verified local metadata ranges.

The caller must verify the frozen plan's data SHA256, metadata_sha256 and
range_report_sha256 before constructing datasets. This module never hashes the
whole prefix, fetches bytes, fills holes, or opens the underlying HDF5 for writing.
"""
from __future__ import annotations

from bisect import bisect_right
import copy
import hashlib
import io
import json
from multiprocessing.util import Finalize
import operator
import os
from pathlib import Path

import numpy as np

from .moonstone import CHANNELS, age_target, geology_targets, normalize_hdf5


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"invalid {name}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"invalid {name}") from exc
    if result < minimum:
        raise ValueError(f"invalid {name}")
    return result


def _signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _consistent_blocks(blocks, logical_size=None):
    """Sort ranges, rejecting contradictory overlap rather than selecting a winner."""
    ordered = []
    for start, body in blocks:
        start = _integer(start, "range start")
        if not isinstance(body, bytes) or not body:
            raise ValueError("cached range must contain immutable nonempty bytes")
        if logical_size is not None and start + len(body) > logical_size:
            raise ValueError("cached range outside logical asset")
        ordered.append((start, body))
    result = []
    for start, body in sorted(ordered, key=lambda item: item[0]):
        if result and start < result[-1][0] + len(result[-1][1]):
            previous, old = result[-1]
            offset = start - previous
            overlap = min(len(body), len(old) - offset)
            if old[offset:offset + overlap] != body[:overlap]:
                raise ValueError("contradictory cached range overlap")
            if overlap < len(body):
                result[-1] = (previous, old + body[overlap:])
        else:
            result.append((start, body))
    return tuple(result)


class MissingLocalBytes(OSError):
    def __init__(self, start, length):
        self.start, self.length = start, length
        super().__init__(f"local HDF5 bytes unavailable: {start}+{length}")


class PrefixFile(io.RawIOBase):
    """Original logical EOF, backed exclusively by the prefix and cached ranges."""

    def __init__(self, path, logical_size, blocks, expected_bytes=None):
        super().__init__()
        self.source = None
        try:
            self.logical_size = _integer(logical_size, "logical size", 1)
            self.blocks = _consistent_blocks(blocks, self.logical_size)
            self.source = Path(path).open("rb", buffering=0)
            self._signature = _signature(os.fstat(self.source.fileno()))
            self.prefix_bytes = self._signature[2]
            if self.prefix_bytes > self.logical_size:
                raise ValueError("prefix exceeds logical asset")
            if expected_bytes is not None and self.prefix_bytes != _integer(expected_bytes, "prefix size"):
                raise ValueError("prefix size changed")
            for start, body in self.blocks:
                length = min(len(body), max(0, self.prefix_bytes - start))
                if length:
                    self.source.seek(start)
                    if self.source.read(length) != body[:length]:
                        raise ValueError("cached range contradicts prefix")
            self._starts = tuple(start for start, _ in self.blocks)
            self.position = 0
            self._check_unchanged()
        except BaseException:
            self.close()
            raise

    def _check_unchanged(self):
        self._checkClosed()
        if _signature(os.fstat(self.source.fileno())) != self._signature:
            raise ValueError("prefix mutated")

    def readable(self):
        self._checkClosed()
        return True

    def seekable(self):
        self._checkClosed()
        return True

    def tell(self):
        self._checkClosed()
        return self.position

    def seek(self, offset, whence=0):
        self._checkClosed()
        offset = operator.index(offset)
        whence = operator.index(whence)
        if whence not in (0, 1, 2):
            raise ValueError("invalid seek origin")
        value = (0, self.position, self.logical_size)[whence] + offset
        if not 0 <= value <= self.logical_size:
            raise ValueError("seek outside logical asset")
        self.position = value
        return value

    def read(self, size=-1):
        self._check_unchanged()
        size = operator.index(size)
        if size < 0:
            raise ValueError("unbounded read forbidden")
        end = min(self.position + size, self.logical_size)
        parts = []
        while self.position < end:
            if self.position < self.prefix_bytes:
                length = min(end, self.prefix_bytes) - self.position
                self.source.seek(self.position)
                body = self.source.read(length)
                if len(body) != length:
                    raise MissingLocalBytes(self.position, length)
            else:
                block_index = bisect_right(self._starts, self.position) - 1
                if block_index < 0:
                    raise MissingLocalBytes(self.position, end - self.position)
                start, block = self.blocks[block_index]
                if self.position >= start + len(block):
                    raise MissingLocalBytes(self.position, end - self.position)
                length = min(end - self.position, start + len(block) - self.position)
                body = block[self.position - start:self.position - start + length]
            parts.append(body)
            self.position += len(body)
        self._check_unchanged()
        return b"".join(parts)

    def readinto(self, target):
        view = memoryview(target).cast("B")
        if view.readonly:
            raise TypeError("readinto needs a writable buffer")
        body = self.read(view.nbytes)
        view[:len(body)] = body
        return len(body)

    def write(self, data):
        self._checkClosed()
        raise io.UnsupportedOperation("read-only retained prefix")

    def close(self):
        try:
            if self.source is not None:
                self.source.close()
        finally:
            super().close()


def cached_blocks(report_path, expected_sha256=None):
    """Verify local cached ranges; the report itself is authenticated by the caller."""
    report_path = Path(report_path)
    raw = report_path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("range report changed")
    report = json.loads(raw)
    root = report_path.parent.resolve()
    blocks = []
    for descriptor in report["hdf5_ranges"]:
        start = _integer(descriptor["start"], "range start")
        length = _integer(descriptor["length"], "range length", 1)
        relative = Path(descriptor["cache"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("cache path escapes root")
        with path.open("rb") as source:
            if os.fstat(source.fileno()).st_size != length:
                raise ValueError("cached HDF5 range length mismatch")
            body = source.read(length + 1)
        if len(body) != length or hashlib.sha256(body).hexdigest() != descriptor["sha256"]:
            raise ValueError("cached HDF5 range checksum mismatch")
        blocks.append((start, body))
    return _consistent_blocks(blocks)


def _close_handles(handle, adapter):
    try:
        if handle is not None:
            handle.close()
    finally:
        if adapter is not None:
            adapter.close()


class PrefixClassifierDataset:
    """Frozen Train/Val subset; IDs and raw labels stay outside normalized pixels.

    Metadata and targets are loaded without reading pixels. Each worker lazily
    opens its own read-only adapter and verifies selected physical chunk records.
    Full-prefix and plan input hash validation belongs to the caller, once per run.
    """

    def __init__(self, path, stats, mapping, metadata_path, range_report, plan, task="age", split=0):
        self._handle, self._adapter, self._pid = None, None, None
        self._finalizer = None
        if task not in ("age", "geology") or isinstance(split, bool) or split not in (0, 1):
            raise ValueError("Train/Val classification only")
        self.task, self.split = task, split
        self.track = "retained_prefix_stored_subset"
        self.path = str(Path(path).resolve())
        self.range_report = str(Path(range_report).resolve())
        self.stats = copy.deepcopy(stats)
        # Check statistics with a tiny array, never by materializing an entire split.
        normalize_hdf5(np.zeros((len(CHANNELS), 1, 1), dtype=np.float32), self.stats)
        self.logical_bytes = _integer(plan["data"]["upstream_bytes"], "logical size", 1)
        self.prefix_bytes = _integer(plan["data"]["bytes"], "prefix size", 1)
        self._prefix_signature = _signature(Path(self.path).stat())
        if self.prefix_bytes > self.logical_bytes or self._prefix_signature[2] != self.prefix_bytes:
            raise ValueError("prefix size differs from frozen plan")
        self._report_hash = hashlib.sha256(Path(self.range_report).read_bytes()).hexdigest()
        raw = json.loads(Path(metadata_path).read_text())
        if tuple(raw["attributes"]["channel_names"]) != CHANNELS:
            raise ValueError("channel order mismatch")
        meta = raw["metadata"]
        self.ids = np.asarray(meta["metadata/patch_id"])
        names = np.asarray(meta["metadata/geology/dominant_unit_name"])
        units = np.asarray(meta["metadata/geology/dominant_unit_id"])
        stored = np.asarray(meta["metadata/split"])
        n = len(self.ids) if self.ids.ndim == 1 else 0
        if n == 0 or any(values.shape != (n,) for values in (self.ids, names, units, stored)):
            raise ValueError("metadata lengths/shapes disagree")
        if (self.ids.dtype.kind not in "iuU" or any(not str(value) for value in self.ids)
                or len(set(map(str, self.ids))) != n):
            raise ValueError("invalid or duplicate IDs")
        if stored.dtype.kind not in "iu" or not np.isin(stored, [0, 1, 2]).all():
            raise ValueError("invalid stored split")
        geology = geology_targets(units, names, mapping)
        targets = np.asarray([age_target(name) for name in names]) if task == "age" else geology
        records = {}
        for row in plan["complete_chunks"]:
            index = _integer(row["index"], "chunk index")
            offset = _integer(row["byte_offset"], "chunk offset")
            length = _integer(row["compressed_bytes"], "chunk length", 1)
            if (index >= n or index in records or row["sample_id"] != str(self.ids[index])
                    or offset + length > self.prefix_bytes):
                raise ValueError("invalid, duplicate or incomplete chunk record")
            records[index] = (offset, length)
        selections = {}
        for code, key in enumerate(("train", "val")):
            frozen = plan["splits"][key]
            indices = frozen["indices"]
            if not isinstance(indices, (list, tuple)):
                raise ValueError("subset indices must be a sequence")
            chosen = np.asarray([_integer(index, "subset index") for index in indices], dtype=np.int64)
            if (not len(chosen) or len(np.unique(chosen)) != len(chosen)
                    or (chosen >= n).any() or not (stored[chosen] == code).all()):
                raise ValueError("invalid, duplicate, wrong-split or Test index in subset")
            if any(int(index) not in records for index in chosen):
                raise ValueError("subset has no complete chunk record")
            ordered_hash = hashlib.sha256("\n".join(map(str, self.ids[chosen])).encode()).hexdigest()
            if (_integer(frozen["rows"], "subset rows", 1) != len(chosen)
                    or ordered_hash != frozen["ordered_ids_sha256"]):
                raise ValueError("subset rows/order differs from frozen plan")
            selections[code] = (chosen, ordered_hash)
        self.indices, self.ordered_ids_hash = selections[split]
        self.targets = targets[self.indices]
        classes = 5 if task == "age" else 49
        if not ((self.targets >= 0) & (self.targets < classes)).all():
            raise ValueError("invalid target in frozen subset; task filtering is forbidden")
        self.class_counts = np.bincount(self.targets, minlength=classes)
        self._records = {int(index): records[int(index)] for index in self.indices}

    def __len__(self):
        return len(self.indices)

    def _file(self):
        import h5py
        if self._handle is None or self._pid != os.getpid():
            self.close()
            adapter, handle = None, None
            try:
                adapter = PrefixFile(self.path, self.logical_bytes,
                                     cached_blocks(self.range_report, self._report_hash), self.prefix_bytes)
                if adapter._signature != self._prefix_signature:
                    raise ValueError("prefix changed since dataset initialization")
                handle = h5py.File(adapter, "r")
                if not isinstance(handle.get("patches", getlink=True), h5py.HardLink):
                    raise ValueError("patches must be a local dataset")
                patches = handle["patches"]
                shape = (len(self.ids), len(CHANNELS), 256, 256)
                if (not isinstance(patches, h5py.Dataset) or patches.shape != shape
                        or patches.chunks != (1, len(CHANNELS), 256, 256)
                        or patches.dtype != np.dtype("float32")
                        or patches.is_virtual or patches.external):
                    raise ValueError("unexpected pixel schema")
                for index, expected in self._records.items():
                    chunk = patches.id.get_chunk_info_by_coord((index, 0, 0, 0))
                    if ((chunk.byte_offset, chunk.size) != expected
                            or tuple(chunk.chunk_offset or ()) != (index, 0, 0, 0)):
                        raise ValueError("physical chunk differs from frozen record")
            except BaseException:
                try:
                    if handle is not None:
                        handle.close()
                finally:
                    if adapter is not None:
                        adapter.close()
                raise
            self._adapter, self._handle, self._pid = adapter, handle, os.getpid()
            # Multiprocessing may exit without collecting the dataset. Close HDF5
            # before Python/HDF5 teardown, while its file-object adapter is alive.
            self._finalizer = Finalize(self, _close_handles, (handle, adapter), exitpriority=10)
        self._adapter._check_unchanged()
        return self._handle

    def __getitem__(self, index):
        # Resolve a scalar index before opening anything; slices cannot broaden pixel reads.
        index = operator.index(index)
        source = int(self.indices[index])
        raw = self._file()["patches"][source]
        pixels, _, _ = normalize_hdf5(raw, self.stats)
        return {"pixels": pixels, "target": np.int64(self.targets[index]),
                "sample_id": str(self.ids[source])}

    def close(self):
        handle, adapter = self._handle, self._adapter
        self._handle, self._adapter, self._pid = None, None, None
        if self._finalizer is not None:
            self._finalizer.cancel()
            self._finalizer = None
        _close_handles(handle, adapter)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_handle=None, _adapter=None, _pid=None, _finalizer=None)
        return state

    def __del__(self):
        self.close()
