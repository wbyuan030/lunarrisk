"""Small, strict HTTP range reads for inspecting large public assets without downloading them.

The caller shares one ByteBudget across all metadata and samples. A server that ignores Range is
rejected before its body is read. No upstream Python or serialized model is executed by this module.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import io
import re
import urllib.request


@dataclass
class ByteBudget:
    limit: int
    received: int = 0

    def __post_init__(self):
        if self.limit < 0 or not 0 <= self.received <= self.limit:
            raise ValueError("invalid transfer budget")

    def check(self, requested):
        if requested < 0 or requested > self.limit - self.received:
            raise ValueError("BLOCKED_TRANSFER_BUDGET: request exceeds remaining body-byte budget")

    def read(self, response, size):
        self.check(size)
        result = bytearray()
        while len(result) < size:
            piece = response.read(min(65536, size - len(result)))
            self.received += len(piece)
            result.extend(piece)
            if not piece:
                break
        return bytes(result)


def small_get(url, budget, max_bytes=1024 * 1024, opener=urllib.request.urlopen):
    """Read a bounded metadata body; actual bytes, including failed size checks, consume budget."""
    budget.check(max_bytes + 1)
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with opener(request, timeout=30) as response:
        raw = budget.read(response, max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("metadata body exceeds declared limit")
    return raw


def range_get(url, start, length, total_size, budget, opener=urllib.request.urlopen):
    if start < 0 or length <= 0 or start + length > total_size:
        raise ValueError("invalid asset byte range")
    budget.check(length)
    end = start + length - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                                  "Accept-Encoding": "identity"})
    with opener(request, timeout=30) as response:
        if response.status != 206:
            raise ValueError(f"BLOCKED_PARTIAL_DATA_ACCESS: HTTP {response.status}; body not read")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match or tuple(map(int, match.groups())) != (start, end, total_size):
            raise ValueError("invalid Content-Range; body not read")
        if response.headers.get("Content-Encoding", "identity") not in ("identity", ""):
            raise ValueError("encoded range response; body not read")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != length:
            raise ValueError("invalid Content-Length; body not read")
        raw = budget.read(response, length)
    if len(raw) != length:
        raise ValueError("truncated byte range")
    return raw


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file object for h5py; only a bounded LRU of blocks stays in memory."""

    def __init__(self, url, size, budget, block_size=65536, cache_blocks=64,
                 opener=urllib.request.urlopen):
        super().__init__()
        if size < 1 or block_size < 1 or cache_blocks < 1:
            raise ValueError("invalid range-file configuration")
        self.url, self.size, self.budget = url, size, budget
        self.block_size, self.cache_blocks, self.opener = block_size, cache_blocks, opener
        self.position, self.requests = 0, 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        if whence not in origin or origin[whence] + offset < 0:
            raise ValueError("invalid seek")
        self.position = origin[whence] + offset
        return self.position

    def read(self, size=-1):
        self._checkClosed()
        size = max(0, self.size-self.position) if size < 0 else min(size, max(0, self.size-self.position))
        result = bytearray()
        while len(result) < size:
            number, within = divmod(self.position, self.block_size)
            if number not in self.cache:
                start = number * self.block_size
                self.cache[number] = range_get(self.url, start, min(self.block_size, self.size-start),
                                              self.size, self.budget, self.opener)
                self.requests += 1
                if len(self.cache) > self.cache_blocks:
                    self.cache.popitem(last=False)
            block = self.cache[number]
            self.cache.move_to_end(number)
            piece = block[within:within + size-len(result)]
            result.extend(piece)
            self.position += len(piece)
        return bytes(result)

    def readinto(self, buffer):
        raw = self.read(len(buffer))
        buffer[:len(raw)] = raw
        return len(raw)
