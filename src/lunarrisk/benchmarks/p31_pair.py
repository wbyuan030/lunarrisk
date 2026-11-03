"""P31-02 coverage-only selection, exact full-row plans and bounded comparison utilities."""
from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from .http_range import ByteBudget, range_get
from .mmap_pair import DATA_REVISION, SHAPE, PATCH, RTOL, ATOL
from .moonstone import CHANNELS, GROUP_CHANNELS, UPSTREAM_COMMIT, age_target, normalize_hdf5

CAP = 1 << 30
PRIOR_ACTUAL = 78502175
PRIOR_CONSERVATIVE = 83468143
HDF5_BYTES = 74449119347
HDF5_LIMIT = 64 << 20
HDF5_REQUESTS = 2048
REPOSITORY = "ayushprd/Moonstone"


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def select_records(inspection, mapping):
    """One crop per split if it covers all globally available channels and partial M3.

    Minimum two crops follows the requirement to exercise both stored splits. Fail closed if a
    two-crop cover is unavailable; a larger selection must be separately reviewed and locked.
    Targets are checked after coverage selection and never rank candidates.
    """
    if inspection.get("status") != "REAL_HDF5_METADATA_READ_NOT_LOADER_EQUIVALENCE":
        raise ValueError("verified real metadata required")
    if tuple(inspection["attributes"]["channel_names"]) != CHANNELS:
        raise ValueError("channel order mismatch")
    m = inspection["metadata"]
    ids, splits = m["metadata/patch_id"], m["metadata/split"]
    if len(set(map(str, ids))) != len(ids):
        raise ValueError("duplicate metadata IDs")
    present = np.asarray(m["metadata/has_channel"], dtype=bool)
    if present.shape != (len(ids), len(CHANNELS)):
        raise ValueError("invalid presence shape")
    available = present[np.isin(splits, [0, 1])].any(axis=0)
    if any(not any(available[CHANNELS.index(c)] for c in group)
           for group in GROUP_CHANNELS.values()):
        raise ValueError("a complete modality group is unavailable")
    records = []
    for split in (0, 1):
        candidates = [i for i in range(len(ids)) if splits[i] == split
                      and abs(float(m["metadata/center_lat"][i])) <= 60
                      and np.all(present[i, available])
                      and 0 < float(m["metadata/m3/m3_valid_fraction"][i]) < 1]
        if not candidates:
            raise ValueError("no two-crop coverage solution; revise plan offline")
        i = min(candidates, key=lambda j: (int(ids[j]), j))
        name, unit = m["metadata/geology/dominant_unit_name"][i], int(m["metadata/geology/dominant_unit_id"][i])
        if mapping.get(name) != unit or not 1 <= unit <= 49:
            raise ValueError("label map mismatch")
        records.append({"index": i, "sample_id": str(ids[i]), "stored_split": split,
                        "row_idx": int(m["metadata/row_idx"][i]),
                        "col_idx": int(m["metadata/col_idx"][i]),
                        "center_lat": float(m["metadata/center_lat"][i]),
                        "unit_name": name, "unit_id": unit, "geology_target": unit-1,
                        "age_target": age_target(name), "metadata_has_channel": present[i].tolist(),
                        "m3_valid_fraction": float(m["metadata/m3/m3_valid_fraction"][i]),
                        "valid_fraction": float(m["metadata/valid_fraction"][i])})
    return records, [c for c, flag in zip(CHANNELS, available) if not flag]


def ranges(records):
    """Never coalesce across a crop, split, gap, channel or scene."""
    seen_ids, seen_tiles = set(), set()
    for record in records:
        row, col = record["row_idx"], record["col_idx"]
        if (record["stored_split"] not in (0, 1) or not 0 <= row < SHAPE[0]//PATCH
                or not 0 <= col < SHAPE[1]//PATCH or record["sample_id"] in seen_ids
                or (row, col) in seen_tiles):
            raise ValueError("invalid, Test or duplicate crop")
        seen_ids.add(record["sample_id"])
        seen_tiles.add((row, col))
    for record in records:
        for channel in CHANNELS:
            for y in range(PATCH):
                start = ((record["row_idx"]*PATCH+y)*SHAPE[1]+record["col_idx"]*PATCH)*4
                yield {"sample_id": record["sample_id"], "channel": channel, "y": y,
                       "start": start, "length": PATCH*4}


def build_plan(inspection, listing, stats, mapping, hashes):
    if listing["repository"] != REPOSITORY or listing["revision"] != DATA_REVISION:
        raise ValueError("unpinned source")
    if sorted(mapping.values()) != list(range(1, 50)):
        raise ValueError("invalid geology mapping")
    records, unavailable = select_records(inspection, mapping)
    files = {entry["rfilename"]: entry for entry in listing["files"]}
    assets = []
    for path, size in [("benchmark/lunar_patches_v4.h5", HDF5_BYTES)]+[
            (f"pretraining/mmap/{c}.npy", SHAPE[0]*SHAPE[1]*4) for c in CHANNELS]:
        entry = files[path]
        if entry["size"] != size:
            raise ValueError("asset size/headerless layout mismatch")
        assets.append({"path": path, "bytes": size,
                       "url": f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{DATA_REVISION}/{path}",
                       "remote_lfs_sha256_claim": entry.get("lfs", {}).get("sha256"),
                       "whole_file_verified": False})
    # Validate fixed statistics even before downloading pixels.
    normalize_hdf5(np.zeros((28, 1, 1), dtype=np.float32), stats)
    requests = list(ranges(records))
    mmap_bytes = sum(r["length"] for r in requests)
    plan = {"protocol": "P31_02_LOCKED_FULL_CROP_PAIR", "status": "LOCKED_OFFLINE_NOT_EXECUTED",
            "repository": REPOSITORY, "revision": DATA_REVISION, "upstream_commit": UPSTREAM_COMMIT,
            "records": records, "channels": list(CHANNELS), "groups": GROUP_CHANNELS,
            "shape": [28, 256, 256], "mmap_shape": list(SHAPE),
            "selection": "minimum one Train and one Val; each covers all available Train/Val channels, nonpolar <=60 degrees and partial M3; lowest numeric patch ID per split; no scores or targets used for ranking",
            "unavailable_train_val_channels": unavailable,
            "raw_zero_coverage": "UNKNOWN_UNTIL_PIXELS; require finite raw zero in downloaded crops or fail coverage; no adaptive extra downloads",
            "rtol": RTOL, "atol": ATOL, "mask": "exact finite/nonzero normalized full-pixel masks and channel/group ANY presence",
            "normalization": "fixed HF stats; raw nonfinite restored to zero; finite raw zero retained; no second radar log",
            "input_sha256": hashes, "assets": assets,
            "mmap_ranges": {"representation": "one run per sample/channel; expand all y=0..255",
                "dtype": "headerless little-endian float32 C order", "row_stride_bytes": SHAPE[1]*4,
                "rows_per_run": PATCH, "row_length_bytes": PATCH*4,
                "runs": [r for r in requests if r["y"] == 0],
                "expanded_requests_sha256": canonical_hash(requests)},
            "transfer": {"cap_bytes": CAP, "prior_actual_bytes": PRIOR_ACTUAL,
                "prior_conservative_bytes": PRIOR_CONSERVATIVE, "mmap_requests": len(requests),
                "mmap_body_bytes": mmap_bytes, "hdf5_body_upper_bound": HDF5_LIMIT,
                "hdf5_request_upper_bound": HDF5_REQUESTS,
                "new_body_upper_bound": mmap_bytes+HDF5_LIMIT,
                "cumulative_conservative_upper_bound": PRIOR_CONSERVATIVE+mmap_bytes+HDF5_LIMIT,
                "request_upper_bound": len(requests)+HDF5_REQUESTS,
                "hdf5_exact_requests_bytes": "UNKNOWN before apply; exact h5py reads, no aligned blocks/gap overfetch; hard caps apply",
                "redirect_headers": "not counted as range requests; at most 8 per range, redirect bodies not read",
                "elapsed_time": "UNKNOWN; no throughput measurement or completion-time promise"},
            "execution": {"workers": 4, "persistent_session_per_worker": True, "retries": 0,
                "hdf5_access": "selected chunked patches only; metadata/scalars may be fetched; exact requested bytes, no read-ahead",
                "cache_files_upper_bound": len(requests)+HDF5_REQUESTS+len(records)+1,
                "cache_body_upper_bound": mmap_bytes+HDF5_LIMIT+len(records)*28*PATCH*PATCH*4,
                "cache_note": "range files + raw crop .npy files + report; filesystem allocation and JSON overhead extra",
                "test_pixels_requested": 0, "requires_fresh_output_directory": True}}
    if plan["transfer"]["cumulative_conservative_upper_bound"] > CAP:
        raise ValueError("cumulative initial budget exceeded")
    plan["lock_sha256"] = canonical_hash(plan)
    return plan


def check_lock(plan):
    content = dict(plan)
    lock = content.pop("lock_sha256", None)
    if canonical_hash(content) != lock:
        raise ValueError("locked plan hash mismatch")


def compare_crop(raw, remote, stats):
    raw, remote = np.asarray(raw, dtype=np.float32), np.asarray(remote, dtype=np.float32)
    if raw.shape != (28, PATCH, PATCH) or remote.shape != raw.shape:
        raise ValueError("full 28x256x256 crops required")
    derived, hc, hg = normalize_hdf5(raw, stats)
    rv = np.isfinite(remote) & (remote != 0)
    dv = np.isfinite(derived) & (derived != 0)
    official = np.where(np.isfinite(remote), remote, np.float32(0))
    close = np.isclose(derived, official, rtol=RTOL, atol=ATOL)
    rows = []
    for i, name in enumerate(CHANNELS):
        error = np.abs(derived[i].astype(np.float64)-official[i].astype(np.float64))
        rows.append({"channel": name, "compared_pixels": PATCH*PATCH,
                     "raw_finite_pixels": int(np.isfinite(raw[i]).sum()),
                     "raw_missing_pixels": int((~np.isfinite(raw[i])).sum()),
                     "raw_finite_zero_pixels": int((np.isfinite(raw[i]) & (raw[i] == 0)).sum()),
                     "derived_valid_pixels": int(dv[i].sum()), "derived_missing_pixels": int((~dv[i]).sum()),
                     "mmap_valid_pixels": int(rv[i].sum()), "mmap_missing_pixels": int((~rv[i]).sum()),
                     "mmap_nonfinite_pixels": int((~np.isfinite(remote[i])).sum()),
                     "not_close_pixels": int((~close[i]).sum()),
                     "mask_disagreements": int((dv[i] != rv[i]).sum()),
                     "max_abs_error": float(error.max())})
    rc = rv.reshape(28, -1).any(axis=1)
    rg = [bool(any(rc[CHANNELS.index(c)] for c in group)) for group in GROUP_CHANNELS.values()]
    passed = bool(close.all() and np.array_equal(dv, rv) and np.array_equal(hc, rc)
                  and np.array_equal(hg, rg))
    return {"numerical_and_masks_pass": passed, "channels": rows,
            "derived_channel_presence": hc.astype(bool).tolist(), "mmap_channel_presence": rc.tolist(),
            "derived_group_presence": hg.astype(bool).tolist(), "mmap_group_presence": rg}


class ExactRangeFile(io.RawIOBase):
    """Unbuffered h5py adapter: request exactly read(size), cap request count and all bodies.

    Unlike the general block adapter, this never rounds a read into neighboring patch chunks.
    No retry/cache replay: every attempted read is recorded; repeated HDF5 metadata reads count.
    """
    def __init__(self, asset, budget, opener, sink, request_limit=HDF5_REQUESTS):
        super().__init__()
        self.asset, self.budget, self.opener, self.sink = asset, budget, opener, sink
        self.position, self.attempts, self.request_limit = 0, 0, request_limit

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = {0: 0, 1: self.position, 2: self.asset["bytes"]}[whence]+offset
        if not 0 <= position <= self.asset["bytes"]:
            raise ValueError("invalid HDF5 seek")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            raise ValueError("unbounded HDF5 read forbidden")
        size = min(size, self.asset["bytes"]-self.position)
        if not size:
            return b""
        if self.attempts >= self.request_limit:
            raise ValueError("HDF5 request ceiling reached")
        self.attempts += 1
        start = self.position
        raw = range_get(self.asset["url"], start, size, self.asset["bytes"], self.budget, self.opener)
        self.sink({"start": start, "length": size}, raw)
        self.position += len(raw)
        return raw

    def readinto(self, target):
        raw = self.read(len(target))
        target[:len(raw)] = raw
        return len(raw)


def reserved_workers(requests, workers, remaining):
    """Reserve every worker's complete allocation before starting any network request."""
    if not 1 <= workers <= 8:
        raise ValueError("workers must be 1..8")
    lanes = [requests[i::workers] for i in range(workers)]
    limits = [sum(r["length"] for r in lane) for lane in lanes]
    if sum(limits) > remaining:
        raise ValueError("cannot reserve workers within cumulative budget")
    return [(lane, ByteBudget(limit)) for lane, limit in zip(lanes, limits)]


def run_workers(reservations, worker):
    with ThreadPoolExecutor(max_workers=len(reservations)) as pool:
        futures = [pool.submit(worker, i, lane, budget)
                   for i, (lane, budget) in enumerate(reservations)]
        # All workers finish before the caller records even a failure's received bytes.
        return [future.result() for future in futures]


class PersistentOpener:
    """requests session behind the existing strict range_get header-before-body validator."""
    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0, pool_connections=4,
                                                                    pool_maxsize=1))

    def __call__(self, request, timeout):
        from urllib.parse import urljoin, urlparse
        url = request.full_url
        for _ in range(9):
            if urlparse(url).scheme != "https":
                raise ValueError("non-HTTPS redirect forbidden")
            response = self.session.get(url, headers=dict(request.header_items()), timeout=timeout,
                                        stream=True, allow_redirects=False)
            if response.status_code not in (301, 302, 303, 307, 308):
                return _Response(response)
            location = response.headers.get("Location")
            response.close()  # Never consume redirect bodies (requests' automatic redirects do).
            if not location:
                raise ValueError("redirect missing location")
            url = urljoin(url, location)
        raise ValueError("redirect ceiling reached")

    def close(self): self.session.close()


class _Response:
    def __init__(self, response):
        self.response, self.status, self.headers = response, response.status_code, response.headers
    def __enter__(self): return self
    def read(self, size): return self.response.raw.read(size, decode_content=False)
    def __exit__(self, *args): self.response.close()
