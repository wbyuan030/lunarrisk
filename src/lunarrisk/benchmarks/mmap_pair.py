"""Pinned, exact-row mmap probes: a mismatch disproves equivalence; a match never proves it."""
from __future__ import annotations

import hashlib
import numpy as np

from .moonstone import CHANNELS

DATA_REVISION = "53099efd0f2db81eee96b28eae87e1efdd281310"
SHAPE = (23040, 46080)
PATCH = 256
PROBE_ROWS = (0, 128, 255)
RTOL, ATOL = 1e-5, 1e-6
FIXED_IDS = ("0", "1", "2", "3", "20", "21", "22", "23")


def build_plan(records, listing):
    """Merge only adjacent selected crops; never bridge a gap with unselected/Test pixels."""
    if listing["repository"] != "ayushprd/Moonstone" or listing["revision"] != DATA_REVISION:
        raise ValueError("unexpected repository/revision")
    if tuple(str(r["sample_id"]) for r in records) != FIXED_IDS:
        raise ValueError("probe requires the previously fixed eight IDs in original order")
    tiles = set()
    for row in records:
        if int(row["stored_split"]) not in (0, 1):
            raise ValueError("Test pixels are forbidden in the probe")
        r, c = int(row["row_idx"]), int(row["col_idx"])
        if not (0 <= r < 90 and 0 <= c < 180) or (r, c) in tiles:
            raise ValueError("invalid or duplicate tile coordinates")
        tiles.add((r, c))
    groups = []
    for record in sorted(records, key=lambda x: (int(x["row_idx"]), int(x["col_idx"]))):
        r, c, split = (int(record[k]) for k in ("row_idx", "col_idx", "stored_split"))
        if groups and (r, c, split) == (groups[-1]["row"], groups[-1]["last_col"]+1, groups[-1]["split"]):
            groups[-1]["last_col"] = c
            groups[-1]["ids"].append(str(record["sample_id"]))
        else:
            groups.append({"row": r, "first_col": c, "last_col": c, "split": split,
                           "ids": [str(record["sample_id"])]})
    files = {row["rfilename"]: row for row in listing["files"]}
    assets, requests = [], []
    total = SHAPE[0]*SHAPE[1]*4
    for channel in CHANNELS:
        path = f"pretraining/mmap/{channel}.npy"
        entry = files[path]
        if entry["size"] != total:
            raise ValueError("mmap size disagrees with headerless float32 grid")
        assets.append({"channel": channel, "path": path, "bytes": total,
                       "url": f"https://huggingface.co/datasets/ayushprd/Moonstone/resolve/{DATA_REVISION}/{path}",
                       "remote_lfs_sha256_claim": entry.get("lfs", {}).get("sha256"),
                       "whole_file_verified": False})
        for group in groups:
            for y in PROBE_ROWS:
                requests.append({"channel": channel, "ids": group["ids"], "tile_row": group["row"],
                    "first_col": group["first_col"], "local_y": y,
                    "start": ((group["row"]*PATCH+y)*SHAPE[1]+group["first_col"]*PATCH)*4,
                    "length": len(group["ids"])*PATCH*4})
    return {"protocol": "FIXED_ID_MMAP_ROW_PROBE", "dtype": "little-endian float32, headerless C order",
            "shape": list(SHAPE), "channels": list(CHANNELS), "sample_ids": list(FIXED_IDS),
            "probe_rows": list(PROBE_ROWS), "rtol": RTOL, "atol": ATOL,
            "test_pixels_requested": 0, "full_pixel_equivalence_validated": False,
            "full_channel_group_masks_validated": False, "assets": assets, "requests": requests,
            "new_body_bytes": sum(row["length"] for row in requests),
            "full_crops_body_bytes": len(records)*len(CHANNELS)*PATCH*PATCH*4,
            "full_crops_requests_after_merging": len(groups)*len(CHANNELS)*PATCH}


def compare_row(derived, remote):
    """Follow upstream nonfinite-to-zero and finite/nonzero presence; remote is the reference."""
    derived = np.asarray(derived, dtype=np.float32)
    remote = np.asarray(remote, dtype=np.float32)
    if derived.shape != remote.shape or derived.ndim != 1:
        raise ValueError("paired pixel rows must have identical one-dimensional shapes")
    if not np.isfinite(derived).all():
        raise ValueError("nonfinite derived normalization")
    remote_valid = np.isfinite(remote) & (remote != 0)
    official = np.where(np.isfinite(remote), remote, np.float32(0))
    absolute = np.abs(derived.astype(np.float64)-official.astype(np.float64))
    bad = ~np.isclose(derived, official, rtol=RTOL, atol=ATOL)
    indices = np.flatnonzero(bad)[:3]
    return {"compared_pixels": int(derived.size), "not_close_pixels": int(bad.sum()),
            "pixel_mask_disagreements": int(((derived != 0) != remote_valid).sum()),
            "remote_nonfinite_pixels": int((~np.isfinite(remote)).sum()),
            "max_abs_error": float(absolute.max()) if absolute.size else 0.0,
            "sum_abs_error": float(absolute.sum()),
            "examples": [{"x": int(x), "derived": float(derived[x]), "official": float(official[x])}
                         for x in indices]}


def row_digest(raw):
    return hashlib.sha256(raw).hexdigest()
