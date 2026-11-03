#!/usr/bin/env python3
"""Extract fixed first Train/Val Age samples via strict ranges; default prints the plan only.

This creates HDF5_DERIVED_PILOT data, not an official mmap-equivalence certificate. Never selects
examples by model score or reads Test pixels. Original patch IDs and stored splits are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunarrisk.benchmarks.http_range import ByteBudget, HTTPRangeFile
from lunarrisk.benchmarks.moonstone import CHANNELS, age_target, decode_name


def sample_plan(metadata, per_split):
    import numpy as np
    if per_split < 1 or per_split > 8:
        raise ValueError("initial pilot requires 1..8 samples per Train/Val split")
    if metadata.get("status") != "REAL_HDF5_METADATA_READ_NOT_LOADER_EQUIVALENCE":
        raise ValueError("real HDF5 metadata must be verified before sample extraction")
    table = metadata["metadata"]
    splits = np.asarray(table["metadata/split"])
    names = table["metadata/geology/dominant_unit_name"]
    valid = np.asarray([age_target(name) >= 0 for name in names])
    indices = []
    for split in (0, 1):
        selected = np.flatnonzero((splits == split) & valid)[:per_split]
        if len(selected) < per_split:
            raise ValueError("not enough valid stored Train/Val samples")
        indices.extend(selected.tolist())
    return {"protocol": "HDF5_DERIVED_PILOT", "selection": "first valid original indices per stored Train/Val split",
            "indices": indices, "patch_ids": [table["metadata/patch_id"][index] for index in indices],
            "stored_splits": [int(splits[index]) for index in indices], "test_pixels_requested": 0,
            "per_split": per_split}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=4)
    parser.add_argument("--prior-transfer-bytes", type=int, required=True)
    parser.add_argument("--transfer-cap", type=int, default=256*1024*1024)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.transfer_cap > 1073741824:
        raise ValueError("initial transfer cap cannot exceed 1 GiB")
    metadata = json.loads(args.metadata.read_text())
    plan = sample_plan(metadata, args.per_split)
    plan["source_metadata_sha256"] = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    print(json.dumps({**plan, "mode": "apply" if args.apply else "dry_run"}, indent=2), flush=True)
    if not args.apply:
        return 0
    import h5py
    import numpy as np
    if args.out.exists():
        raise ValueError("pilot output already exists")
    budget = ByteBudget(args.transfer_cap, args.prior_transfer_bytes)
    remote = metadata["plan"]
    stream = HTTPRangeFile(remote["url"], remote["asset_bytes"], budget, block_size=1024*1024)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"plan": plan, "source_url": remote["url"]}
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    try:
        with h5py.File(stream, "r") as source, h5py.File(partial, "x") as output:
            # Check shape/order against the pinned remote metadata before any pixel extraction.
            expected = metadata["layout"]["patches"]["shape"]
            if list(source["patches"].shape) != expected or expected[1:] != [28, 256, 256]:
                raise ValueError("source patch schema changed")
            channel_names = source.attrs["channel_names"]
            if isinstance(channel_names, (str, bytes)):
                channel_names = json.loads(decode_name(channel_names))
            if tuple(map(decode_name, channel_names)) != CHANNELS:
                raise ValueError("source channel order differs from the pinned contract")
            output.attrs["channel_names"] = json.dumps(CHANNELS)
            output.attrs["p30_protocol"] = "HDF5_DERIVED_PILOT"
            patches = output.create_dataset("patches", shape=(len(plan["indices"]), 28, 256, 256),
                                             dtype="float32", chunks=(1, 28, 256, 256), compression="gzip")
            sample_hashes = {}
            for row, index in enumerate(plan["indices"]):
                raw = np.asarray(source["patches"][index], dtype=np.float32)
                patches[row] = raw
                sample_hashes[str(plan["patch_ids"][row])] = hashlib.sha256(raw.tobytes()).hexdigest()
            for name, values in metadata["metadata"].items():
                if len(values) != expected[0]:
                    continue
                parent, leaf = name.rsplit("/", 1)
                values = [values[index] for index in plan["indices"]]
                array = np.asarray(values)
                if array.dtype.kind in ("U", "O"):
                    array = np.asarray(values, dtype=h5py.string_dtype("utf-8"))
                output.require_group(parent).create_dataset(leaf, data=array)
        if args.out.exists():
            raise ValueError("output appeared during extraction; partial retained")
        partial.rename(args.out)
        result.update(status="REAL_PIXEL_PILOT_EXTRACTED_NOT_EQUIVALENCE", sample_sha256=sample_hashes,
                      pilot_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())
    except Exception as error:
        result.update(status="BLOCKED_PARTIAL_DATA_ACCESS", error_type=type(error).__name__,
                      error=str(error).split("?")[0])
    finally:
        stream.close()
    result["transfer"] = {"accounted_body_bytes": budget.received,
                          "new_body_bytes": budget.received-args.prior_transfer_bytes,
                          "range_requests": stream.requests}
    args.out.with_suffix(".audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"].startswith("REAL_PIXEL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
