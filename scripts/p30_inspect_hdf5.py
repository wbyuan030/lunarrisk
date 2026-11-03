#!/usr/bin/env python3
"""Inspect an allowlisted HDF5 via strict Range requests; default is a zero-transfer dry-run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunarrisk.benchmarks.http_range import ByteBudget, HTTPRangeFile


def json_value(value):
    import numpy as np
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def inspect(stream):
    import h5py
    layout = {}
    metadata = {}
    with h5py.File(stream, "r") as handle:
        attributes = {key: json_value(value) for key, value in handle.attrs.items()}

        def record(name, obj):
            if isinstance(obj, h5py.Dataset):
                layout[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype),
                                "chunks": obj.chunks, "compression": obj.compression,
                                "attributes": {key: json_value(value) for key, value in obj.attrs.items()}}
                if name.startswith("metadata/") and obj.size * max(8, obj.dtype.itemsize) < 8*1024*1024:
                    metadata[name] = json_value(obj[...])
        handle.visititems(record)
    return {"attributes": attributes, "layout": layout, "metadata": metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--transfer-cap", type=int, default=64*1024*1024)
    parser.add_argument("--prior-transfer-bytes", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.transfer_cap <= 1073741824:
        raise ValueError("initial transfer cap must be in (0, 1 GiB]")
    plan = {"url": args.url, "asset_bytes": args.size, "maximum_session_body_bytes": args.transfer_cap,
            "prior_body_bytes": args.prior_transfer_bytes, "mode": "apply" if args.apply else "dry_run"}
    if not args.apply:
        print(json.dumps(plan, indent=2))
        return 0
    if args.out.exists():
        raise ValueError("output already exists; use a new audit path")
    budget = ByteBudget(args.transfer_cap, args.prior_transfer_bytes)
    stream = HTTPRangeFile(args.url, args.size, budget)
    result = {"plan": plan}
    try:
        result.update(inspect(stream))
        result["status"] = "REAL_HDF5_METADATA_READ_NOT_LOADER_EQUIVALENCE"
    except Exception as error:
        # Keep a reviewable blocker without exposing signed redirect URLs or machine paths.
        result.update(status="BLOCKED_PARTIAL_DATA_ACCESS", error_type=type(error).__name__,
                      error=str(error).split("?")[0])
    finally:
        stream.close()
    result["transfer"] = {"accounted_body_bytes": budget.received,
                          "new_body_bytes": budget.received-args.prior_transfer_bytes,
                          "range_requests": stream.requests}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "metadata"}, indent=2))
    return 0 if "layout" in result else 2


if __name__ == "__main__":
    raise SystemExit(main())
