#!/usr/bin/env python3
"""Probe pinned mmap pixels for the fixed eight pilot IDs. Default is offline dry-run.

Three predetermined rows per crop can disprove HDF5-derived equivalence, but cannot certify full
pixels or group/channel masks. --apply uses one cumulative body budget, no retries, no Test pixels.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from lunarrisk.benchmarks.http_range import ByteBudget, range_get
from lunarrisk.benchmarks.mmap_pair import FIXED_IDS, PATCH, build_plan, compare_row, row_digest
from lunarrisk.benchmarks.moonstone import CHANNELS, decode_name, normalize_hdf5


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pilot_arrays(path, records, statistics):
    import h5py
    import numpy as np
    result = {name: {} for name in statistics}
    with h5py.File(path, "r") as handle:
        channels = handle.attrs["channel_names"]
        if isinstance(channels, (str, bytes)):
            channels = json.loads(decode_name(channels))
        if tuple(map(decode_name, channels)) != CHANNELS or handle["patches"].shape != (8, 28, 256, 256):
            raise ValueError("pilot schema/channel mismatch")
        meta = handle["metadata"]
        if tuple(map(str, meta["patch_id"][:])) != FIXED_IDS:
            raise ValueError("pilot ID/order mismatch")
        for i, record in enumerate(records):
            for source, key in (("row_idx", "row_idx"), ("col_idx", "col_idx"), ("split", "stored_split"),
                                ("geology/dominant_unit_id", "unit_id")):
                if int(meta[source][i]) != int(record[key]):
                    raise ValueError("pilot and archived metadata disagree")
            if decode_name(meta["geology/dominant_unit_name"][i]) != record["unit_name"]:
                raise ValueError("pilot target name mismatch")
            raw = np.asarray(handle["patches"][i], dtype=np.float32)
            for name, stats in statistics.items():
                result[name][record["sample_id"]] = normalize_hdf5(raw, stats)[0]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--metadata-rows", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--hf-stats", type=Path, required=True)
    parser.add_argument("--git-stats", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--prior-transfer-bytes", type=int, required=True)
    parser.add_argument("--transfer-cap", type=int, default=1073741824)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.transfer_cap <= 1073741824:
        raise ValueError("first-round cumulative cap cannot exceed 1 GiB")
    with gzip.open(args.metadata_rows, "rt") as source:
        records = [r for r in csv.DictReader(source) if r["sample_id"] in FIXED_IDS]
    plan = build_plan(records, json.loads(args.listing.read_text()))
    budget = ByteBudget(args.transfer_cap, args.prior_transfer_bytes)
    budget.check(plan["new_body_bytes"])
    print(json.dumps({"mode": "apply" if args.apply else "dry_run", "plan": plan}, indent=2), flush=True)
    if not args.apply:
        return 0
    import numpy as np
    statistics = {name: json.loads(path.read_text()) for name, path in
                  (("hf", args.hf_stats), ("git_diagnostic_only", args.git_stats))}
    arrays = pilot_arrays(args.pilot, records, statistics)
    args.outdir.mkdir(parents=True, exist_ok=False)
    cache = args.outdir/"ranges"
    cache.mkdir()
    sources = (Path(__file__).resolve(), ROOT/"src/lunarrisk/benchmarks/mmap_pair.py",
               ROOT/"src/lunarrisk/benchmarks/http_range.py", ROOT/"src/lunarrisk/benchmarks/moonstone.py")
    result = {"plan": plan, "code_base_commit": os.environ.get("LUNARRISK_GIT_SHA", "UNKNOWN"),
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
              "inputs": {name: digest(path) for name, path in
                  (("pilot", args.pilot), ("hf_stats", args.hf_stats), ("git_stats", args.git_stats),
                   ("listing", args.listing), ("metadata_rows", args.metadata_rows))},
              "status": "BLOCKED_PARTIAL_DATA_ACCESS", "rows": []}
    assets = {asset["channel"]: asset for asset in plan["assets"]}
    try:
        for index, request in enumerate(plan["requests"]):
            channel = request["channel"]
            asset = assets[channel]
            raw = range_get(asset["url"], request["start"], request["length"], asset["bytes"], budget)
            filename = f"ranges/{index:04d}.bin"
            (args.outdir/filename).write_bytes(raw)
            official = np.frombuffer(raw, dtype="<f4")
            row = {**request, "cache": filename, "sha256": row_digest(raw), "comparisons": {}}
            ch = CHANNELS.index(channel)
            for name in statistics:
                derived = np.concatenate([arrays[name][sample][ch, request["local_y"]] for sample in request["ids"]])
                row["comparisons"][name] = compare_row(derived, official)
            result["rows"].append(row)
            if (index+1) % 6 == 0:
                print(json.dumps({"completed_channel": channel, "requests": index+1,
                                  "cumulative_body_bytes": budget.received}), flush=True)
        failed = any(r["comparisons"]["hf"]["not_close_pixels"] or
                     r["comparisons"]["hf"]["pixel_mask_disagreements"] for r in result["rows"])
        result["status"] = "HDF5_MMAP_EQUIVALENCE_FAILED" if failed else "PARTIAL_ROW_MATCH_NOT_EQUIVALENCE"
    except Exception as error:
        result["error_type"] = type(error).__name__  # Never archive signed redirect URLs or proxy values.
    result["transfer"] = {"prior_conservative_body_bytes": args.prior_transfer_bytes,
                          "new_body_bytes": budget.received-args.prior_transfer_bytes,
                          "cumulative_conservative_body_bytes": budget.received,
                          "completed_requests": len(result["rows"])}
    (args.outdir/"report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("plan", "rows")}, indent=2))
    return 2 if result["status"] != "PARTIAL_ROW_MATCH_NOT_EQUIVALENCE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
