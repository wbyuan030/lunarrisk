#!/usr/bin/env python3
"""Produce an exact, offline download allowlist from saved HF metadata. Never downloads assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

SMALL_FILES = {
    "ayushprd/Moonstone": ("README.md", "benchmark/geologic_units.json",
                           "pretraining/channel_stats.json", "pretraining/mmap/channel_index.json"),
    "nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model": (
        "README.md", "backbone/config.yaml", "tokenizers/WAC_vis/config.yaml"),
}


def build_plan(dataset_info, model_info, cap=1073741824):
    if cap < 1 or cap > 1073741824:
        raise ValueError("first-round transfer cap must be in (0, 1 GiB]")
    assets = []
    for kind, metadata in (("dataset", dataset_info), ("model", model_info)):
        repository, revision = metadata["id"], metadata["sha"]
        if repository not in SMALL_FILES or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("unexpected repository or unpinned revision")
        files = {row["rfilename"]: row for row in metadata["siblings"]}
        for name in SMALL_FILES[repository]:
            row = files[name]
            if not 0 < row["size"] <= 128*1024:
                raise ValueError("small-file allowlist exceeds declared limit")
            prefix = "datasets/" if kind == "dataset" else ""
            assets.append({"repository": repository, "revision": revision, "path": name,
                           "url": f"https://huggingface.co/{prefix}{repository}/resolve/{revision}/{name}",
                           "bytes": row["size"], "operation": "bounded_small_get"})
        large_name = "benchmark/lunar_patches_v4.h5" if kind == "dataset" else "backbone/checkpoint.pt"
        large = files[large_name]
        prefix = "datasets/" if kind == "dataset" else ""
        assets.append({"repository": repository, "revision": revision, "path": large_name,
                       "url": f"https://huggingface.co/{prefix}{repository}/resolve/{revision}/{large_name}",
                       "bytes": large["size"], "operation": "strict_range_only",
                       "complete_download_allowed": False, "remote_lfs_sha256_claim": large.get("lfs", {}).get("sha256"),
                       "checksum_status": "remote listing claim; complete file not verified"})
    small_bytes = sum(row["bytes"] for row in assets if row["operation"] == "bounded_small_get")
    return {"mode": "dry_run", "network_requests": 0, "transfer_cap_bytes": cap,
            "small_allowlist_bytes": small_bytes, "assets": assets,
            "full_benchmark_or_pretraining": "NOT_AUTHORIZED_IN_INITIAL_ROUND",
            "range_policy": "206 and exact Content-Range required before reading body; shared cumulative budget",
            "unlisted_assets": "deny; metadata listing is not a download authorization"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moonstone-info", type=Path, required=True)
    parser.add_argument("--nasa-info", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    plan = build_plan(json.loads(args.moonstone_info.read_text()), json.loads(args.nasa_info.read_text()))
    text = json.dumps(plan, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as output:
            output.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
