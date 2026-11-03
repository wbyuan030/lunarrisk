#!/usr/bin/env python3
"""Offline plan lock and bounded full-crop comparison. Default: no network, no output creation.

--write-plan creates the offline lock; --apply is a separate explicit execution. HDF5 request sizes
are unknown until h5py traverses the pinned file, bounded by the locked byte/request ceilings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunarrisk.benchmarks.http_range import ByteBudget, range_get
from lunarrisk.benchmarks.moonstone import CHANNELS, decode_name
from lunarrisk.benchmarks.p31_pair import (
    CAP, HDF5_LIMIT, PRIOR_CONSERVATIVE, ExactRangeFile, PersistentOpener, build_plan,
    canonical_hash, check_lock, compare_crop, digest, ranges, reserved_workers, run_workers)


def read_json(path):
    return json.loads(path.read_text())


def inputs(args):
    return {"metadata": args.metadata, "listing": args.listing,
            "hf_stats": args.hf_stats, "class_map": args.class_map}


def make_plan(args):
    paths = inputs(args)
    hashes = {name: digest(path) for name, path in paths.items()}
    for relative in ("scripts/p31_loader_pair.py", "src/lunarrisk/benchmarks/p31_pair.py",
                     "src/lunarrisk/benchmarks/http_range.py", "src/lunarrisk/benchmarks/moonstone.py",
                     "src/lunarrisk/benchmarks/mmap_pair.py"):
        hashes[relative] = digest(ROOT/relative)
    return build_plan(read_json(args.metadata), read_json(args.listing),
                      read_json(args.hf_stats), read_json(args.class_map), hashes)


def execute(args, plan):
    import h5py
    import numpy as np
    out = args.outdir
    out.mkdir(parents=True, exist_ok=False)
    (out / "hdf5_ranges").mkdir()
    (out / "mmap_ranges").mkdir()
    started = time.monotonic()
    assets = {asset["path"]: asset for asset in plan["assets"]}
    hbudget = ByteBudget(HDF5_LIMIT)
    requests = list(ranges(plan["records"]))
    reservations = reserved_workers(requests, plan["execution"]["workers"],
                                    CAP-args.prior_transfer_bytes-HDF5_LIMIT)
    source_files = [Path(__file__).resolve(), ROOT/"src/lunarrisk/benchmarks/p31_pair.py",
                    ROOT/"src/lunarrisk/benchmarks/http_range.py", ROOT/"src/lunarrisk/benchmarks/moonstone.py",
                    ROOT/"src/lunarrisk/benchmarks/mmap_pair.py"]
    result = {"status": "BLOCKED_REAL_IO", "plan_lock_sha256": plan["lock_sha256"],
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
              "input_sha256": plan["input_sha256"], "hdf5_ranges": [], "mmap_ranges": [],
              "comparisons": [], "whole_file_hashes_verified": False}
    raw_crops = {}
    official = {r["sample_id"]: np.empty((28, 256, 256), dtype=np.float32) for r in plan["records"]}
    opener = PersistentOpener()

    def save_hdf5(request, raw):
        name = f"hdf5_ranges/{len(result['hdf5_ranges']):05d}.bin"
        (out/name).write_bytes(raw)
        result["hdf5_ranges"].append({**request, "sha256": hashlib.sha256(raw).hexdigest(), "cache": name})

    stream = ExactRangeFile(assets["benchmark/lunar_patches_v4.h5"], hbudget, opener, save_hdf5)
    try:
        with h5py.File(stream, "r") as source:
            channels = source.attrs["channel_names"]
            if isinstance(channels, (str, bytes)):
                channels = json.loads(decode_name(channels))
            patches = source["patches"]
            if (tuple(map(decode_name, channels)) != CHANNELS or patches.shape != (16200, 28, 256, 256)
                    or patches.chunks != (1, 28, 256, 256) or str(patches.dtype) != "float32"
                    or patches.compression != "gzip"):
                raise ValueError("pinned HDF5 schema mismatch")
            for record in plan["records"]:
                i, sample = record["index"], record["sample_id"]
                # Only scalar metadata and the two selected patch chunks; no Test pixels.
                for field, key in (("patch_id", "sample_id"), ("row_idx", "row_idx"),
                                   ("col_idx", "col_idx"), ("split", "stored_split"),
                                   ("geology/dominant_unit_id", "unit_id")):
                    if int(source["metadata/"+field][i]) != int(record[key]):
                        raise ValueError("remote ID/label/split mismatch")
                if decode_name(source["metadata/geology/dominant_unit_name"][i]) != record["unit_name"]:
                    raise ValueError("remote unit name mismatch")
                if not np.array_equal(source["metadata/has_channel"][i], record["metadata_has_channel"]):
                    raise ValueError("remote metadata presence mismatch")
                raw = np.asarray(patches[i], dtype=np.float32)
                raw_crops[sample] = raw
                np.save(out/f"raw_{sample}.npy", raw, allow_pickle=False)
        stream.close()
        opener.close()

        def worker(worker_id, lane, budget):
            session = PersistentOpener()
            rows, attempts = [], 0
            try:
                for request in lane:
                    asset = assets[f"pretraining/mmap/{request['channel']}.npy"]
                    attempts += 1
                    raw = range_get(asset["url"], request["start"], request["length"], asset["bytes"], budget, session)
                    name = f"mmap_ranges/{request['sample_id']}_{CHANNELS.index(request['channel']):02d}_{request['y']:03d}.bin"
                    (out/name).write_bytes(raw)
                    official[request["sample_id"]][CHANNELS.index(request["channel"]), request["y"]] = np.frombuffer(raw, dtype="<f4")
                    rows.append({**request, "cache": name, "sha256": hashlib.sha256(raw).hexdigest()})
                    if len(rows) % 512 == 0:
                        print(json.dumps({"worker": worker_id, "completed_ranges": len(rows),
                                          "returned_body_bytes": budget.received}), flush=True)
            except Exception as error:
                return {"worker": worker_id, "attempts": attempts, "rows": rows, "error_type": type(error).__name__}
            finally:
                session.close()
            return {"worker": worker_id, "attempts": attempts, "rows": rows}

        workers = run_workers(reservations, worker)
        result["workers"] = [{k: v for k, v in w.items() if k != "rows"} for w in workers]
        result["mmap_ranges"] = sorted([row for w in workers for row in w["rows"]],
                                             key=lambda r: (r["sample_id"], r["channel"], r["y"]))
        if any("error_type" in w for w in workers):
            raise ValueError("worker failed; no retries or automatic continuation")
        stats = read_json(args.hf_stats)
        for record in plan["records"]:
            sample = record["sample_id"]
            comparison = compare_crop(raw_crops[sample], official[sample], stats)
            comparison.update(record=record, raw_crop_sha256=digest(out/f"raw_{sample}.npy"),
                              raw_pixel_sha256=hashlib.sha256(raw_crops[sample].tobytes()).hexdigest(),
                              mmap_pixel_sha256=hashlib.sha256(official[sample].tobytes()).hexdigest())
            result["comparisons"].append(comparison)
        comparisons = result["comparisons"]
        active = np.any([c["mmap_channel_presence"] for c in comparisons], axis=0)
        group_coverage = np.all([c["mmap_group_presence"] for c in comparisons])
        raw_zero = any(ch["raw_finite_zero_pixels"] for c in comparisons for ch in c["channels"])
        partial = any(0 < ch["raw_missing_pixels"] < 65536 for c in comparisons for ch in c["channels"])
        expected = [ch not in plan["unavailable_train_val_channels"] for ch in CHANNELS]
        every_split_channels = all(np.all(np.asarray(c["mmap_channel_presence"])[expected]) for c in comparisons)
        coverage = bool(every_split_channels and group_coverage and raw_zero and partial)
        result["coverage"] = {"pass": coverage, "raw_zero_observed": raw_zero,
                              "partial_missing_observed": partial, "all_groups_each_split": bool(group_coverage),
                              "all_available_channels_each_split": every_split_channels,
                              "observed_channel_presence": active.tolist(),
                              "unavailable_metadata_channels": plan["unavailable_train_val_channels"]}
        result["status"] = ("VALIDATED_WITHIN_SELECTED_SAMPLE_COVERAGE" if coverage and all(
            c["numerical_and_masks_pass"] for c in comparisons) else "FULL_CROP_COMPARISON_OR_COVERAGE_FAILED")
    except Exception as error:
        # Avoid signed URLs, proxy values and host paths in public evidence.
        result["error_type"] = type(error).__name__
    finally:
        stream.close()
        opener.close()
        received = hbudget.received+sum(b.received for _, b in reservations)
        result["transfer"] = {"prior_conservative_body_bytes": args.prior_transfer_bytes,
                              "hdf5_body_bytes": hbudget.received,
                              "hdf5_attempts": stream.attempts,
                              "worker_body_bytes": [b.received for _, b in reservations],
                              "new_returned_body_bytes": received,
                              "new_conservative_body_bytes": plan["transfer"]["new_body_upper_bound"],
                              "cumulative_conservative_body_bytes": args.prior_transfer_bytes+plan["transfer"]["new_body_upper_bound"],
                              "accounting_note": "whole reservation charged conservatively even on failure; transport may consume bytes before raising without returning them to ByteBudget; do not reuse unreturned allocation without audit",
                              "elapsed_seconds": time.monotonic()-started}
        result["cache_files"] = sum(1 for p in out.rglob("*") if p.is_file())+1
        (out/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "transfer": result["transfer"]}, indent=2))
    return 0 if result["status"] == "VALIDATED_WITHIN_SELECTED_SAMPLE_COVERAGE" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--listing", type=Path, default=ROOT/"reports/evidence/p30_20261103/moonstone_listing.json")
    parser.add_argument("--hf-stats", type=Path, default=ROOT/"reports/evidence/p30_20261103/hf_channel_stats.json")
    parser.add_argument("--class-map", type=Path, default=ROOT/"reports/evidence/p30_20261103/geologic_units.json")
    parser.add_argument("--plan", type=Path, default=ROOT/"protocols/p31_loader_pair_plan.json")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--prior-transfer-bytes", type=int, default=PRIOR_CONSERVATIVE)
    args = parser.parse_args(argv)
    if args.apply and args.write_plan:
        parser.error("lock offline before a separate apply invocation")
    if not PRIOR_CONSERVATIVE <= args.prior_transfer_bytes <= CAP:
        parser.error("cumulative conservative count cannot be reset")
    plan = make_plan(args)
    if args.write_plan:
        with args.plan.open("x") as output:
            output.write(json.dumps(plan, indent=2)+"\n")
    else:
        locked = read_json(args.plan)
        check_lock(locked)
        if canonical_hash(locked) != canonical_hash(plan):
            raise ValueError("inputs or implementation differ from locked plan; review offline")
        plan = locked
    if args.prior_transfer_bytes+plan["transfer"]["new_body_upper_bound"] > CAP:
        raise ValueError("updated cumulative budget insufficient")
    print(json.dumps({"mode": "apply" if args.apply else "dry_run", "lock_sha256": plan["lock_sha256"],
                      "records": plan["records"], "transfer": plan["transfer"],
                      "execution": plan["execution"]}, indent=2), flush=True)
    if not args.apply:
        return 0
    if args.outdir is None:
        parser.error("--apply requires a fresh --outdir")
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
