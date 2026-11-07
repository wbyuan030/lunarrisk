#!/usr/bin/env python3
"""Diagnose every fixed Train/Val raw terrain row, without models or filtering.

Numerical finiteness is not a provenance assertion. Candidate sentinel matches
are diagnostics, never an inferred NoData mask. Test pixels are inaccessible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunarrisk.benchmarks.moonstone import CHANNELS
from lunarrisk.benchmarks.prefix import PrefixClassifierDataset, _signature
from lunarrisk.nasa_transfer.cohort import (
    TRAIN_PLAN_PATH, METADATA_ARCHIVE_PATH, TRAIN_PLAN_SHA256,
    _read_pinned, load_authoritative_cohorts,
)
from lunarrisk.nasa_transfer.contracts import file_sha256

CANDIDATES = (-9999, -32768, 32767, 65535)
SOURCES = ("scripts/p33_audit_terrain_cohort.py", "src/lunarrisk/benchmarks/prefix.py",
           "src/lunarrisk/benchmarks/moonstone.py", "src/lunarrisk/nasa_transfer/cohort.py",
           "src/lunarrisk/nasa_transfer/contracts.py")


def summarize(values):
    """Count observed zeros/nonfinite separately; never replace pixel values."""
    values = np.asarray(values)
    if values.shape != (2, 256, 256) or values.dtype != np.float32:
        raise ValueError("raw terrain schema must be float32 (2,256,256)")
    result = {}
    for name, band in zip(("elevation", "slope"), values):
        finite = np.isfinite(band)
        observed = band[finite]
        result[name] = {
            "finite": int(finite.sum()), "nan": int(np.isnan(band).sum()),
            "posinf": int(np.isposinf(band).sum()), "neginf": int(np.isneginf(band).sum()),
            "zero": int((finite & (band == 0)).sum()),
            "min": float(observed.min()) if observed.size else None,
            "max": float(observed.max()) if observed.size else None,
            **{f"candidate_{value}": int((band == value).sum()) for value in CANDIDATES},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "metadata", "range-report", "outdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=ROOT / "reports/evidence/p30_20261103/hf_channel_stats.json")
    parser.add_argument("--class-map", type=Path, default=ROOT / "reports/evidence/p30_20261103/geologic_units.json")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_hashes = {name: file_sha256(ROOT / name) for name in SOURCES}
    plan_path = ROOT / TRAIN_PLAN_PATH
    cohorts, authority = load_authoritative_cohorts(plan_path, ROOT / METADATA_ARCHIVE_PATH)
    plan = json.loads(_read_pinned(plan_path, TRAIN_PLAN_SHA256, "Train plan"))
    paths = {"sha256": args.data, "metadata_sha256": args.metadata,
             "range_report_sha256": args.range_report, "stats_sha256": args.stats,
             "class_map_sha256": args.class_map}
    signatures, assets = {}, {}
    for key, path in paths.items():
        signatures[key] = _signature(path.stat())
        assets[key] = file_sha256(path)
        if assets[key] != plan["data"][key] or _signature(path.stat()) != signatures[key]:
            raise ValueError(f"historical asset mismatch/mutation: {key}")
    if args.data.stat().st_size != plan["data"]["bytes"]:
        raise ValueError("prefix length mismatch")
    print("historical assets authenticated", flush=True)
    stats, mapping = json.loads(args.stats.read_text()), json.loads(args.class_map.read_text())
    summaries = {}
    with (args.outdir / "rows.csv").open("x", newline="") as output:
        writer = None
        for split, code in (("train", 0), ("val", 1)):
            dataset = PrefixClassifierDataset(args.data, stats, mapping, args.metadata,
                                              args.range_report, plan, task="geology", split=code)
            try:
                ids = tuple(str(dataset.ids[index]) for index in dataset.indices)
                if ids != tuple(cohorts[split]) or tuple(map(int, dataset.targets)) != tuple(cohorts[split].values()):
                    raise ValueError("full authoritative IDs/targets mismatch")
                aggregate = {"rows": 0, "all_finite_rows": 0, "nonfinite_rows": [],
                             "candidate_match_rows": [], "channels": {},
                             "ordered_ids_sha256": dataset.ordered_ids_hash}
                digest = hashlib.sha256()
                patches = dataset._file()["patches"]
                fillvalue = float(patches.fillvalue)
                aggregate["hdf5_fillvalue"] = fillvalue if np.isfinite(fillvalue) else str(fillvalue)
                for ordinal, index in enumerate(dataset.indices):
                    dataset._file()  # retain prefix mutation guard at every read
                    # One contiguous two-channel selection; no normalized reader.
                    start = CHANNELS.index("elevation")
                    if CHANNELS[start:start + 2] != ("elevation", "slope"):
                        raise ValueError("terrain channels must be adjacent")
                    raw = np.asarray(patches[int(index), start:start + 2, :, :])
                    values = summarize(raw)
                    digest.update(ids[ordinal].encode() + b"\0" + raw.tobytes(order="C"))
                    row = {"split": split, "ordinal": ordinal, "source_index": int(index),
                           "sample_id": ids[ordinal], "target": int(dataset.targets[ordinal])}
                    for name, measures in values.items():
                        row.update({name + "_" + key: value for key, value in measures.items()})
                        total = aggregate["channels"].setdefault(name, {key: None if key in ("min", "max") else 0 for key in measures})
                        for key, value in measures.items():
                            if key in ("min", "max"):
                                if value is not None:
                                    total[key] = value if total[key] is None else (min if key == "min" else max)(total[key], value)
                            else:
                                total[key] += value
                    good = all(measure["finite"] == 65536 for measure in values.values())
                    aggregate["rows"] += 1
                    aggregate["all_finite_rows"] += int(good)
                    if not good:
                        aggregate["nonfinite_rows"].append(ids[ordinal])
                    if any(measure[f"candidate_{value}"] for measure in values.values() for value in CANDIDATES):
                        aggregate["candidate_match_rows"].append(ids[ordinal])
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    if (ordinal + 1) % 500 == 0:
                        print(f"{split}: {ordinal + 1}/{len(dataset)}", flush=True)
                aggregate["raw_terrain_ordered_sha256"] = digest.hexdigest()
                summaries[split] = aggregate
            finally:
                dataset.close()
    for key, path in paths.items():
        if _signature(path.stat()) != signatures[key]:
            raise ValueError(f"asset mutated during scan: {key}")
    if source_hashes != {name: file_sha256(ROOT / name) for name in SOURCES}:
        raise ValueError("source changed during scan")
    report = {"status": "DIAGNOSTIC_SCAN_COMPLETE", "model_execution": "NOT_RUN",
              "training": "NOT_RUN", "scores": "NOT_RUN", "test_pixels_read": 0, "Test": "HOLD",
              "authority": authority, "asset_sha256": assets, "source_sha256": source_hashes,
              "splits": summaries, "rows_csv_sha256": file_sha256(args.outdir / "rows.csv"),
              "finite_sentinel_absence": "NOT_ESTABLISHED",
              "original_raster_execution_lineage": "NOT_INDEPENDENTLY_REPRODUCED",
              "candidate_values": list(CANDIDATES),
              "candidate_interpretation": "Exploratory equality counts, not declared NoData or a mask; zero remains observed",
              "cohort_action": "UNCHANGED_NO_FILTERING_OR_IMPUTATION",
              "elapsed_seconds": time.monotonic() - started,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "python": sys.version.split()[0], "numpy": np.__version__,
              "torch_imported": "torch" in sys.modules}
    with (args.outdir / "summary.json").open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"status": report["status"], "splits": {key: value["rows"] for key, value in summaries.items()}}))


if __name__ == "__main__":
    main()
