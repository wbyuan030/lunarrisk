#!/usr/bin/env python3
"""Audit actual Moonstone IDs, stored/geographic splits and targets without reading Test pixels."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from lunarrisk.benchmarks.moonstone import age_target, geology_targets, select_split


def audit(metadata, mapping):
    if metadata.get("status") != "REAL_HDF5_METADATA_READ_NOT_LOADER_EQUIVALENCE":
        raise ValueError("verified real metadata required")
    table = metadata["metadata"]
    names = table["metadata/geology/dominant_unit_name"]
    units = table["metadata/geology/dominant_unit_id"]
    targets = geology_targets(units, names, mapping)
    ages = np.asarray([age_target(name) for name in names])
    ids = np.asarray(table["metadata/patch_id"])
    row, col = (np.asarray(table[f"metadata/{key}_idx"]) for key in ("row", "col"))
    lat, lon = (np.asarray(table[f"metadata/center_{key}"]) for key in ("lat", "lon"))
    stored = np.asarray(table["metadata/split"])
    arrays = [ids, row, col, lat, lon, stored, ages, targets]
    if any(len(array) != len(ids) for array in arrays) or len(np.unique(ids)) != len(ids):
        raise ValueError("inconsistent metadata lengths or duplicate patch IDs")
    if len(set(zip(row.tolist(), col.tolist()))) != len(ids):
        raise ValueError("duplicate spatial tiles")
    if not ((row >= 0) & (row < 90) & (col >= 0) & (col < 180)).all():
        raise ValueError("tile index outside official grid")
    if not (np.allclose(lat, 90-(row+0.5)*2, rtol=0, atol=1e-5) and
            np.allclose(lon, -180+(col+0.5)*2, rtol=0, atol=1e-5)):
        raise ValueError("centers disagree with the official 128ppd / 256px grid")
    summary = {"rows": len(ids), "unique_ids": len(np.unique(ids)),
               "geology_mapping_validated": True, "age_unknown_total": int((ages < 0).sum()),
               "tracks": {}}
    geographic = np.full(len(ids), -1)
    for track in ("official_stored", "official_geographic"):
        summary["tracks"][track] = {}
        for split in range(3):
            selected = select_split(stored, lat, split, track)
            if track == "official_geographic":
                geographic[selected] = split
            valid_age = selected & (ages >= 0)
            summary["tracks"][track][str(split)] = {
                "rows": int(selected.sum()), "valid_age_rows": int(valid_age.sum()),
                "age_unknown_excluded": int((selected & (ages < 0)).sum()),
                "age_histogram": np.bincount(ages[valid_age], minlength=5).tolist(),
                "geology_histogram": np.bincount(targets[selected], minlength=49).tolist(),
                "ordered_id_sha256": hashlib.sha256("\n".join(map(str, ids[selected])).encode()).hexdigest()}
    summary["geographic_excluded_rows"] = int((geographic < 0).sum())
    records = []
    for i in range(len(ids)):
        records.append({"sample_id": str(ids[i]), "row_idx": int(row[i]), "col_idx": int(col[i]),
                        "center_lat": float(lat[i]), "center_lon": float(lon[i]),
                        "lat_min": float(lat[i]-1), "lat_max": float(lat[i]+1),
                        "lon_min": float(lon[i]-1), "lon_max": float(lon[i]+1),
                        "stored_split": int(stored[i]), "geographic_split": int(geographic[i]),
                        "unit_id": int(units[i]), "unit_name": names[i], "age_target": int(ages[i])})
    return summary, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary, records = audit(json.loads(args.metadata.read_text()), json.loads(args.class_map.read_text()))
    summary["metadata_sha256"] = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    summary["class_map_sha256"] = hashlib.sha256(args.class_map.read_bytes()).hexdigest()
    args.outdir.mkdir(parents=True, exist_ok=False)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    (args.outdir/"rows.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    (args.outdir/"summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
