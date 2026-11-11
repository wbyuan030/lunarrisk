#!/usr/bin/env python3
"""Extract per-patch FeO and TiO2 means from the published LPGRS aligned rasters.

Labels are computed only for the fixed Train/Val tiles that the frozen cohort
already selects; Test tiles are never sampled. Values are the mean of finite
pixels inside each 256x256 window, with the finite count recorded so a partially
covered window cannot silently become a precise label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GRID = (23040, 46080)
PATCH = 256


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="authenticated metadata export")
    parser.add_argument("--cohort-manifest", type=Path, required=True, help="NASA feature cache manifest")
    parser.add_argument("--feo", type=Path, required=True)
    parser.add_argument("--tio2", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    import rasterio
    args.outdir.mkdir(parents=True, exist_ok=False)
    export = json.loads(args.metadata.read_text())["metadata"]
    grid_map = {str(sample): (int(row), int(col)) for sample, row, col in
                zip(export["metadata/patch_id"], export["metadata/row_idx"], export["metadata/col_idx"])}
    manifest = json.loads(args.cohort_manifest.read_text())
    ids = {split: manifest["splits"][split]["ids"] for split in ("train", "val")}
    sources = {}
    for name, path in (("feo", args.feo), ("tio2", args.tio2)):
        handle = rasterio.open(path)
        if (handle.height, handle.width) != GRID or handle.count != 1:
            raise ValueError(f"{name} raster geometry differs from the 128 ppd grid")
        if str(handle.crs).upper() not in ("MOON_2000:30100", "MOON_2000", "IAU_2015:30100"):
            print(f"{name} CRS: {handle.crs}")
        sources[name] = handle
    started = time.monotonic()
    rows = []
    for split in ("train", "val"):
        for sample_id in ids[split]:
            row_idx, col_idx = grid_map[sample_id]
            y0, x0 = row_idx * PATCH, col_idx * PATCH
            record = {"split": split, "sample_id": sample_id, "row_idx": row_idx, "col_idx": col_idx}
            for name, handle in sources.items():
                window = rasterio.windows.Window(x0, y0, PATCH, PATCH)
                values = handle.read(1, window=window).astype(np.float64)
                finite = np.isfinite(values)
                record[name + "_finite"] = int(finite.sum())
                record[name + "_mean"] = float(values[finite].mean()) if finite.any() else None
                record[name + "_std"] = float(values[finite].std()) if finite.any() else None
            rows.append(record)
        print(f"{split}: {len(ids[split])} tiles done", flush=True)
    for handle in sources.values():
        handle.close()
    summary = {"raster_sha256": {"feo": sha256(args.feo), "tio2": sha256(args.tio2)},
               "raster_bytes": {"feo": args.feo.stat().st_size, "tio2": args.tio2.stat().st_size},
               "rows": len(rows), "elapsed_seconds": time.monotonic() - started}
    for name in ("feo", "tio2"):
        values = np.array([row[name + "_mean"] for row in rows if row[name + "_mean"] is not None])
        coverage = np.array([row[name + "_finite"] for row in rows])
        summary[name] = {"count": int(len(values)), "missing": int(len(rows) - len(values)),
                         "mean": float(values.mean()) if len(values) else None,
                         "std": float(values.std()) if len(values) else None,
                         "min": float(values.min()) if len(values) else None,
                         "max": float(values.max()) if len(values) else None,
                         "median_finite_pixels": float(np.median(coverage)),
                         "fully_finite_rows": int((coverage == PATCH * PATCH).sum())}
    with (args.outdir / "labels.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2)[:1200])


if __name__ == "__main__":
    main()
