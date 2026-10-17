#!/usr/bin/env python3
"""P18-05b: audit whether a strict group-independent held-out set is obtainable.

Compares the hydrated cohort against a larger release index (default: the official
1M-release NAC index) using the same 5 km-block union-find grouping as
``build_manifest.py``:

* tiles present in both indexes,
* leakage groups touched by the 760 hydrated tiles,
* untouched groups and how many tiles/products/this-zones they contain,
* how many candidates survive a centre-distance buffer from every used tile.

Writes ``outputs/p18_source_audit.json`` and prints a markdown summary. No payload is
downloaded and no model metric is touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_manifest import assign_blocks, build_groups  # noqa: E402

LUNAR_RADIUS_KM = 1737.4
INDEX_COLUMNS = ["PRODUCT_ID", "ROW", "COL", "LTM_CODE", "BOUNDS_XMIN", "BOUNDS_YMIN",
                 "CENTER_LATITUDE", "CENTER_LONGITUDE", "DATASET", "SLOPE_3M_TILE"]


def unit_sphere(lat_deg, lon_deg) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    return np.column_stack([
        LUNAR_RADIUS_KM * np.cos(lat) * np.cos(lon),
        LUNAR_RADIUS_KM * np.cos(lat) * np.sin(lon),
        LUNAR_RADIUS_KM * np.sin(lat),
    ])


def sample_id(frame: pd.DataFrame) -> pd.Series:
    return (frame["PRODUCT_ID"].astype(str) + "_r" + frame["ROW"].astype(str)
            + "_c" + frame["COL"].astype(str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest_100k_final.parquet"))
    parser.add_argument("--index", type=Path, default=Path("data/s3_1m/NAC_HighRes_1M.parquet"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels_hydrated.csv"))
    parser.add_argument("--buffer-km", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/p18_source_audit.json"))
    args = parser.parse_args(argv)

    big = pd.read_parquet(args.index, columns=INDEX_COLUMNS)
    big["sample_id"] = sample_id(big)
    big = big[big["SLOPE_3M_TILE"].notna()].reset_index(drop=True)
    small = pd.read_parquet(args.manifest, columns=["sample_id", "NAC_EDR_ID", "group_id", "LTM_CODE"])
    used = pd.read_csv(args.labels, usecols=["sample_id", "NAC_EDR_ID", "group_id",
                                             "center_lat", "center_lon"])

    small_ids = set(small["sample_id"].astype(str))
    big_ids = set(big["sample_id"].astype(str))
    used_ids = set(used["sample_id"].astype(str))

    blocks, groups = build_groups(big, block_m=5000)
    big["block_5km"] = blocks
    big["group_id"] = groups
    group_sizes = big["group_id"].value_counts()

    used_in_big = big["sample_id"].isin(used_ids)
    used_groups = set(big.loc[used_in_big, "group_id"].astype(str))
    all_groups = set(big["group_id"].astype(str))
    untouched_groups = all_groups - used_groups
    untouched = big[big["group_id"].astype(str).isin(untouched_groups)]

    used_points = unit_sphere(used["center_lat"], used["center_lon"])
    tree = cKDTree(used_points)
    distances = np.full(len(untouched), np.inf)
    if len(untouched):
        points = unit_sphere(untouched["CENTER_LATITUDE"], untouched["CENTER_LONGITUDE"])
        distances, _ = tree.query(points, k=1)

    report = {
        "inputs": {
            "manifest": {"path": str(args.manifest), "tiles": int(len(small))},
            "index": {"path": str(args.index), "tiles": int(len(big)),
                      "products": int(big["PRODUCT_ID"].nunique()),
                      "zones": int(big["LTM_CODE"].nunique())},
            "labels": {"path": str(args.labels), "tiles": int(len(used))},
        },
        "coverage": {
            "manifest_tiles_present_in_index": int(len(small_ids & big_ids)),
            "manifest_tiles_missing_from_index": int(len(small_ids - big_ids)),
            "used_tiles_present_in_index": int(len(used_ids & big_ids)),
            "used_tiles_missing_from_index": sorted(used_ids - big_ids)[:10],
            "index_products_not_in_manifest": len(
                set(big["PRODUCT_ID"].astype(str)) - set(small["NAC_EDR_ID"].astype(str))),
        },
        "groups": {
            "index_groups": len(all_groups),
            "used_groups": len(used_groups),
            "untouched_groups": len(untouched_groups),
            "untouched_tiles": int(len(untouched)),
            "untouched_products": int(untouched["PRODUCT_ID"].nunique()) if len(untouched) else 0,
            "untouched_zones": int(untouched["LTM_CODE"].nunique()) if len(untouched) else 0,
            "untouched_group_sizes": {
                "median": float(group_sizes[list(untouched_groups)].median()) if untouched_groups else 0.0,
                "max": int(group_sizes[list(untouched_groups)].max()) if untouched_groups else 0,
            },
            "largest_untouched_groups": [
                {"group_id": str(g), "tiles": int(group_sizes[g])}
                for g in group_sizes[list(untouched_groups)].head(10).index
            ] if untouched_groups else [],
        },
        "spatial": {
            "buffer_km": args.buffer_km,
            "untouched_tiles_outside_buffer": int((distances >= args.buffer_km).sum()) if len(untouched) else 0,
            "min_buffer_km": float(np.min(distances)) if len(distances) else None,
        },
        "verdict": "GO_STRICT_OOD" if (
            len(untouched_groups) > 0
            and (distances >= args.buffer_km).sum() >= 30
        ) else "PRODUCT_ONLY",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
