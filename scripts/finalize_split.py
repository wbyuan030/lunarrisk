#!/usr/bin/env python3
"""P1-02: finalize the manifest split (train/cal/val/test) with a spatial buffer.

Input is ``outputs/manifest_100k.parquet`` from ``scripts/build_manifest.py``.

Rules
-----
* Calibration is carved out of the cleaned train split at *group* level using a
  deterministic hash, so no leakage group crosses train/cal.
* A 10 km buffer is measured between tile centres on the lunar sphere.
  Train/cal tiles within 10 km of val/test, and val tiles within 10 km of test,
  are marked ``drop_buffer``. Tiles in test are never dropped.
* No payload files are downloaded and nothing is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SPLITS = ("train", "cal", "val", "test")
LUNAR_RADIUS_KM = 1737.4


def cal_assignment(groups: pd.Series, seed: int = 42, mod: int = 10) -> set[str]:
    """Deterministically assign ~1/mod of groups to calibration."""
    chosen = set()
    for group in sorted(groups.astype(str).unique()):
        digest = hashlib.sha256(f"{group}|{seed}".encode("utf-8")).digest()
        if int.from_bytes(digest[:4], "big") % mod == 0:
            chosen.add(group)
    return chosen


def unit_sphere(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg.astype(float))
    lon = np.radians(lon_deg.astype(float))
    return np.column_stack([
        LUNAR_RADIUS_KM * np.cos(lat) * np.cos(lon),
        LUNAR_RADIUS_KM * np.cos(lat) * np.sin(lon),
        LUNAR_RADIUS_KM * np.sin(lat),
    ])


def nearest_other_split(frame: pd.DataFrame, split_column: str, points: np.ndarray) -> pd.DataFrame:
    """For each row, distance to the nearest row in a different split."""
    labels = frame[split_column].astype(str).to_numpy()
    distances = np.full(len(frame), np.inf)
    nearest = np.full(len(frame), "", dtype=object)
    for split in SPLITS:
        own = labels == split
        if not own.any():
            continue
        other_mask = ~own
        if not other_mask.any():
            continue
        tree = cKDTree(points[other_mask])
        dist, indices = tree.query(points[own], k=1)
        distances[own] = dist
        other_labels = labels[other_mask]
        nearest[own] = other_labels[indices]
    return pd.DataFrame({
        "nearest_other_split": nearest,
        "nearest_other_split_km": distances,
    }, index=frame.index)


PROTECTED_BY = {
    "val": ("test",),
    "cal": ("val", "test"),
    "train": ("cal", "val", "test"),
}


def find_buffer_drops(frame: pd.DataFrame, points: np.ndarray, buffer_km: float = 10.0,
                      split_column: str = "split_pre_buffer") -> np.ndarray:
    """Drop lower-priority tiles that sit within ``buffer_km`` of a *surviving* higher split.

    Priority order is test > val > cal > train. Each step queries the protected sets that
    are still present, so dropping cannot cascade into a violation: a tile removed earlier
    neither protects nor threatens later steps. Test tiles are never dropped.

    Note this deliberately does not use a single "nearest other split" label: if a val tile
    happens to be closest to a train tile, that must not excuse it from the val-vs-test
    distance requirement (the shadowing bug).
    """
    labels = frame[split_column].astype(str).to_numpy()
    drop = np.zeros(len(frame), dtype=bool)
    for own_split, protected in PROTECTED_BY.items():
        own_mask = (labels == own_split) & ~drop
        ref_mask = np.isin(labels, list(protected)) & ~drop
        if not own_mask.any() or not ref_mask.any():
            continue
        distances, _ = cKDTree(points[ref_mask]).query(points[own_mask], k=1)
        drop[np.where(own_mask)[0][distances < buffer_km]] = True
    return drop


def apply_buffer(frame: pd.DataFrame, points: np.ndarray, buffer_km: float = 10.0,
                 split_column: str = "split_pre_buffer") -> pd.Series:
    """Return split_final using the priority-ordered buffer drop."""
    final = frame[split_column].astype(str).copy()
    drop = find_buffer_drops(frame, points, buffer_km, split_column)
    final[drop] = "drop_buffer"
    return final


def surviving_pair_distances(final: pd.Series, points: np.ndarray) -> dict:
    """Minimum centre distance between every pair of surviving splits (buffer contract check)."""
    result = {}
    labels = final.astype(str).to_numpy()
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            left_mask = labels == left
            right_mask = labels == right
            if not left_mask.any() or not right_mask.any():
                continue
            distances, _ = cKDTree(points[right_mask]).query(points[left_mask], k=1)
            result[f"{left}_{right}"] = {
                "min_km": round(float(distances.min()), 4),
                "tiles_left": int(left_mask.sum()),
                "tiles_right": int(right_mask.sum()),
            }
    return result


def split_audit(frame: pd.DataFrame, split_column: str) -> dict:
    result = {}
    keys = {}
    for split in SPLITS:
        part = frame[frame[split_column] == split]
        groups = set(part["group_id"].astype(str))
        products = set(part["NAC_EDR_ID"].astype(str))
        blocks = set(part["block_5km"].astype(str))
        keys[split] = (groups, products, blocks)
        result[split] = {
            "tiles": int(len(part)),
            "usable_tiles": int(part["has_slope3m"].sum()),
            "groups": len(groups),
            "products": len(products),
            "blocks_5km": len(blocks),
            "zones": int(part["LTM_CODE"].nunique()),
        }
    overlap = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            lg, lp, lb = keys[left]
            rg, rp, rb = keys[right]
            overlap[f"{left}_{right}"] = {
                "groups": len(lg & rg),
                "products": len(lp & rp),
                "blocks_5km": len(lb & rb),
            }
    return {"splits": result, "overlap": overlap}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest_100k.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/manifest_100k_final.parquet"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/split_summary.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-km", type=float, default=10.0)
    args = parser.parse_args(argv)

    frame = pd.read_parquet(args.manifest)
    if "center_lat" not in frame.columns or "center_lon" not in frame.columns:
        raise SystemExit("manifest is missing center_lat/center_lon; rebuild with build_manifest.py")

    clean = frame["split_group_clean"].astype(str)
    cal_groups = cal_assignment(frame.loc[clean == "train", "group_id"], args.seed)
    frame = frame.copy()
    frame["split_pre_buffer"] = clean
    frame.loc[frame["group_id"].astype(str).isin(cal_groups), "split_pre_buffer"] = "cal"
    # Only groups that were in the cleaned train split can become cal.
    frame.loc[clean != "train", "split_pre_buffer"] = clean[clean != "train"]

    points = unit_sphere(frame["center_lat"].to_numpy(), frame["center_lon"].to_numpy())
    nearest = nearest_other_split(frame, "split_pre_buffer", points)
    frame["nearest_other_split"] = nearest["nearest_other_split"]
    frame["nearest_other_split_km"] = nearest["nearest_other_split_km"].round(4)
    frame["split_final"] = apply_buffer(frame, points, args.buffer_km)
    frame["drop_buffer"] = frame["split_final"] == "drop_buffer"

    audits = split_audit(frame, "split_final")
    kept = frame[frame["split_final"] != "drop_buffer"]
    distances = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            left_mask = frame["split_final"] == left
            right_mask = frame["split_final"] == right
            if not left_mask.any() or not right_mask.any():
                continue
            dist, _ = cKDTree(points[right_mask]).query(points[left_mask], k=1)
            distances[f"{left}_{right}"] = {
                "min_km": round(float(dist.min()), 3),
                "p1_km": round(float(np.percentile(dist, 1)), 3),
                "median_km": round(float(np.median(dist)), 3),
                "tiles_left": int(left_mask.sum()),
                "tiles_right": int(right_mask.sum()),
                "within_buffer": int((dist < args.buffer_km).sum()),
            }
    contract = surviving_pair_distances(frame["split_final"], points)
    violations = {pair: entry for pair, entry in contract.items() if entry["min_km"] < args.buffer_km}

    summary = {
        "manifest": str(args.manifest),
        "seed": args.seed,
        "buffer_km": args.buffer_km,
        "rows": int(len(frame)),
        "cal_method": "sha256(group_id|seed) first 4 bytes % 10 == 0, train groups only",
        "cal_groups": len(cal_groups),
        "split_final_counts": {s: int((frame["split_final"] == s).sum()) for s in (*SPLITS, "drop_buffer")},
        "split_final_usable_counts": {
            s: int(((frame["split_final"] == s) & frame["has_slope3m"]).sum()) for s in (*SPLITS, "drop_buffer")
        },
        "buffer_dropped": int((frame["split_final"] == "drop_buffer").sum()),
        "buffer_contract": {
            "buffer_km": args.buffer_km,
            "surviving_pairs": contract,
            "violations": violations,
            "satisfied": not violations,
        },
        "split_audit": audits,
        "cross_split_min_distance_km": distances,
        "kept_rows": int(len(kept)),
        "caveats": [
            "geo join still pending; this is a split/manifest artifact only",
            "cal is carved from cleaned train groups, not from val/test",
            "buffer is centre-to-centre on a sphere, not polygon-to-polygon",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "rows", "cal_groups", "split_final_counts", "split_final_usable_counts",
        "buffer_dropped")}, indent=2))
    print("final overlap:", audits["overlap"])
    print("distances:", json.dumps(distances, indent=1))
    print(f"wrote {args.out} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
