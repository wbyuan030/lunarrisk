#!/usr/bin/env python3
"""P18-05: freeze an untouched `heldout_v2` cohort before any model sees it.

Selection is restricted to official ``test``-split tiles that were never hydrated,
whose NAC product was never used (and, in strict mode, whose leakage group was
never used), and that sit at least
``--buffer-km`` from every hydrated tile. The freeze writes the ID list and its
SHA256; no payload is downloaded and no model metric is read for these tiles.

Refuses to freeze when too few independent tiles survive, reporting
``BLOCKED_NEW_HELDOUT`` instead of substituting easier samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
LUNAR_RADIUS_KM = 1737.4
MIN_FROZEN_TILES = 30
REQUIRED_COLUMNS = [
    "sample_id", "NAC_EDR_ID", "group_id", "LTM_CODE", "split_final",
    "center_lat", "center_lon", "bounds_xmin", "bounds_ymin", "footprint_wkt",
    "image_path", "dtm3m_path", "slope3m_path",
]


class FreezeError(RuntimeError):
    pass


def unit_sphere(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    return np.column_stack([
        LUNAR_RADIUS_KM * np.cos(lat) * np.cos(lon),
        LUNAR_RADIUS_KM * np.cos(lat) * np.sin(lon),
        LUNAR_RADIUS_KM * np.sin(lat),
    ])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=False).stdout.strip() or "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def group_coverage(manifest: pd.DataFrame, used: pd.DataFrame) -> dict:
    """How much of the corpus's leakage groups the hydrated cohort already spans."""
    all_groups = set(manifest["group_id"].astype(str))
    used_groups = set(used["group_id"].astype(str))
    all_products = set(manifest["NAC_EDR_ID"].astype(str))
    used_products = set(used["NAC_EDR_ID"].astype(str))
    return {
        "manifest_groups": len(all_groups),
        "used_groups": len(used_groups),
        "unused_groups": len(all_groups - used_groups),
        "manifest_products": len(all_products),
        "used_products": len(used_products),
        "unused_products": len(all_products - used_products),
    }


def select_heldout(manifest: pd.DataFrame, used: pd.DataFrame, n: int, buffer_km: float,
                   seed: int, exclusion: str = "group",
                   max_per_product: int = 5) -> tuple[pd.DataFrame, dict]:
    """Pick unseen tiles with unused products/groups, far from every hydrated tile.

    ``exclusion='group'`` is the strict protocol (needs unused leakage groups);
    ``exclusion='product'`` relaxes the group rule and is only a product/spatial holdout.
    """
    if used.empty:
        raise FreezeError("no hydrated tiles found; cannot define what 'untouched' means")
    if exclusion not in ("group", "product"):
        raise FreezeError(f"unknown exclusion rule {exclusion!r}")
    used_products = set(used["NAC_EDR_ID"].astype(str))
    used_groups = set(used["group_id"].astype(str))
    used_points = unit_sphere(used["center_lat"].to_numpy(), used["center_lon"].to_numpy())
    tree = cKDTree(used_points)

    candidates = manifest[manifest["split_final"] == "test"].copy()
    candidates = candidates[candidates["has_slope3m"].astype(bool)]
    candidates = candidates[~candidates["NAC_EDR_ID"].astype(str).isin(used_products)]
    if exclusion == "group":
        candidates = candidates[~candidates["group_id"].astype(str).isin(used_groups)]
    if candidates.empty:
        raise FreezeError("no candidate tiles survive the product/group exclusion")
    points = unit_sphere(candidates["center_lat"].to_numpy(), candidates["center_lon"].to_numpy())
    distances, _ = tree.query(points, k=1)
    candidates = candidates.assign(buffer_km_to_used=distances)
    candidates = candidates[candidates["buffer_km_to_used"] >= buffer_km]
    if candidates.empty:
        raise FreezeError(f"no candidate tile is at least {buffer_km} km from the hydrated set")

    rng = np.random.default_rng(seed)
    candidates = candidates.iloc[rng.permutation(len(candidates))].reset_index(drop=True)
    # One tile per NAC product first, round-robin over LTM zones for spatial diversity.
    by_zone = {}
    for zone, part in candidates.groupby("LTM_CODE", sort=True):
        by_zone[str(zone)] = part.reset_index(drop=True)
    zones = sorted(by_zone)
    zones = [zones[i] for i in rng.permutation(len(zones))]
    cursor = {zone: 0 for zone in zones}
    per_product: dict[str, int] = {}
    picked = []
    picked_points = []  # keep the V2 tiles mutually spaced as well
    while len(picked) < n and any(cursor[z] < len(by_zone[z]) for z in zones):
        progressed = False
        for zone in zones:
            if len(picked) >= n:
                break
            part = by_zone[zone]
            accepted = False
            while cursor[zone] < len(part):
                row = part.iloc[cursor[zone]]
                cursor[zone] += 1
                product = str(row["NAC_EDR_ID"])
                if per_product.get(product, 0) >= max_per_product:
                    continue
                point = unit_sphere(np.array([row["center_lat"]]), np.array([row["center_lon"]]))
                if picked_points:
                    distances, _ = cKDTree(np.vstack(picked_points)).query(point, k=1)
                    if float(distances[0]) < buffer_km:
                        continue
                per_product[product] = per_product.get(product, 0) + 1
                picked.append(row)
                picked_points.append(point[0])
                progressed = True
                accepted = True
                break
            if accepted and len(picked) >= n:
                break
        if not progressed:
            break
    selected = pd.DataFrame(picked).reset_index(drop=True)
    audit = {
        "exclusion": exclusion,
        "max_per_product": max_per_product,
        "candidates_after_exclusion": int(len(candidates)),
        "groups_after_selection": int(selected["group_id"].nunique()) if len(selected) else 0,
        "selected_groups_already_used": sorted(set(selected["group_id"].astype(str)) & used_groups),
        "buffer_km": buffer_km,
        "min_buffer_km_selected": float(selected["buffer_km_to_used"].min()) if len(selected) else None,
        "used_products": len(used_products),
        "used_groups": len(used_groups),
        "used_tiles": int(len(used)),
    }
    return selected, audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest_100k_final.parquet"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels_hydrated.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("protocols"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--buffer-km", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-tiles", type=int, default=MIN_FROZEN_TILES,
                        help="refuse to freeze fewer independent tiles than this")
    parser.add_argument("--exclusion", choices=("group", "product"), default="group",
                        help="strict group-level independence, or a product/spatial holdout")
    parser.add_argument("--out-prefix", default="heldout_v2")
    parser.add_argument("--max-per-product", type=int, default=5,
                        help="tiles allowed per NAC product in the frozen cohort")
    args = parser.parse_args(argv)

    manifest = pd.read_parquet(args.manifest)
    missing = [c for c in REQUIRED_COLUMNS if c not in manifest.columns]
    if missing:
        raise SystemExit(f"manifest is missing columns {missing}")
    used = pd.read_csv(args.labels) if args.labels.exists() else pd.DataFrame()
    if used.empty:
        raise SystemExit(f"no hydrated labels at {args.labels}; refusing to guess the used set")
    missing_used = [c for c in ("sample_id", "NAC_EDR_ID", "group_id", "center_lat", "center_lon")
                    if c not in used.columns]
    if missing_used:
        raise SystemExit(f"labels file is missing {missing_used}")

    coverage = group_coverage(manifest, used)
    try:
        selected, audit = select_heldout(manifest, used, args.n, args.buffer_km, args.seed,
                                        args.exclusion, args.max_per_product)
    except FreezeError as exc:
        selected, audit = pd.DataFrame(), {"exclusion": args.exclusion, "error": str(exc)}
    if len(selected) < args.min_tiles:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        blocked = {
            "status": "BLOCKED_NEW_HELDOUT",
            "reason": audit.get("error", f"only {len(selected)} candidate tiles available"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "exclusion": args.exclusion,
            "coverage": coverage,
            "audit": audit,
            "sources": {
                "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
                "used_labels": {"path": str(args.labels), "sha256": sha256_file(args.labels)},
            },
            "next_options": [
                "Use a corpus with NAC products outside this 100k release (the hydrated cohort "
                "already spans every leakage group here).",
                "Explicitly accept a product/spatial holdout with --exclusion product and label "
                "the transfer limitation in every claim.",
            ],
        }
        path = args.out_dir / f"{args.out_prefix}_protocol.json"
        path.write_text(json.dumps(blocked, indent=2), encoding="utf-8")
        print(json.dumps({"status": "BLOCKED_NEW_HELDOUT", "selected": int(len(selected)),
                          "minimum": args.min_tiles,
            "max_per_product": args.max_per_product, "coverage": coverage, "audit": audit}, indent=2))
        print(f"wrote {path}")
        return 2

    # Independent double-checks before writing anything.
    used_ids = set(used["sample_id"].astype(str))
    if used_ids & set(selected["sample_id"].astype(str)):
        raise SystemExit("refusing to freeze: V2 overlaps already-used tiles")
    if set(used["NAC_EDR_ID"].astype(str)) & set(selected["NAC_EDR_ID"].astype(str)):
        raise SystemExit("refusing to freeze: V2 reuses a used NAC product")
    if args.exclusion == "group" and set(used["group_id"].astype(str)) & set(selected["group_id"].astype(str)):
        raise SystemExit("refusing to freeze: V2 reuses a used leakage group")

    columns = REQUIRED_COLUMNS + ["buffer_km_to_used"]
    frozen = selected[columns].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / f"{args.out_prefix}_manifest.csv"
    frozen.to_csv(manifest_path, index=False)
    frozen_hash = sha256_file(manifest_path)
    group_disjoint = not audit.get("selected_groups_already_used")
    protocol = {
        "status": "FROZEN" if group_disjoint else "FROZEN_PRODUCT_LEVEL_NOT_GROUP_DISJOINT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "algorithm": {
            "candidate_pool": "split_final == test, has_slope3m",
            "exclusions": (["hydrated sample_id", "used NAC_EDR_ID", "used group_id"]
                           if args.exclusion == "group" else
                           ["hydrated sample_id", "used NAC_EDR_ID"]),
            "exclusion": args.exclusion,
            "group_disjoint": group_disjoint,
            "spatial_rule": f"center distance >= {args.buffer_km} km from every hydrated tile",
            "sampling": "round-robin over LTM zones, seeded shuffle, with max_per_product cap",
            "seed": args.seed,
            "requested": args.n,
            "minimum": args.min_tiles,
            "max_per_product": args.max_per_product,
        },
        "counts": {
            "frozen_tiles": int(len(frozen)),
            "products": int(frozen["NAC_EDR_ID"].nunique()),
            "groups": int(frozen["group_id"].nunique()),
            "zones": int(frozen["LTM_CODE"].nunique()),
            "by_zone": {k: int(v) for k, v in frozen["LTM_CODE"].value_counts().items()},
        },
        "coverage": coverage,
        "audit": audit,
        "sources": {
            "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
            "used_labels": {"path": str(args.labels), "sha256": sha256_file(args.labels)},
            "heldout_manifest_sha256": frozen_hash,
        },
        "overlap_checks": {
            "hydrated_sample_ids": 0,
            "used_nac_products": 0,
            "future_train_expansion": "V2 rows are split_final=test; train expansion only draws "
                                      "from split_final=train, so they cannot be pulled in",
            "group_overlap": (0 if group_disjoint else int(len(audit.get("selected_groups_already_used", [])))),
        },
        "known_limits": [
            "DTM_source_ID is UNKNOWN in the source index: this is a tile/geographic/group-level "
            "holdout, not a proven stereo-DTM-source-level OOD split.",
            ("Group-level independence is NOT achieved: every leakage group in this corpus was "
             "already touched by the hydrated cohort, so this is a product/spatial holdout only."
             if not group_disjoint else
             "Group-level independence verified (no used group appears in V2)."),
            "Footprint-level projection review is still pending (GEO_JOIN_PENDING).",
            "No payload was downloaded and no model metric was computed for these tiles before freezing.",
        ],
    }
    protocol_path = args.out_dir / f"{args.out_prefix}_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "tiles": int(len(frozen)),
                      "products": protocol["counts"]["products"],
                      "groups": protocol["counts"]["groups"],
                      "zones": protocol["counts"]["zones"],
                      "min_buffer_km": audit["min_buffer_km_selected"],
                      "manifest_sha256": frozen_hash}, indent=2))
    print(f"wrote {manifest_path} and {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
