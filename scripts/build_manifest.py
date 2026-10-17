#!/usr/bin/env python3
"""P1-01: build a leakage-audited manifest for the S3 SomBench 100k release.

Grouping policy
---------------
* Nodes are NAC EDR products (PRODUCT_ID).
* Two products are merged when they have tiles in the same 5 km projected block
  inside the same LTM zone. The resulting connected component is the leakage
  group: no product and no 5 km block can cross a group boundary.
* The official split is preserved except that any group containing tiles from
  more than one official split is assigned wholly to its majority split.

This script does not download payload files and does not train anything.
SHA256/read checks are filled only for the 12 tiles previously downloaded and
verified by ``scripts/s3_sample.py``; other rows keep those fields null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

S3_PREFIX = "pretraining_dataset/100k_release/NAC_HighRes/"
SPLIT_ORDER = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_blocks(frame: pd.DataFrame, block_m: int) -> pd.Series:
    x = np.floor(frame["BOUNDS_XMIN"].to_numpy() / block_m).astype(np.int64)
    y = np.floor(frame["BOUNDS_YMIN"].to_numpy() / block_m).astype(np.int64)
    return pd.Series(
        [f"{zone}|{bx}|{by}" for zone, bx, by in zip(frame["LTM_CODE"].astype(str), x, y)],
        index=frame.index,
    )


def build_groups(frame: pd.DataFrame, block_m: int = 5000) -> tuple[pd.Series, pd.Series]:
    """Return (block key, leakage group id) for every row."""
    blocks = assign_blocks(frame, block_m)
    products = frame["PRODUCT_ID"].astype(str).to_numpy()
    product_index = {product: i for i, product in enumerate(pd.unique(products))}
    parent = list(range(len(product_index)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_product_in_block: dict[str, int] = {}
    for block, product in zip(blocks.to_numpy(), products):
        node = product_index[product]
        if block in first_product_in_block:
            union(first_product_in_block[block], node)
        else:
            first_product_in_block[block] = node

    root_to_products: dict[int, list[str]] = {}
    for product, node in product_index.items():
        root_to_products.setdefault(find(node), []).append(product)
    root_to_name = {
        root: "G_" + min(products_in_group)
        for root, products_in_group in root_to_products.items()
    }
    group_ids = pd.Series(
        [root_to_name[find(product_index[product])] for product in products],
        index=frame.index,
    )
    return blocks, group_ids


def clean_split(frame: pd.DataFrame, group_column: str = "group_id") -> tuple[pd.Series, list[str]]:
    """Assign each group wholly to its majority official split (ties -> train)."""
    majority: dict[str, str] = {}
    reassigned: list[str] = []
    for group, part in frame.groupby(group_column, sort=False):
        counts = part["DATASET"].value_counts()
        best = max(SPLIT_ORDER, key=lambda split: (int(counts.get(split, 0)), -SPLIT_ORDER.index(split)))
        majority[str(group)] = best
        if counts.index.size > 1:
            reassigned.append(str(group))
    cleaned = frame[group_column].astype(str).map(majority)
    return cleaned, reassigned


def footprint_wkt(row: pd.Series) -> str:
    xmin, xmax = float(row["BOUNDS_XMIN"]), float(row["BOUNDS_XMAX"])
    ymin, ymax = float(row["BOUNDS_YMIN"]), float(row["BOUNDS_YMAX"])
    return (
        f"POLYGON (({xmin:.3f} {ymin:.3f}, {xmax:.3f} {ymin:.3f}, "
        f"{xmax:.3f} {ymax:.3f}, {xmin:.3f} {ymax:.3f}, {xmin:.3f} {ymin:.3f}))"
    )


def load_sample_verification(path: Path) -> tuple[dict[tuple[str, int, int], dict], str | None]:
    if not path.exists():
        return {}, None
    report = json.loads(path.read_text(encoding="utf-8"))
    verification = {}
    for row in report.get("rows", []):
        key = (str(row["product_id"]), int(row["row"]), int(row["col"]))
        assets = {}
        for column, info in row["files"].items():
            relative = info["path"].split("data/s3/sample/", 1)[-1]
            assets[column] = {"relative": relative, "sha256": info["sha256"], "bytes": info["bytes"]}
        verification[key] = {
            "assets": assets,
            "slope_finite_fraction": row["slope_finite_fraction"],
            "bounds_aligned": row["bounds_aligned"],
            "units_ok": row["units_empirically_consistent_with_degrees"],
        }
    crs = None
    sample_dir = path.parent.parent / "data" / "s3" / "sample"
    candidates = sorted(sample_dir.glob("nac/*.nc"))
    if candidates:
        try:
            import h5netcdf  # noqa: F401
            import hdf5plugin  # noqa: F401
            import xarray as xr
            with xr.open_dataset(candidates[0], engine="h5netcdf") as dataset:
                crs = dataset.attrs.get("crs")
        except Exception:
            crs = None
    return verification, crs


def group_sweep(frame: pd.DataFrame) -> list[dict]:
    sweep = []
    for block_m in (1000, 5000, 10000, 25000):
        blocks, groups = build_groups(frame, block_m)
        sizes = groups.value_counts()
        mixed = groups.groupby(groups).apply(lambda part: frame.loc[part.index, "DATASET"].nunique())
        sweep.append({
            "block_m": block_m,
            "n_blocks": int(blocks.nunique()),
            "n_groups": int(groups.nunique()),
            "largest_group_tiles": int(sizes.iloc[0]),
            "largest_group_tile_fraction": round(float(sizes.iloc[0] / len(frame)), 4),
            "median_group_tiles": float(sizes.median()),
            "mixed_groups": int((mixed > 1).sum()),
            "tiles_in_mixed_groups": int(sizes[mixed > 1].sum()),
        })
    return sweep


def split_audit(frame: pd.DataFrame, split_column: str) -> dict:
    result = {"split_column": split_column, "splits": {}}
    product_column = "NAC_EDR_ID" if "NAC_EDR_ID" in frame.columns else "PRODUCT_ID"
    keys = {}
    for split in SPLIT_ORDER:
        part = frame[frame[split_column] == split]
        groups = set(part["group_id"].astype(str))
        products = set(part[product_column].astype(str))
        blocks = set(part["block_5km"].astype(str))
        keys[split] = (groups, products, blocks)
        result["splits"][split] = {
            "tiles": int(len(part)),
            "groups": len(groups),
            "products": len(products),
            "blocks_5km": len(blocks),
        }
    result["overlap"] = {}
    for i, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[i + 1:]:
            lg, lp, lb = keys[left]
            rg, rp, rb = keys[right]
            result["overlap"][f"{left}_{right}"] = {
                "groups": len(lg & rg),
                "products": len(lp & rp),
                "blocks_5km": len(lb & rb),
            }
    return result


def build_manifest(
    frame: pd.DataFrame,
    verification: dict[tuple[str, int, int], dict],
    crs_wkt: str | None,
    index_sha256: str,
    block_m: int = 5000,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["block_5km"], frame["group_id"] = build_groups(frame, block_m)
    frame["split_group_clean"], reassigned_groups = clean_split(frame)
    frame["group_reassigned"] = frame["group_id"].astype(str).isin(reassigned_groups)
    verification_by_key = verification

    rows = []
    for _, row in frame.iterrows():
        key = (str(row["PRODUCT_ID"]), int(row["ROW"]), int(row["COL"]))
        checked = verification_by_key.get(key)
        assets = checked["assets"] if checked else {}
        sample_id = f"{row['PRODUCT_ID']}_r{int(row['ROW'])}_c{int(row['COL'])}"
        rows.append({
            "sample_id": sample_id,
            "dataset_version": "S3_100k_release",
            "s3_index_sha256": index_sha256,
            "repo_revision": None,
            "NAC_EDR_ID": str(row["PRODUCT_ID"]),
            "DTM_source_ID": None,
            "LTM_CODE": str(row["LTM_CODE"]),
            "ROW": int(row["ROW"]),
            "COL": int(row["COL"]),
            "image_path": S3_PREFIX + str(row["NAC_TILE"]),
            "dtm3m_path": S3_PREFIX + str(row["DTM_3M_TILE"]) if pd.notna(row["DTM_3M_TILE"]) else None,
            "slope3m_path": S3_PREFIX + str(row["SLOPE_3M_TILE"]) if pd.notna(row["SLOPE_3M_TILE"]) else None,
            "coarse_dtm_path": S3_PREFIX + str(row["DTM_60M_TILE"]) if pd.notna(row["DTM_60M_TILE"]) else None,
            "coarse_slope_path": S3_PREFIX + str(row["SLOPE_60M_TILE"]) if pd.notna(row["SLOPE_60M_TILE"]) else None,
            "metadata_path": S3_PREFIX + str(row["METADATA_TILE"]) if pd.notna(row["METADATA_TILE"]) else None,
            "bounds_xmin": float(row["BOUNDS_XMIN"]),
            "bounds_xmax": float(row["BOUNDS_XMAX"]),
            "bounds_ymin": float(row["BOUNDS_YMIN"]),
            "bounds_ymax": float(row["BOUNDS_YMAX"]),
            "center_lat": float(row["CENTER_LATITUDE"]),
            "center_lon": float(row["CENTER_LONGITUDE"]),
            "footprint_wkt": footprint_wkt(row),
            "crs_wkt": crs_wkt,
            "gsd_nac_m": 1.0,
            "gsd_3m_m": 2.9941520467836256,
            "split_official": str(row["DATASET"]),
            "split_group_clean": str(row["split_group_clean"]),
            "group_reassigned": bool(row["group_reassigned"]),
            "group_id": str(row["group_id"]),
            "block_5km": str(row["block_5km"]),
            "has_dtm3m": bool(pd.notna(row["DTM_3M_TILE"])),
            "has_slope3m": bool(pd.notna(row["SLOPE_3M_TILE"])),
            "dtm3m_fraction_null_index": None if pd.isna(row["DTM_3M_FRACTION_NULL"]) else float(row["DTM_3M_FRACTION_NULL"]),
            "slope3m_fraction_null_index": None if pd.isna(row["SLOPE_3M_FRACTION_NULL"]) else float(row["SLOPE_3M_FRACTION_NULL"]),
            "sha256": assets.get("NAC_TILE", {}).get("sha256"),
            "nac_sha256": assets.get("NAC_TILE", {}).get("sha256"),
            "dtm3m_sha256": assets.get("DTM_3M_TILE", {}).get("sha256"),
            "slope3m_sha256": assets.get("SLOPE_3M_TILE", {}).get("sha256"),
            "metadata_sha256": assets.get("METADATA_TILE", {}).get("sha256"),
            "modality_read_ok": bool(checked) if checked else None,
            "valid_fraction": checked["slope_finite_fraction"] if checked else None,
            "co_registration_status": "SAMPLED_VERIFIED" if checked else "INDEX_BOUNDS_ONLY",
            "geo_join_status": "GEO_JOIN_PENDING",
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/s3/NAC_HighRes_100k.parquet"))
    parser.add_argument("--sample-audit", type=Path, default=Path("outputs/s3_sample_audit.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/manifest_100k.parquet"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/manifest_summary.json"))
    parser.add_argument("--sample-csv", type=Path, default=Path("outputs/manifest_sample.csv"))
    parser.add_argument("--block-m", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    raw = pd.read_parquet(args.index)
    index_sha256 = sha256_file(args.index)
    verification, crs_wkt = load_sample_verification(args.sample_audit)
    manifest = build_manifest(raw, verification, crs_wkt, index_sha256, args.block_m)
    manifest = manifest.sort_values("sample_id").reset_index(drop=True)

    official = split_audit(manifest, "split_official")
    cleaned = split_audit(manifest, "split_group_clean")
    usable = manifest[manifest["has_slope3m"]]
    summary = {
        "manifest_rows": int(len(manifest)),
        "usable_rows_with_slope3m": int(len(usable)),
        "n_products": int(manifest["NAC_EDR_ID"].nunique()),
        "n_ltm_zones": int(manifest["LTM_CODE"].nunique()),
        "n_groups": int(manifest["group_id"].nunique()),
        "block_m": args.block_m,
        "block_5km_count": int(manifest["block_5km"].nunique()),
        "s3_index_sha256": index_sha256,
        "crs_wkt": crs_wkt,
        "verification_rows": len(verification),
        "verification_with_sha256": int(manifest["nac_sha256"].notna().sum()),
        "verification_units_ok": sum(bool(v["units_ok"]) for v in verification.values()),
        "official_split": official,
        "clean_split": cleaned,
        "reassigned": {
            "groups": int(manifest.loc[manifest["group_reassigned"], "group_id"].nunique()),
            "tiles_in_reassigned_groups": int(manifest["group_reassigned"].sum()),
            "tiles_split_changed": int((manifest["split_official"] != manifest["split_group_clean"]).sum()),
            "changes": {
                f"{old}->{new}": int(((manifest["split_official"] == old) & (manifest["split_group_clean"] == new)).sum())
                for old in SPLIT_ORDER for new in SPLIT_ORDER if old != new
            },
        },
        "group_size_sweep": group_sweep(raw),
        "split_counts": {
            split: {
                "official": int((manifest["split_official"] == split).sum()),
                "clean": int((manifest["split_group_clean"] == split).sum()),
            } for split in SPLIT_ORDER
        },
        "usable_split_counts": {
            split: int((usable["split_group_clean"] == split).sum()) for split in SPLIT_ORDER
        },
        "caveats": [
            "geo join is still pending: index bounds and sampled grid checks only",
            "per-tile DTM source ID is not provided; keep null",
            "sha256/read checks exist only for the 12 sampled tiles",
            "slope units are documented/empirically verified, not stored in NetCDF attrs",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.out, index=False)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sample_columns = [
        "sample_id", "NAC_EDR_ID", "LTM_CODE", "split_official", "split_group_clean",
        "group_id", "block_5km", "has_slope3m", "valid_fraction", "geo_join_status",
    ]
    manifest[sample_columns].head(2000).to_csv(args.sample_csv, index=False)

    print(json.dumps({k: summary[k] for k in (
        "manifest_rows", "usable_rows_with_slope3m", "n_products", "n_ltm_zones",
        "n_groups", "block_5km_count", "verification_rows")}, indent=2))
    print("official overlap:", official["overlap"])
    print("clean overlap:", cleaned["overlap"])
    print("reassigned:", summary["reassigned"])
    print("clean split counts:", summary["usable_split_counts"])
    print(f"wrote {args.out}, {args.summary}, {args.sample_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
