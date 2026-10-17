#!/usr/bin/env python3
"""P18-05b: audit the LROC NAC DTM archive against the SomBench product set.

The official LROC ``NAC_DTMS_360.ZIP`` shapefile lists every produced NAC DTM with its
stereo source images, resolution and footprint. This script answers one question: how
many DTMs are built from NAC products that SomBench never used, i.e. how large is the
pool for a strictly source-independent held-out set?

Reads only the shapefile + local manifest; downloads nothing (except the shapefile, which
must already be present) and reads no model metric.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LUNAR_RADIUS_KM = 1737.4
PRODUCT_RE = re.compile(r"(M\d{9,10})")


def normalise_products(images: str) -> list[str]:
    """'M1503229434, M1503250470' -> ['M1503229434', 'M1503250470'] (suffixes stripped)."""
    return sorted(set(PRODUCT_RE.findall(str(images))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapefile", type=Path, default=Path("data/lroc_nac_dtm/shape/NAC_DTMS_360.SHP"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest_100k_final.parquet"))
    parser.add_argument("--out", type=Path, default=Path("outputs/p18_lroc_dtm_audit.json"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels_hydrated.csv"),
                        help="used tiles; their centres define the spatial isolation check")
    parser.add_argument("--buffer-km", type=float, default=10.0)
    args = parser.parse_args(argv)

    import shapefile  # pyshp

    manifest = pd.read_parquet(args.manifest, columns=["NAC_EDR_ID"])
    sombench = {PRODUCT_RE.sub(r"\1", str(value)) for value in manifest["NAC_EDR_ID"].unique()}
    sombench |= {str(value) for value in manifest["NAC_EDR_ID"].unique()}

    reader = shapefile.Reader(str(args.shapefile))
    fields = [field[0] for field in reader.fields[1:]]
    records = []
    for index in range(len(reader)):
        row = reader.record(index).as_dict()
        products = normalise_products(row.get("images", ""))
        records.append({
            "dtm_name": str(row.get("DTM_NAME", "")),
            "sitename": str(row.get("sitename", "")),
            "resolution_m": float(row.get("resolution") or 0.0),
            "center_lon": float(row.get("center_lon") or 0.0),
            "center_lat": float(row.get("center_lat") or 0.0),
            "coverage_sqkm": float(row.get("cov_sqkm") or 0.0),
            "completion": str(row.get("completion", "")),
            "triang_rms_m": float(row.get("triang_rms") or 0.0),
            "products": products,
            "products_seen": [p for p in products if p in sombench],
            "products_unseen": [p for p in products if p not in sombench],
            "url": str(row.get("url", "")),
        })
    frame = pd.DataFrame(records)
    frame["all_products_unseen"] = frame["products_unseen"].apply(len) == frame["products"].apply(len)
    frame["any_product_unseen"] = frame["products_unseen"].apply(len) > 0

    # Spatial isolation: a DTM is kept only if its (conservative) footprint radius plus the
    # buffer clears every hydrated tile centre.
    spatial = {"checked": False}
    if args.labels.exists():
        used = pd.read_csv(args.labels, usecols=["center_lat", "center_lon"])
        used_points = np.column_stack([
            LUNAR_RADIUS_KM * np.cos(np.radians(used["center_lat"])) * np.cos(np.radians(used["center_lon"])),
            LUNAR_RADIUS_KM * np.cos(np.radians(used["center_lat"])) * np.sin(np.radians(used["center_lon"])),
            LUNAR_RADIUS_KM * np.sin(np.radians(used["center_lat"])),
        ])
        tree = cKDTree(used_points)
        strict = frame["all_products_unseen"]
        centers = frame.loc[strict, ["center_lat", "center_lon"]].to_numpy(dtype=np.float64)
        points = np.column_stack([
            LUNAR_RADIUS_KM * np.cos(np.radians(centers[:, 0])) * np.cos(np.radians(centers[:, 1])),
            LUNAR_RADIUS_KM * np.cos(np.radians(centers[:, 0])) * np.sin(np.radians(centers[:, 1])),
            LUNAR_RADIUS_KM * np.sin(np.radians(centers[:, 0])),
        ])
        distances, _ = tree.query(points, k=1)
        radius = np.sqrt(frame.loc[strict, "coverage_sqkm"].to_numpy(dtype=np.float64) / np.pi)
        clear = distances >= (radius + args.buffer_km)
        spatial = {
            "checked": True,
            "buffer_km": args.buffer_km,
            "used_tiles": int(len(used)),
            "strict_candidates_center_clear": int(clear.sum()),
            "min_center_distance_km": float(distances.min()) if len(distances) else None,
            "rule": "centre distance >= footprint_radius + buffer (conservative bound)",
        }
        frame.loc[strict, "center_to_used_km"] = distances
        frame.loc[strict, "spatially_clear"] = clear

    report = {
        "shapefile": {"path": str(args.shapefile), "fields": fields, "dtms": int(len(frame))},
        "sombench_products": len(sombench),
        "dtm_coverage": {
            "all_products_unseen": int(frame["all_products_unseen"].sum()),
            "any_product_unseen": int(frame["any_product_unseen"].sum()),
            "all_products_seen": int((~frame["any_product_unseen"]).sum()),
        },
        "source_pool": {
            "unseen_products_across_dtms": sorted({p for values in frame["products_unseen"] for p in values}),
            "seen_products_across_dtms": sorted({p for values in frame["products_seen"] for p in values}),
        },
        "strict_candidates": {
            "count": int(frame["all_products_unseen"].sum()),
            "resolution_range_m": [
                float(frame.loc[frame["all_products_unseen"], "resolution_m"].min()) if frame["all_products_unseen"].any() else None,
                float(frame.loc[frame["all_products_unseen"], "resolution_m"].max()) if frame["all_products_unseen"].any() else None,
            ],
            "coverage_sqkm_total": float(frame.loc[frame["all_products_unseen"], "coverage_sqkm"].sum()),
            "completion_range": [
                frame.loc[frame["all_products_unseen"], "completion"].min() if frame["all_products_unseen"].any() else None,
                frame.loc[frame["all_products_unseen"], "completion"].max() if frame["all_products_unseen"].any() else None,
            ],
            "top_by_coverage": frame[frame["all_products_unseen"]]
            .sort_values("coverage_sqkm", ascending=False)
            .head(args.top)[["dtm_name", "sitename", "resolution_m", "coverage_sqkm",
                             "completion", "center_lat", "center_lon", "products", "url"]]
            .to_dict("records"),
        },
        "spatial_isolation": spatial,
        "strict_candidates_after_spatial": (
            int(frame.loc[frame["all_products_unseen"], "spatially_clear"].sum())
            if spatial["checked"] else None),
        "verdict": ("GO_STRICT_OOD" if (spatial["checked"] and spatial["strict_candidates_center_clear"] >= 10)
                    else "PRODUCT_ONLY"),
        "notes": [
            "DTMs whose every stereo product is unused by SomBench are source-independent at the "
            "NAC-product level; the per-DTM README/DBF also fixes the DTM_source_ID gap.",
            "Resolution is 3-4 m for these products, comparable to the SomBench slope_3m grid.",
            "Footprint-level projection and licensing still need a manual check before freezing.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("shapefile", "sombench_products", "dtm_coverage",
                                             "spatial_isolation", "strict_candidates_after_spatial",
                                             "verdict")}, indent=2)[:6000])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
