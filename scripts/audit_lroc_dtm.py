#!/usr/bin/env python3
"""P18-05b: screen LROC NAC DTMs as provisional external holdout candidates.

The official LROC ``NAC_DTMS_360.ZIP`` shapefile lists every produced NAC DTM with its
stereo source images, resolution and footprint. This script answers one question: how
many DTMs are built from NAC products that SomBench never used, i.e. how large is the
pool of possible source-independent candidates. Source completeness and actual
footprint separation must be verified before a strict OOD claim.

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
PRODUCT_RE = re.compile(r"(M\d{9,10})(?!\d)", re.IGNORECASE)
MENTION_RE = re.compile(r"M\d+", re.IGNORECASE)


def normalise_products(images: str) -> list[str]:
    """Canonical NAC EDR IDs (left/right suffixes are not part of the ID)."""
    return sorted({p.upper() for p in PRODUCT_RE.findall(str(images))})


def source_id_metadata_valid(images: str) -> bool:
    """Require nonempty, well-formed DBF numeric IDs; README still required."""
    mentions = MENTION_RE.findall(str(images or ""))
    return bool(mentions) and all(len(m) in (10, 11) for m in mentions)


def unit_sphere(lat_deg, lon_deg) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    return np.column_stack([
        LUNAR_RADIUS_KM * np.cos(lat) * np.cos(lon),
        LUNAR_RADIUS_KM * np.cos(lat) * np.sin(lon),
        LUNAR_RADIUS_KM * np.sin(lat),
    ])


def vertex_outer_radius_km(center_lat: float, center_lon: float, vertices) -> float | None:
    """Max spherical chord to vertices; NOT a guarantee of polygon clearance."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        return None
    if not np.isfinite(vertices).all() or not np.isfinite([center_lat, center_lon]).all():
        return None
    if abs(center_lat) > 90 or np.any(np.abs(vertices[:, 1]) > 90):
        return None
    center = unit_sphere([center_lat], [center_lon])[0]
    points = unit_sphere(vertices[:, 1], vertices[:, 0])
    return float(np.linalg.norm(points - center, axis=1).max())


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
    sombench = {p for raw in manifest["NAC_EDR_ID"].dropna()
                for p in normalise_products(raw)}

    reader = shapefile.Reader(str(args.shapefile))
    fields = [field[0] for field in reader.fields[1:]]
    records = []
    for index in range(len(reader)):
        row = reader.record(index).as_dict()
        raw_sources = row.get("images", "")
        source_valid = source_id_metadata_valid(raw_sources)
        products = normalise_products(raw_sources) if source_valid else []
        vertex_radius = vertex_outer_radius_km(
            float(row.get("center_lat") or 0.0),
            float(row.get("center_lon") or 0.0),
            reader.shape(index).points,
        )
        records.append({
            "source_ids_nonempty_and_wellformed": source_valid,
            "vertex_outer_radius_km": vertex_radius,
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
    frame["all_products_unseen"] = (
        frame["source_ids_nonempty_and_wellformed"]
        & frame["products"].apply(lambda ids: len(ids) > 0)
        & (frame["products_unseen"].apply(len) == frame["products"].apply(len))
    )
    frame["any_product_unseen"] = frame["products_unseen"].apply(len) > 0

    # Stage-one screening using actual listed vertices. This does not yet
    # establish true DTM/used-tile footprint clearance or validate shape CRS.
    spatial = {"checked": False, "footprint_verified": False}
    if args.labels.exists():
        used = pd.read_csv(args.labels, usecols=["center_lat", "center_lon"])
        used = used.replace([np.inf, -np.inf], np.nan).dropna()
        if len(used):
            tree = cKDTree(unit_sphere(used["center_lat"], used["center_lon"]))
            candidates = frame["all_products_unseen"] & frame["vertex_outer_radius_km"].notna()
            centers = frame.loc[candidates, ["center_lat", "center_lon"]].to_numpy(dtype=np.float64)
            distances = np.array([], dtype=np.float64)
            passed = 0
            if len(centers):
                distances, _ = tree.query(unit_sphere(centers[:, 0], centers[:, 1]), k=1)
                radii = frame.loc[candidates, "vertex_outer_radius_km"].to_numpy(dtype=np.float64)
                clear = distances >= (radii + args.buffer_km)
                frame.loc[candidates, "center_to_used_chord_km"] = distances
                frame.loc[candidates, "vertex_screen_clear"] = clear
                passed = int(clear.sum())
            spatial = {
                "checked": True,
                "footprint_verified": False,
                "buffer_km": args.buffer_km,
                "used_tiles": int(len(used)),
                "candidates_with_vertices": int(candidates.sum()),
                "provisional_vertex_screen_clear": passed,
                "min_center_chord_km": float(distances.min()) if len(distances) else None,
                "rule": "center-to-used chord >= maximum listed vertex chord radius + buffer",
                "caveat": "Metadata screen only: verify shape CRS, true polygon-to-polygon "
                          "distances (including used tile footprints), and PDS source README.",
            }

    report = {
        "shapefile": {"path": str(args.shapefile), "fields": fields, "dtms": int(len(frame))},
        "sombench_products": len(sombench),
        "dtm_coverage": {
            "all_products_unseen": int(frame["all_products_unseen"].sum()),
            "empty_or_malformed_source_records": int(
                (~frame["source_ids_nonempty_and_wellformed"]).sum()),
            "missing_or_invalid_vertex_geometry": int(
                frame["vertex_outer_radius_km"].isna().sum()),
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
        # Strict footprint-verified count remains UNKNOWN until polygon QA.
        "strict_candidates_after_spatial": None,
        "provisional_candidates_after_vertex_screen": (
            spatial.get("provisional_vertex_screen_clear") if spatial["checked"] else None),
        "verdict": ("CANDIDATES_REQUIRE_POLYGON_AND_SOURCE_QA"
                    if spatial.get("provisional_vertex_screen_clear", 0) > 0
                    else "BLOCKED_OR_UNVERIFIED"),
        "notes": [
            "DBF image IDs are preliminary; verify the complete stereo source list with PDS README.",
            "Vertex screen replaces the unsafe equal-area-circle radius, but is not polygon clearance.",
            "Native DTM resolution varies; define labels and resampling before final testing.",
            "No GO_STRICT_OOD until source provenance, polygon distances and CRS QA pass.",
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
