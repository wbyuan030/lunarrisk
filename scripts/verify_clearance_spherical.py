#!/usr/bin/env python3
"""Independent re-computation of candidate clearance with pure spherical geometry.

The Gate 1 audit computes polygon-to-polygon distances in a local azimuthal-equidistant
projection. This script recomputes the same quantity with no projection at all: every
vertex of the candidate polygon is measured against the great-circle arcs of each used tile
boundary, and every tile corner against the arcs of the candidate polygon. Agreement
between the two independent implementations is what makes the ">= 10 km" clearance claim
trustworthy.

Note: this measures boundary-to-boundary distance. When one polygon contains the other the
audit reports 0 (intersecting geometries) while this script reports the boundary gap; the
differing rows are always containment cases.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile

ROOT = Path(__file__).resolve().parents[1]
LUNAR_RADIUS_KM = 1737.4


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_lroc_footprints",
                                                  ROOT / "scripts" / "audit_lroc_footprints.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unit(lat, lon):
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def angle(first, second) -> float:
    return float(np.arccos(np.clip(float(np.dot(first, second)), -1.0, 1.0)))


def point_to_arc(point, start, end) -> float:
    """Angular distance from ``point`` to the great-circle arc ``start``-``end``."""
    d_start, d_end, span = angle(point, start), angle(point, end), angle(start, end)
    normal = np.cross(start, end)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return min(d_start, d_end)
    normal = normal / norm
    foot = point - float(np.dot(point, normal)) * normal
    foot_norm = float(np.linalg.norm(foot))
    if foot_norm < 1e-12:
        return min(d_start, d_end)
    foot = foot / foot_norm
    if abs(angle(start, foot) + angle(foot, end) - span) >= 1e-9:
        return min(d_start, d_end)  # the foot falls outside the arc
    return angle(point, foot)


def clearance_km(geometry, tiles) -> float:
    rings = ([np.asarray(geometry.exterior.coords)] if geometry.geom_type == "Polygon"
             else [np.asarray(part.exterior.coords) for part in geometry.geoms])
    best = math.inf
    for tile in tiles:
        lons = np.asarray([tile[0], tile[2], tile[2], tile[0]])
        lats = np.asarray([tile[1], tile[1], tile[3], tile[3]])
        corners = unit(lats, lons)
        edges = [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
        for ring in rings:
            vertices = unit(ring[:, 1], ring[:, 0])
            for vertex in vertices:
                for start, end in edges:
                    best = min(best, point_to_arc(vertex, start, end))
            for corner in corners:
                for k in range(len(vertices) - 1):
                    best = min(best, point_to_arc(corner, vertices[k], vertices[k + 1]))
    return best * LUNAR_RADIUS_KM


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest_100k_final_v2.parquet"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels_hydrated.csv"))
    parser.add_argument("--protected", type=Path,
                        default=Path("protocols/heldout_v2_product_level_manifest.csv"))
    parser.add_argument("--shapefile", type=Path, default=Path("data/lroc_nac_dtm/shape/NAC_DTMS_360.SHP"))
    parser.add_argument("--audit", type=Path, default=Path("outputs/p19_lroc_footprint_audit.csv"))
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--tol-m", type=float, default=5.0)
    args = parser.parse_args(argv)

    audit_module = load_audit_module()
    manifest = pd.read_parquet(args.manifest)
    used_ids = set(pd.read_csv(args.labels, usecols=["sample_id"])["sample_id"].astype(str))
    used_ids |= set(pd.read_csv(args.protected, usecols=["sample_id"])["sample_id"].astype(str))
    used = manifest[manifest["sample_id"].astype(str).isin(used_ids)]
    tiles = [(float(audit_module.normalise_lon([row.upper_left_lon])[0]), float(row.upper_left_lat),
              float(audit_module.normalise_lon([row.lower_right_lon])[0]), float(row.lower_right_lat))
             for row in used.itertuples()]
    tile_centers = unit(np.array([0.5 * (t[1] + t[3]) for t in tiles]),
                        np.array([0.5 * (t[0] + t[2]) for t in tiles]))

    reader = shapefile.Reader(str(args.shapefile))
    index = {record.as_dict()["DTM_NAME"]: i for i, record in enumerate(reader.iterRecords())}
    audit = pd.read_csv(args.audit)
    passing = audit[audit["source_disjoint"] & audit["footprint_disjoint"]]
    sample = passing.nsmallest(args.scenes, "min_polygon_distance_km")

    print(f"scenes checked: {len(sample)} (tightest passing candidates)")
    print(f"{'DTM':16s} {'aeqd_km':>9s} {'spherical_km':>13s} {'delta_m':>9s}")
    worst = 0.0
    for row in sample.itertuples():
        shape = reader.shape(index[row.DTM_name])
        geometry, status = audit_module.dtm_polygon(shape.points, list(shape.parts))
        if status != "OK":
            print(f"{row.DTM_name:16s} geometry {status}")
            continue
        center = unit(row.center_lat, row.center_lon)
        near = [tile for tile, tile_center in zip(tiles, tile_centers)
                if angle(tile_center, center) < math.radians(3.0)]
        distance = clearance_km(geometry, near)
        delta = (distance - row.min_polygon_distance_km) * 1000.0
        worst = max(worst, abs(delta))
        print(f"{row.DTM_name:16s} {row.min_polygon_distance_km:9.3f} {distance:13.3f} {delta:9.1f}")
    print(f"worst |delta| = {worst:.1f} m (tolerance {args.tol_m} m)")
    return 0 if worst <= args.tol_m else 1


if __name__ == "__main__":
    raise SystemExit(main())
