#!/usr/bin/env python3
"""P19-01: real footprint and source audit for LROC NAC DTM holdout candidates.

Unlike the metadata screen, this script:

* reads the DTM polygons from the official shapefile (``.prj`` = geographic Moon 2000,
  sphere R = 1737400 m) and validates each ring,
* rebuilds every *used* tile footprint from the geographic corner coordinates in the
  manifest (the per-zone LTM ``BOUNDS_*`` metres are not usable across zones),
* computes true polygon-to-polygon distances in a local azimuthal-equidistant projection
  centred on each candidate,
* fetches the PDS README for each candidate and audits the complete stereo source list
  (the DBF ``images`` field is only preliminary and is truncated for mosaics),
* writes per-candidate PASS/FAIL reasons and never downloads DEM/NAC payloads.

Only candidates whose **source list and footprint are both verified** may later enter the
frozen ``heldout_v2_external`` set.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from functools import lru_cache

from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import explain_validity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_lroc_dtm import (  # noqa: E402
    MENTION_RE,
    PRODUCT_RE,
    SOURCE_TOKEN_RE,
    classify_source_field,
    normalise_products,
)

LUNAR_RADIUS_M = 1737400.0
README_BASE = ("https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/"
               "lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/SDP/NAC_DTM/{name}/"
               "NAC_DTM_{name}_README.TXT")


def normalise_lon(values):
    """0..360 and -180..180 both land in [-180, 180)."""
    values = np.asarray(values, dtype=np.float64)
    return (values + 180.0) % 360.0 - 180.0


def unwrap_around(lons, center_lon):
    """Shift longitudes by ±360 so that they stay contiguous around ``center_lon``."""
    lons = np.asarray(lons, dtype=np.float64)
    return center_lon + (lons - center_lon + 180.0) % 360.0 - 180.0


def crs_check(prj_path: Path) -> dict:
    """Verify the shapefile CRS is lunar geographic degrees on the expected sphere."""
    text = prj_path.read_text(encoding="utf-8", errors="replace") if prj_path.exists() else ""
    result = {"has_prj": bool(text), "wkt": text[:300], "valid": False, "reason": ""}
    if not text:
        result["reason"] = "missing .prj"
        return result
    try:
        crs = CRS.from_wkt(text)
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"unparsable .prj: {type(exc).__name__}: {exc}"
        return result
    ellipsoid = crs.ellipsoid
    result["name"] = crs.name
    result["radius_m"] = float(ellipsoid.semi_major_metre)
    result["is_geographic"] = bool(crs.is_geographic)
    result["valid"] = (crs.is_geographic and abs(result["radius_m"] - LUNAR_RADIUS_M) < 1.0)
    if not result["valid"]:
        result["reason"] = (f"expected lunar geographic degrees with R={LUNAR_RADIUS_M}; "
                            f"got geographic={crs.is_geographic} R={result['radius_m']}")
    return result


def dtm_polygon(points: list, parts: list[int]) -> tuple[Polygon | MultiPolygon | None, str]:
    """Build a validated lon/lat polygon (multipart supported) from pyshp output."""
    if not points:
        return None, "EMPTY_GEOMETRY"
    parts = list(parts) + [len(points)]
    rings = []
    for start, end in zip(parts[:-1], parts[1:]):
        ring = [(float(x), float(y)) for x, y in points[start:end]]
        if len(ring) < 3:
            return None, "RING_TOO_SHORT"
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if len(set(ring)) < 3:
            return None, "DEGENERATE_RING"
        rings.append(ring)
    try:
        polygons = []
        for ring in rings:
            lon = normalise_lon([p[0] for p in ring])
            lat = np.asarray([p[1] for p in ring], dtype=np.float64)
            if np.any(~np.isfinite(lon)) or np.any(~np.isfinite(lat)) or np.any(np.abs(lat) > 90):
                return None, "INVALID_COORDINATE"
            radians = np.radians(lon)
            center_lon = math.degrees(math.atan2(float(np.sin(radians).mean()),
                                                 float(np.cos(radians).mean())))
            polygons.append(Polygon(zip(unwrap_around(lon, center_lon), lat)))
        # Inner rings (holes, island-in-lake parts) are filled by keeping only the outermost
        # shells: that can only *increase* the candidate area, so the clearance test stays
        # conservative. A self-intersecting shell is kept as INVALID and excluded downstream.
        outers = [ring for ring in polygons
                  if not any(other is not ring and other.contains(ring) for other in polygons)]
        geometry = outers[0] if len(outers) == 1 else MultiPolygon(outers)
        if geometry.is_empty or not geometry.is_valid:
            return None, f"INVALID_POLYGON:{explain_validity(geometry)[:60]}"
        return geometry, "OK"
    except Exception as exc:  # noqa: BLE001
        return None, f"GEOMETRY_ERROR:{type(exc).__name__}"


def tile_footprints(manifest: pd.DataFrame, sample_ids: set[str]) -> list[tuple[str, Polygon | MultiPolygon]]:
    """Geographic bbox footprints from the UL/LR corners (antimeridian split)."""
    subset = manifest[manifest["sample_id"].astype(str).isin(sample_ids)]
    footprints = []
    for row in subset.itertuples():
        lon1, lat1 = float(row.upper_left_lon), float(row.upper_left_lat)
        lon2, lat2 = float(row.lower_right_lon), float(row.lower_right_lat)
        lon1, lon2 = float(normalise_lon([lon1])[0]), float(normalise_lon([lon2])[0])
        lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
        if abs(lon1 - lon2) > 180.0:  # crosses the antimeridian: split into two boxes
            west, east = max(lon1, lon2), min(lon1, lon2)
            geometry = unary_union([box(west, lat_min, 180.0, lat_max),
                                    box(-180.0, lat_min, east, lat_max)])
        else:
            geometry = box(min(lon1, lon2), lat_min, max(lon1, lon2), lat_max)
        footprints.append((str(row.sample_id), geometry))
    return footprints


@lru_cache(maxsize=512)
def local_transformer(center_lat: float, center_lon: float):
    """Azimuthal-equidistant projection on the lunar sphere, cached per centre."""
    crs_lonlat = CRS.from_proj4(f"+proj=longlat +R={LUNAR_RADIUS_M} +no_defs")
    crs_local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.6f} +lon_0={center_lon:.6f} "
        f"+R={LUNAR_RADIUS_M} +units=m +no_defs")
    return Transformer.from_crs(crs_lonlat, crs_local, always_xy=True)


def project_geometry(geometry, center_lat: float, center_lon: float):
    """Project a lon/lat geometry into the local plane of the given centre."""
    project = local_transformer(center_lat, center_lon)

    def ring_coords(coords):
        lon = unwrap_around([point[0] for point in coords], center_lon)
        lat = [point[1] for point in coords]
        x, y = project.transform(lon, lat)
        return list(zip(x, y))

    if isinstance(geometry, Polygon):
        return Polygon(ring_coords(geometry.exterior.coords))
    return MultiPolygon([Polygon(ring_coords(part.exterior.coords)) for part in geometry.geoms])


def distance_km(polygon, footprints, center_lat: float, center_lon: float) -> tuple[float, str]:
    """Minimum polygon-to-polygon distance in metres via a local aeqd projection."""
    local = project_geometry(polygon, center_lat, center_lon)
    best_km, best_id = float("inf"), ""
    for sample_id, geometry in footprints:
        distance = local.distance(project_geometry(geometry, center_lat, center_lon)) / 1000.0
        if distance < best_km:
            best_km, best_id = distance, sample_id
    return best_km, best_id


def pair_distance_km(first, second, first_center, second_center) -> float:
    """Exact polygon-to-polygon distance for two nearby candidates (aeqd at the midpoint)."""
    mid_lat = (first_center[0] + second_center[0]) / 2.0
    mid_lon = math.degrees(math.atan2(
        math.sin(math.radians(first_center[1])) + math.sin(math.radians(second_center[1])),
        math.cos(math.radians(first_center[1])) + math.cos(math.radians(second_center[1]))))
    return project_geometry(first, mid_lat, mid_lon).distance(
        project_geometry(second, mid_lat, mid_lon)) / 1000.0


def candidate_structure(entries: list[tuple[str, object, tuple[float, float]]],
                        max_pair_km: float = 400.0,
                        disjoint_km: float = 10.0) -> dict:
    """Pairwise polygon clearance, nearest neighbour and <disjoint_km> groups among candidates."""
    names = [name for name, _, _ in entries]
    nearest = {name: (float("inf"), "") for name in names}
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(first, second):
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for i, (name_a, geometry_a, center_a) in enumerate(entries):
        for name_b, geometry_b, center_b in entries[i + 1:]:
            if abs(center_a[0] - center_b[0]) > max_pair_km / 30.0:
                continue
            distance = pair_distance_km(geometry_a, geometry_b, center_a, center_b)
            if distance > max_pair_km:
                continue
            for name, other in ((name_a, name_b), (name_b, name_a)):
                if distance < nearest[name][0]:
                    nearest[name] = (distance, other)
            if distance < disjoint_km:
                union(name_a, name_b)

    roots = {name: find(name) for name in names}
    ordered = {root: f"CDTM_{index + 1:04d}"
               for index, root in enumerate(sorted(set(roots.values())))}
    groups = {name: ordered[root] for name, root in roots.items()}
    sizes: dict[str, int] = {}
    for group in groups.values():
        sizes[group] = sizes.get(group, 0) + 1
    return {"nearest": nearest, "groups": groups, "group_count": len(ordered),
            "largest_group": max(sizes.values()) if sizes else 0,
            "group_sizes": dict(sorted(sizes.items()))}


def candidate_spacing(entries: list[tuple[str, object, tuple[float, float]]],
                      max_pair_km: float = 400.0) -> dict[str, tuple[float, str]]:
    """Nearest neighbouring candidate DTM for each candidate (polygon distance)."""
    return candidate_structure(entries, max_pair_km)["nearest"]


def audit_reasons(geometry_status: str, crs_valid: bool, crs_reason: str,
                  readme_status: str, readme_overlap: list[str],
                  footprint_disjoint: bool) -> list[str]:
    """PASS/FAIL reasons for one candidate; empty list means the candidate passes."""
    reasons = []
    if geometry_status != "OK":
        reasons.append(f"GEOMETRY:{geometry_status}")
    if not crs_valid:
        reasons.append(f"CRS:{crs_reason or 'invalid'}")
    if readme_status != "OK":
        reasons.append(f"SOURCE_README:{readme_status}")
    if readme_overlap:
        reasons.append("SOURCE_OVERLAP_WITH_SOMBENCH")
    if not footprint_disjoint:
        reasons.append("FOOTPRINT_NOT_DISJOINT")
    return reasons


def parse_readme_sources(text: str) -> dict:
    """Parse the 'LROC NAC SOURCE IMAGES' block from a PDS README."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if "LROC NAC SOURCE IMAGES" in line.upper()), None)
    if start is None:
        return {"section_found": False, "sources": [], "status": "NO_SOURCE_SECTION"}
    sources = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        for token in re.split(r"[,;\s]+", line.strip()):
            if not token:
                continue
            if SOURCE_TOKEN_RE.match(token):
                sources.append(PRODUCT_RE.search(token).group(1).upper())
            elif MENTION_RE.search(token):
                return {"section_found": True, "sources": [], "status": "MALFORMED_TOKEN",
                        "token": token}
    if not sources:
        return {"section_found": True, "sources": [], "status": "EMPTY_SECTION"}
    return {"section_found": True, "sources": sorted(set(sources)), "status": "OK"}


def fetch_readme(name: str, cache_dir: Path, timeout: int = 40) -> tuple[str | None, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace"), "CACHED"
    url = README_BASE.format(name=name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
        path.write_text(text, encoding="utf-8")
        return text, "FETCHED"
    except urllib.error.HTTPError as exc:
        return None, f"HTTP_{exc.code}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapefile", type=Path, default=Path("data/lroc_nac_dtm/shape/NAC_DTMS_360.SHP"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest_100k_final_v2.parquet"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels_hydrated.csv"))
    parser.add_argument("--protected", type=Path,
                        default=Path("protocols/heldout_v2_product_level_manifest.csv"))
    parser.add_argument("--readme-cache", type=Path, default=Path("data/lroc_nac_dtm/readmes"))
    parser.add_argument("--buffer-km", type=float, default=10.0)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/p19_lroc_footprint_audit.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/p19_lroc_footprint_audit.json"))
    parser.add_argument("--no-readme", action="store_true", help="skip PDS README fetching")
    args = parser.parse_args(argv)

    import shapefile  # pyshp

    manifest = pd.read_parquet(args.manifest)
    needed = {"sample_id", "upper_left_lon", "upper_left_lat", "lower_right_lon", "lower_right_lat"}
    missing = needed - set(manifest.columns)
    if missing:
        raise SystemExit(f"manifest lacks geographic corners: {sorted(missing)}; "
                         "rebuild with scripts/build_manifest.py")

    used_labels = pd.read_csv(args.labels, usecols=["sample_id"])
    used_ids = set(used_labels["sample_id"].astype(str))
    protected_ids = set()
    if args.protected.exists():
        protected_ids = set(pd.read_csv(args.protected)["sample_id"].astype(str))
    protected_ids |= used_ids
    footprints = tile_footprints(manifest, protected_ids)
    sombench_products = {p for raw in manifest["NAC_EDR_ID"].dropna()
                         for p in normalise_products(raw)}

    prj = args.shapefile.with_suffix(".PRJ")
    if not prj.exists():
        prj = args.shapefile.with_suffix(".prj")
    crs_info = crs_check(prj)

    reader = shapefile.Reader(str(args.shapefile))
    image_field = next((f for f in reader.fields[1:] if f[0] == "images"), None)
    image_field_size = int(image_field[2]) if image_field is not None else None
    rows = []
    geometries = []  # (DTM_name, polygon, (center_lat, center_lon)) for spacing checks
    for index in range(len(reader)):
        record = reader.record(index).as_dict()
        raw_sources = record.get("images", "")
        status = classify_source_field(raw_sources, image_field_size)
        if status != "OK":
            continue
        products = normalise_products(raw_sources)
        if not products or any(p in sombench_products for p in products):
            continue
        name = str(record.get("DTM_NAME", ""))
        shape = reader.shape(index)
        geometry, geometry_status = dtm_polygon(shape.points, list(shape.parts))
        center_lat = float(record.get("center_lat") or 0.0)
        center_lon = float(normalise_lon([float(record.get("center_lon") or 0.0)])[0])
        readme = {"status": "SKIPPED", "sources": []}
        if not args.no_readme:
            text, fetch_status = fetch_readme(name, args.readme_cache)
            readme = parse_readme_sources(text) if text else {"status": fetch_status, "sources": []}
        readme_sources = readme.get("sources", [])
        readme_overlap = sorted(set(readme_sources) & sombench_products)
        source_disjoint = readme.get("status") == "OK" and bool(readme_sources) and not readme_overlap
        min_km, nearest = (float("nan"), "")
        footprint_disjoint = False
        if geometry is not None and geometry_status == "OK" and footprints:
            min_km, nearest = distance_km(geometry, footprints, center_lat, center_lon)
            footprint_disjoint = bool(np.isfinite(min_km) and min_km >= args.buffer_km)
        area_km2 = None
        if geometry is not None and geometry_status == "OK":
            geometries.append((name, geometry, (center_lat, center_lon)))
            area_km2 = project_geometry(geometry, center_lat, center_lon).area / 1e6
        reasons = audit_reasons(geometry_status, bool(crs_info["valid"]),
                                crs_info.get("reason", ""), readme.get("status", "SKIPPED"),
                                readme_overlap, footprint_disjoint)
        rows.append({
            "DTM_name": name,
            "all_stereo_EDR_IDs": ";".join(readme_sources) if readme_sources else "",
            "dbf_products": ";".join(products),
            "resolution_m": float(record.get("resolution") or 0.0),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "coverage_sqkm": float(record.get("cov_sqkm") or 0.0),
            "polygon_area_sqkm": round(area_km2, 2) if area_km2 is not None else None,
            "geometry_valid": geometry_status == "OK",
            "geometry_reason": geometry_status,
            "n_parts": len(list(shape.parts)),
            "crs": crs_info.get("name", "UNKNOWN") if crs_info["valid"] else "UNVERIFIED",
            "crs_radius_m": crs_info.get("radius_m"),
            "min_polygon_distance_km": round(min_km, 4) if np.isfinite(min_km) else None,
            "nearest_protected_sample_id": nearest,
            "source_readme_status": readme.get("status"),
            "source_disjoint": source_disjoint,
            "footprint_disjoint": footprint_disjoint,
            "reason": "PASS" if not reasons else "|".join(reasons),
        })

    structure = candidate_structure(geometries)
    spacing = structure["nearest"]
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["min_candidate_distance_km"] = [
            round(spacing[name][0], 4) if name in spacing and np.isfinite(spacing[name][0]) else None
            for name in frame["DTM_name"]
        ]
        frame["nearest_candidate_DTM"] = [spacing.get(name, (0.0, ""))[1] for name in frame["DTM_name"]]
        frame["candidate_dtm_group"] = [structure["groups"].get(name, "") for name in frame["DTM_name"]]
        # None means no other candidate within the 400 km pair window, i.e. clearly disjoint.
        frame["candidate_spacing_disjoint"] = [
            bool(distance is None or distance >= args.buffer_km)
            for distance in frame["min_candidate_distance_km"]
        ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    passed = frame[(frame["source_disjoint"]) & (frame["footprint_disjoint"])]
    report = {
        "inputs": {"shapefile": str(args.shapefile), "manifest": str(args.manifest),
                   "labels": str(args.labels), "protected_tiles": len(protected_ids)},
        "crs_check": crs_info,
        "buffer_km": args.buffer_km,
        "candidates_screened": int(len(frame)),
        "source_audit": {
            key: int(value) for key, value in frame["source_readme_status"].value_counts().items()
        },
        "pass_both": int(len(passed)),
        "pass_names": passed["DTM_name"].tolist()[:50],
        "geometry_failures": int((~frame["geometry_valid"]).sum()),
        "geometry_failure_detail": frame.loc[~frame["geometry_valid"],
                                             ["DTM_name", "geometry_reason"]].values.tolist(),
        "footprint_failures": int((~frame["footprint_disjoint"]).sum()),
        "candidate_spacing": {
            "checked": int(len(geometries)),
            "group_count": structure["group_count"],
            "largest_group": structure["largest_group"],
            "pairs_closer_than_10km": int((~passed["candidate_spacing_disjoint"]).sum())
            if len(passed) else 0,
            "min_km": float(passed["min_candidate_distance_km"].min(skipna=True))
            if len(passed) and passed["min_candidate_distance_km"].notna().any() else None,
        },
        "passed_group_count": int(passed["candidate_dtm_group"].nunique()) if len(passed) else 0,
        "min_distance_km": float(frame["min_polygon_distance_km"].min(skipna=True))
        if frame["min_polygon_distance_km"].notna().any() else None,
        "verdict": ("CANDIDATES_PASS_SOURCE_AND_FOOTPRINT_QA" if len(passed) >= 10
                    else "BLOCKED_OR_UNVERIFIED"),
        "notes": [
            "Distances use a local azimuthal-equidistant projection on R=1737400 m.",
            "Used tile footprints are geographic UL/LR bboxes (conservative outer bound).",
            "Source lists come from the PDS README; DBF images are preliminary and truncated "
            "for mosaics.",
            "This is still an index/label audit: no DEM/NAC payload was downloaded.",
        ],
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:4000])
    print(f"wrote {args.out_csv} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
