"""P2.1 workstream B: real polygon-to-polygon 10 km grouping for the held-out pool.

The previous grouping (``audit_groups_p20.py``) unioned only each DTM's nearest-neighbour edge. That
is not single-linkage: a scene can be within 10 km of another without being anyone's nearest
neighbour, and nearest neighbours were computed over a larger candidate set, so edges can vanish
after filtering. The consequence was visible in the pool: SPARIM1 and SPARIM2 share a centre, both
recorded a nearest distance of 0 km, yet ended up in different groups.

This script rebuilds the graph from geometry:

1. read the LROC DTM polygons from the shapefile (multipart preserved),
2. for every pair of usable scenes, project both polygons into a local azimuthal equidistant
   projection centred on their midpoint (sphere R = 1737.4 km), then take the shapely distance,
3. add an edge for every pair below the threshold and run union-find on all of them,
4. report every pair distance, the members, and the minimum distance between different groups.

Longitudes are shifted by +-360 to within 180 degrees of the local centre before projecting, so
antimeridian-crossing polygons are handled without splitting them.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform

MOON_RADIUS_M = 1737400.0
SRC = CRS.from_proj4(f"+proj=longlat +R={MOON_RADIUS_M} +no_defs")


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        out = {}
        for item in list(self.parent):
            out.setdefault(self.find(item), []).append(item)
        return out


def load_polygons(path: Path, wanted=None) -> dict:
    """Return {DTM_NAME: [rings]}; each ring is a list of (lon, lat) tuples."""
    reader = shapefile.Reader(str(path))
    # pyshp's fields list starts with DeletionFlag but record objects do not include it, so the
    # index has to be computed on the field names minus that flag
    field_names = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
    name_index = field_names.index("DTM_NAME")
    polygons = {}
    for record, shape in zip(reader.records(), reader.shapes()):
        name = record[name_index]
        if wanted is not None and name not in wanted:
            continue
        parts = list(shape.parts) + [len(shape.points)]
        rings = [shape.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
        rings = [ring for ring in rings if len(ring) >= 3]
        if rings:
            polygons[name] = rings
    return polygons


def shift_lon(rings, lon0: float):
    """Shift longitudes into (lon0 - 180, lon0 + 180] so antimeridian polygons project cleanly."""
    out = []
    for ring in rings:
        shifted = []
        for lon, lat in ring:
            while lon - lon0 > 180.0:
                lon -= 360.0
            while lon - lon0 < -180.0:
                lon += 360.0
            shifted.append((lon, lat))
        out.append(shifted)
    return out


def _ring_centre(rings):
    lons = [p[0] for ring in rings for p in ring]
    lats = [p[1] for ring in rings for p in ring]
    # circular mean for longitude
    import math
    east = sum(math.cos(math.radians(v)) for v in lons)
    north = sum(math.sin(math.radians(v)) for v in lons)
    centre_lon = math.degrees(math.atan2(north, east))
    return centre_lon, sum(lats) / len(lats)


def distance_km(rings_a, rings_b) -> float:
    """Minimum polygon-to-polygon distance in km on the Moon sphere."""
    lon_a, lat_a = _ring_centre(rings_a)
    lon_b, lat_b = _ring_centre(rings_b)
    import math
    lat0 = 0.5 * (lat_a + lat_b)
    # midpoint longitude, shortest way round
    delta = ((lon_b - lon_a + 180.0) % 360.0) - 180.0
    lon0 = lon_a + 0.5 * delta
    target = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +R={MOON_RADIUS_M} +units=m +no_defs")
    transformer = Transformer.from_crs(SRC, target, always_xy=True)

    def project(rings):
        polygons = []
        for ring in shift_lon(rings, lon0):
            shell = shapely_transform(transformer.transform, Polygon(ring))
            polygons.append(shell)
        # a multipart DTM: distance to the union is the min over parts
        return polygons

    parts_a, parts_b = project(rings_a), project(rings_b)
    best = float("inf")
    for pa in parts_a:
        for pb in parts_b:
            if not pa.is_valid:
                pa = pa.buffer(0)
            if not pb.is_valid:
                pb = pb.buffer(0)
            best = min(best, pa.distance(pb))
    return best / 1000.0


def overlap_km2(rings_a, rings_b) -> float:
    """Intersection area in km^2 for a pair of DTM polygons.

    Zero distance alone does not distinguish "shares a border" from "covers the same ground", and
    only the second is a duplicate product. Reported for close pairs so the distinction is explicit.
    """
    from shapely.geometry import Polygon
    from shapely.ops import transform as shapely_transform
    from pyproj import CRS, Transformer
    lon_a, lat_a = _ring_centre(rings_a)
    lon_b, lat_b = _ring_centre(rings_b)
    delta = ((lon_b - lon_a + 180.0) % 360.0) - 180.0
    lon0 = lon_a + 0.5 * delta
    target = CRS.from_proj4(
        f"+proj=aeqd +lat_0={0.5 * (lat_a + lat_b)} +lon_0={lon0} +R={MOON_RADIUS_M} "
        f"+units=m +no_defs")
    transformer = Transformer.from_crs(SRC, target, always_xy=True)

    def union(rings):
        polys = []
        for ring in shift_lon(rings, lon0):
            shell = shapely_transform(transformer.transform, Polygon(ring))
            if not shell.is_valid:
                shell = shell.buffer(0)
            polys.append(shell)
        out = polys[0]
        for extra in polys[1:]:
            out = out.union(extra)
        return out

    return union(rings_a).intersection(union(rings_b)).area / 1e6


def build_edges(polygons: dict, threshold_km: float):
    """Every pair below the threshold, not just nearest neighbours."""
    names = sorted(polygons)
    edges = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = distance_km(polygons[a], polygons[b])
            edges.append((a, b, d))
    close = [(a, b, d) for a, b, d in edges if d < threshold_km]
    return edges, close


def group_from_edges(names, edges):
    uf = UnionFind()
    for name in names:
        uf.find(name)
    for a, b, _ in edges:
        uf.union(a, b)
    return {root: sorted(members) for root, members in uf.groups().items()}


def min_intergroup_km(groups, distances: dict) -> dict:
    """Smallest polygon distance between any two different groups."""
    out = {}
    roots = sorted(groups, key=lambda r: groups[r][0])
    for i, ra in enumerate(roots):
        for rb in roots[i + 1:]:
            best = min((distances[(x, y)] if (x, y) in distances else distances[(y, x)])
                       for x in groups[ra] for y in groups[rb])
            out[f"{groups[ra][0]}|{groups[rb][0]}"] = round(best, 4)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapefile", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--threshold-km", type=float, default=10.0)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--groups-json", type=Path, required=True)
    args = parser.parse_args(argv)

    availability = json.loads(args.availability.read_text(encoding="utf-8"))
    scenes = sorted(set(availability.get("scenes_complete") or []))
    polygons = load_polygons(args.shapefile, set(scenes))
    missing = [s for s in scenes if s not in polygons]

    edges, close = build_edges(polygons, args.threshold_km)
    groups = group_from_edges(sorted(polygons), close)
    distances = {(a, b): d for a, b, d in edges}

    touching, overlapping = [], []
    areas = {}
    for a, b, d in close:
        if d > 0.5:
            continue
        area = overlap_km2(polygons[a], polygons[b])
        areas[(a, b)] = area
        entry = [a, b, round(d, 6), round(area, 6)]
        (overlapping if area > 1e-6 else touching).append(entry)

    args.pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene_a", "scene_b", "polygon_distance_km", "edge_below_threshold",
                         "overlap_area_km2"])
        for a, b, d in sorted(edges, key=lambda e: e[2]):
            writer.writerow([a, b, f"{d:.6f}", d < args.threshold_km,
                             "" if (a, b) not in areas else f"{areas[(a, b)]:.6f}"])

    intergroup = min_intergroup_km(groups, distances)
    members = {root: sorted(m) for root, m in groups.items()}
    report = {
        "threshold_km": args.threshold_km,
        "moon_radius_m": MOON_RADIUS_M,
        "scenes_requested": len(scenes),
        "scenes_with_geometry": len(polygons),
        "scenes_missing_geometry": missing,
        "pairs_evaluated": len(edges),
        "edges_below_threshold": len(close),
        "groups": len(groups),
        "group_sizes": sorted((len(m) for m in members.values()), reverse=True),
        "members": members,
        "min_intergroup_distance_km": min(intergroup.values()) if intergroup else None,
        "intergroup_distances_km": intergroup,
        "sparim": {name: [sorted(m) for m in members.values() if name in m]
                   for name in ("SPARIM1", "SPARIM2")},
        "edge_closed_pairs": [[a, b, round(d, 6)] for a, b, d in sorted(close, key=lambda e: e[2])],
        "zero_distance_touching": touching,
        "zero_distance_overlapping": overlapping,
        "overlap_area_km2": {f"{a}|{b}": area for a, b, _, area in overlapping},
    }
    args.groups_json.parent.mkdir(parents=True, exist_ok=True)
    args.groups_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("scenes_requested", "scenes_with_geometry",
                                             "pairs_evaluated", "edges_below_threshold", "groups",
                                             "group_sizes", "min_intergroup_distance_km",
                                             "scenes_missing_geometry", "sparim")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
