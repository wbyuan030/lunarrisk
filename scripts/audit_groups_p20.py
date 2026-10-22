"""P20-03 independence audit: how many separable groups survive a 10 km spacing rule.

DEPRECATED for freeze decisions (P2.1 workstream B). This clustering unioned only each DTM's
nearest-candidate edge, which is not single linkage: a scene can sit within the threshold of another
without being anyone's nearest neighbour, and nearest neighbours were computed over a larger
candidate set. Measured consequence: 51 groups here versus 41 from the all-pairs polygon rebuild in
``scripts/audit_groups_pairwise.py``, which also merges SPARIM1/SPARIM2 that this code splits.
Kept for history; use the pairwise script for any held-out pool.

Inputs are the existing Gate 1 audit products, so this adds no new data claims:

* ``outputs/p19_lroc_footprint_pass.csv`` - one row per candidate DTM with
  ``min_candidate_distance_km`` (minimum polygon-to-polygon distance to any other candidate) and
  ``nearest_candidate_DTM``.
* ``outputs/p20_scene_availability.json`` - which scenes have every acquisition with L+R.

Each row gives the nearest-candidate edge, which is exactly what single-linkage clustering needs, so
groups are formed by union-find with a distance threshold instead of by name prefix. A group here is
the unit a bootstrap must resample; scenes inside one group may not be counted as independent.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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


def cluster(rows, threshold_km: float, scenes=None):
    """Single-linkage groups from nearest-candidate edges; only edges under the threshold merge."""
    keep = {r["DTM_name"] for r in rows} if scenes is None else set(scenes)
    uf = UnionFind()
    for row in rows:
        name = row["DTM_name"]
        if name not in keep:
            continue
        uf.find(name)
        other = row.get("nearest_candidate_DTM") or ""
        distance = row.get("min_candidate_distance_km") or ""
        if other in keep and other and distance != "":
            if float(distance) < threshold_km:
                uf.union(name, other)
    return {root: sorted(members) for root, members in uf.groups().items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--footprint-csv", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--threshold-km", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    with args.footprint_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    availability = json.loads(args.availability.read_text(encoding="utf-8"))
    complete = availability.get("scenes_complete") or []
    by_name = {r["DTM_name"]: r for r in rows}

    usable = [name for name in complete if name in by_name]
    missing_rows = [name for name in complete if name not in by_name]

    groups = cluster(rows, args.threshold_km, usable)
    sizes = sorted((len(m) for m in groups.values()), reverse=True)

    report = {"threshold_km": args.threshold_km,
              "scenes_complete": len(complete),
              "scenes_with_footprint_row": len(usable),
              "scenes_missing_footprint_row": missing_rows,
              "groups": len(groups),
              "group_sizes": sizes,
              "independent_scenes": sum(1 for s in sizes if s == 1),
              "largest_group": max(groups.values(), key=len) if groups else [],
              "multi_scene_groups": [members for members in groups.values() if len(members) > 1]}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
