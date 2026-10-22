"""P2.1 workstream B: build the v2 held-out group pool from real polygon grouping.

Consumes the output of ``scripts/audit_groups_pairwise.py`` (all 2346 polygon distances, 10 km
single-linkage groups, and the measured overlap areas of the zero-distance pairs) and the DTM
shapefile metadata, then writes a v2 pool plus a deterministic selection.

Two rules go beyond v1 because the geometry proved they matter:

* One scene per overlapping neighbour inside a group. Several products cover the same ground from
  different stereos (LICHTENBER2 and LICHTENBER7 share 112.7 km2), so taking two of them would
  spend capacity on the same terrain twice.
* Groups are the bootstrap unit and are separated by at least the threshold; the measured minimum
  inter-group distance is reported so a reviewer can check the margin.

Ordering stays ``sha256(group_id)`` and scene choice ``sha256(scene)``, so the selection cannot
depend on any measurement. v1 files are left untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import importlib.util
from pathlib import Path


def _load(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_id(members) -> str:
    return "GRP_" + hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:12]


def scene_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def build_pool(groups_json: dict, scene_meta: dict, probes) -> list:
    """One record per polygon group, sorted by selection rank."""
    overlap_pairs = {(a, b): area for a, b, _d, area in groups_json["zero_distance_overlapping"]}
    overlap_pairs.update({(b, a): area for (a, b), area in list(overlap_pairs.items())})

    records = []
    for members in groups_json["members"].values():
        members = sorted(members)
        gid = group_id(members)
        resolutions = sorted({str(scene_meta[m]["resolution"]) for m in members})
        overlaps = [overlap_pairs.get((a, b), 0.0)
                    for i, a in enumerate(members) for b in members[i + 1:]]
        records.append({
            "group_id": gid,
            "members": members,
            "n_scenes": len(members),
            "native_resolution_m": resolutions,
            "stereo_sources": sorted({str(scene_meta[m]["images"]) for m in members}),
            "max_internal_overlap_km2": round(max(overlaps) if overlaps else 0.0, 4),
            "center_lat": round(sum(float(scene_meta[m]["center_lat"]) for m in members) / len(members), 5),
            "center_lon": round(sum(float(scene_meta[m]["center_lon"]) for m in members) / len(members), 5),
            "footprint_disjoint_all": all(scene_meta[m]["footprint_disjoint"] == "True" for m in members),
            "source_disjoint_all": all(scene_meta[m]["source_disjoint"] == "True" for m in members),
            "contains_probe": [m for m in members if m in probes],
        })
    records.sort(key=lambda r: hashlib.sha256(r["group_id"].encode()).hexdigest())
    for rank, record in enumerate(records, start=1):
        record["selection_rank"] = rank
    return records


def eligible(record) -> bool:
    return not record["contains_probe"] and record["footprint_disjoint_all"] \
        and record["source_disjoint_all"]


def select(records, target_groups: int, max_scenes_per_group: int, overlap_pairs):
    """Groups in rank order; inside a group, scenes that do not overlap an already chosen one."""
    ordered = sorted(records, key=lambda r: r.get("selection_rank", 0))
    chosen = [r for r in ordered if eligible(r)][:target_groups]
    scenes, per_group, skipped = [], {}, []
    for record in chosen:
        picked = []
        for name in sorted(record["members"], key=scene_key):
            if len(picked) >= max_scenes_per_group:
                break
            if any(overlap_pairs.get((name, other), 0.0) > 1.0 for other in picked):
                skipped.append({"group_id": record["group_id"], "scene": name,
                                "reason": "overlaps an already selected scene by >1 km2"})
                continue
            picked.append(name)
        per_group[record["group_id"]] = picked
        scenes.extend(picked)
    return {"groups": [r["group_id"] for r in chosen], "scenes": scenes,
            "scenes_per_group": per_group, "skipped": skipped}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-json", type=Path, required=True)
    parser.add_argument("--footprint-csv", type=Path, required=True)
    parser.add_argument("--shapefile", type=Path, default=None,
                        help="to read stereo sources and resolution per DTM")
    parser.add_argument("--probe", action="append", default=None)
    parser.add_argument("--target-groups", type=int, default=15)
    parser.add_argument("--max-scenes-per-group", type=int, default=2)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    groups_json = json.loads(args.groups_json.read_text(encoding="utf-8"))
    probes = tuple(args.probe) if args.probe else ("FRSHCRATER10", "RANGER", "PLANCKFLOOR")

    meta = {}
    with args.footprint_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        name_field = "DTM_name" if "DTM_name" in (reader.fieldnames or []) else "DTM_NAME"
        for row in reader:
            meta[row[name_field]] = {"footprint_disjoint": row["footprint_disjoint"],
                                     "source_disjoint": row["source_disjoint"],
                                     "center_lat": row["center_lat"],
                                     "center_lon": row["center_lon"],
                                     "resolution": row["resolution_m"],
                                     "images": row.get("all_stereo_EDR_IDs", "")}
    if args.shapefile:
        import shapefile
        reader = shapefile.Reader(str(args.shapefile))
        fields = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
        index = {name: i for i, name in enumerate(fields)}
        for record in reader.records():
            name = record[index["DTM_NAME"]]
            if name in meta:
                meta[name]["images"] = str(record[index["images"]]).strip()
                meta[name]["resolution"] = record[index["resolution"]]

    records = build_pool(groups_json, meta, probes)
    overlap_pairs = {(a, b): area for a, b, _d, area in groups_json["zero_distance_overlapping"]}
    overlap_pairs.update({(b, a): area for (a, b), area in list(overlap_pairs.items())})
    selection = select(records, args.target_groups, args.max_scenes_per_group, overlap_pairs)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["selection_rank", "group_id", "n_scenes", "members", "native_resolution_m",
              "stereo_sources", "max_internal_overlap_km2", "center_lat", "center_lon",
              "footprint_disjoint_all", "source_disjoint_all", "contains_probe"]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {k: record[k] for k in fields}
            row["members"] = ";".join(record["members"])
            row["native_resolution_m"] = ";".join(record["native_resolution_m"])
            row["stereo_sources"] = "|".join(record["stereo_sources"])
            row["contains_probe"] = ";".join(record["contains_probe"])
            writer.writerow(row)

    summary = {
        "groups": len(records),
        "eligible_groups": sum(1 for r in records if eligible(r)),
        "eligible_scenes": sum(r["n_scenes"] for r in records if eligible(r)),
        "largest_group": max((r["n_scenes"] for r in records), default=0),
        "min_intergroup_distance_km": groups_json.get("min_intergroup_distance_km"),
        "overlapping_pairs_within_groups": len(groups_json.get("zero_distance_overlapping", [])),
        "touching_pairs": len(groups_json.get("zero_distance_touching", [])),
        "probe_groups": [r["group_id"] for r in records if r["contains_probe"]],
        "selection": {
            "target_groups": args.target_groups,
            "max_scenes_per_group": args.max_scenes_per_group,
            "groups": selection["groups"],
            "scenes": selection["scenes"],
            "expected_scenes": len(selection["scenes"]),
            "scenes_per_group": selection["scenes_per_group"],
            "skipped_scenes": selection["skipped"],
            "selection_hash": hashlib.sha256("|".join(selection["groups"]).encode()).hexdigest(),
            "scene_hash": hashlib.sha256("|".join(selection["scenes"]).encode()).hexdigest(),
        },
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("groups", "eligible_groups", "eligible_scenes",
                                              "largest_group", "min_intergroup_distance_km",
                                              "overlapping_pairs_within_groups",
                                              "touching_pairs")}, indent=2))
    print("selected", summary["selection"]["expected_scenes"], "scenes in",
          len(summary["selection"]["groups"]), "groups;",
          len(summary["selection"]["skipped_scenes"]), "scene(s) skipped for overlap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
