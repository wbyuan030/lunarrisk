"""P20-04 prerequisite: build the held-out group pool and its preregistered selection order.

Selection must be deterministic and fixed *before* any score exists, so it cannot depend on which
scene looks easy. Two inputs only:

* ``outputs/p19_lroc_footprint_pass.csv`` - candidate DTMs with footprint/source flags,
* ``outputs/p20_scene_availability.json`` - which scenes have every acquisition with both cameras.

Groups come from the same 10 km single-linkage rule as ``scripts/audit_groups_p20.py`` (imported, not
reimplemented, so the two cannot drift). The ordering key is ``sha256(group_id)``: it is stable,
independent of names and of any measurement, and lets a reviewer re-derive the order by hand.

Scenes used as probes are excluded by name and their groups are marked, because a scene whose
orthos were inspected may not enter the held-out set.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def _load_audit():
    """Import the group audit from the same directory without requiring sys.path setup."""
    path = Path(__file__).resolve().parent / "audit_groups_p20.py"
    spec = importlib.util.spec_from_file_location("audit_groups_p20", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()

PROBES = ("FRSHCRATER10", "RANGER", "PLANCKFLOOR")


def group_id(members) -> str:
    """Stable identifier from the sorted members, so re-grouping cannot silently rename a group."""
    return "GRP_" + hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:12]


def order_key(gid: str) -> str:
    """Preregistered ordering: hash of the group id, not of any measurement."""
    return hashlib.sha256(gid.encode("utf-8")).hexdigest()


def _min_distance(by_name, members):
    """Minimum nearest-candidate distance within a group; '' means no distance was recorded."""
    values = []
    for name in members:
        raw = by_name[name].get("min_candidate_distance_km", "")
        if raw not in ("", None):
            values.append(float(raw))
    return min(values) if values else None


def build(rows, availability, threshold_km: float, probes=PROBES):
    by_name = {r["DTM_name"]: r for r in rows}
    complete = [name for name in availability.get("scenes_complete", []) if name in by_name]
    groups = audit.cluster(rows, threshold_km, complete)

    records = []
    for members in groups.values():
        members = sorted(members)
        gid = group_id(members)
        resolutions = sorted({by_name[m]["resolution_m"] for m in members})
        records.append({
            "group_id": gid,
            "members": members,
            "n_scenes": len(members),
            "native_resolution_m": resolutions,
            "center_lat": round(sum(float(by_name[m]["center_lat"]) for m in members) / len(members), 5),
            "center_lon": round(sum(float(by_name[m]["center_lon"]) for m in members) / len(members), 5),
            "footprint_disjoint_all": all(by_name[m]["footprint_disjoint"] == "True" for m in members),
            "source_disjoint_all": all(by_name[m]["source_disjoint"] == "True" for m in members),
            "contains_probe": [m for m in members if m in probes],
            "min_candidate_distance_km": _min_distance(by_name, members),
        })
    records.sort(key=lambda r: order_key(r["group_id"]))
    for rank, record in enumerate(records, start=1):
        record["selection_rank"] = rank
    return records


def scene_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def select(records, target_groups: int, max_scenes_per_group: int):
    """Pick groups in rank order and, inside each, scenes by hash -- deterministic by construction.

    ``max_scenes_per_group`` exists because one group holds 11 LICHTENBERG scenes: without a cap
    the held-out set would be one site sampled eleven times, which the group rule already forbids
    for bootstrap and should forbid for coverage too.
    """
    ordered = sorted(records, key=lambda r: r.get("selection_rank", 0))
    chosen = [r for r in ordered if eligible(r)][:target_groups]
    scenes, per_group = [], {}
    for record in chosen:
        members = sorted(record["members"], key=scene_key)[:max_scenes_per_group]
        per_group[record["group_id"]] = members
        scenes.extend(members)
    return {"groups": [r["group_id"] for r in chosen], "scenes": scenes,
            "scenes_per_group": per_group}


def eligible(record) -> bool:
    """A group may be held out only if it is untouched by the probes and fully disjoint."""
    return not record["contains_probe"] and record["footprint_disjoint_all"] \
        and record["source_disjoint_all"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--footprint-csv", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--threshold-km", type=float, default=10.0)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--target-groups", type=int, default=None)
    parser.add_argument("--max-scenes-per-group", type=int, default=2)
    parser.add_argument("--probe", action="append", default=None,
                        help="scene name already inspected as a probe (repeatable)")
    args = parser.parse_args(argv)

    import csv
    with args.footprint_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    availability = json.loads(args.availability.read_text(encoding="utf-8"))
    records = build(rows, availability, args.threshold_km,
                    tuple(args.probe) if args.probe else PROBES)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["selection_rank", "group_id", "n_scenes", "members", "native_resolution_m",
              "center_lat", "center_lon", "footprint_disjoint_all", "source_disjoint_all",
              "contains_probe", "min_candidate_distance_km"]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["members"] = ";".join(record["members"])
            row["native_resolution_m"] = ";".join(record["native_resolution_m"])
            row["contains_probe"] = ";".join(record["contains_probe"])
            writer.writerow({k: row[k] for k in fields})

    summary = {"threshold_km": args.threshold_km, "groups": len(records),
               "eligible_groups": sum(1 for r in records if eligible(r)),
               "eligible_scenes": sum(r["n_scenes"] for r in records if eligible(r)),
               "probe_groups": [r["group_id"] for r in records if r["contains_probe"]],
               "ordering": "sha256(group_id), ascending",
               "largest_group": max((r["n_scenes"] for r in records), default=0)}
    if args.target_groups:
        selection = select(records, args.target_groups, args.max_scenes_per_group)
        summary["selection"] = {
            "target_groups": args.target_groups,
            "max_scenes_per_group": args.max_scenes_per_group,
            "groups": selection["groups"],
            "scenes": selection["scenes"],
            "expected_scenes": len(selection["scenes"]),
            "scenes_per_group": selection["scenes_per_group"],
            "selection_hash": hashlib.sha256("|".join(selection["groups"]).encode()).hexdigest(),
            "scene_hash": hashlib.sha256("|".join(selection["scenes"]).encode()).hexdigest()}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
