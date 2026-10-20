#!/usr/bin/env python3
"""P20-03: per-scene CDR availability check against the README-verified acquisition list.

For every acquisition in the Gate-1 strict pool (native DTM post spacing <= 3 m, source and
footprint both PASS) the resolver output `outputs/p19_edr_resolved_urls.csv` is joined against
the acquisition list, so a scene is "available" only when both cameras resolved to a CDR
product. Missing halves, missing scenes and scene counts are reported; nothing is downloaded
and no model data is touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("outputs/p19_lroc_footprint_audit.csv"))
    parser.add_argument("--requests", type=Path,
                        default=Path("outputs/p19_edr_acquisition_list.csv"))
    parser.add_argument("--resolved", type=Path,
                        default=Path("outputs/p19_edr_resolved_urls.csv"))
    parser.add_argument("--max-resolution-m", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/p20_scene_availability.json"))
    args = parser.parse_args(argv)

    audit = pd.read_csv(args.audit)
    pool = audit[(audit["source_disjoint"].fillna(False))
                 & (audit["footprint_disjoint"].fillna(False))
                 & (audit["resolution_m"] <= args.max_resolution_m)]
    requests = pd.read_csv(args.requests, dtype=str)
    pool_ids = set(pool["DTM_name"])
    rows = requests[requests["DTM_name"].isin(pool_ids)]

    resolved = pd.read_csv(args.resolved, dtype=str)
    resolved = resolved[resolved["status"] == "UNVERIFIED_URL"]
    resolved["base"] = resolved["product_id"].str[:-2]
    resolved["camera"] = resolved["product_id"].str[-2]
    cameras = resolved.groupby("base")["camera"].apply(set).to_dict()

    scenes = {}
    for dtm_name, group in rows.groupby("DTM_name"):
        entries = []
        for edr_id in sorted(set(group["edr_id"].astype(str))):
            got = cameras.get(edr_id, set())
            entries.append({
                "edr_id": edr_id,
                "left": "L" in got,
                "right": "R" in got,
                "status": "BOTH" if {"L", "R"} <= got else ("PARTIAL" if got else "MISSING"),
            })
        scenes[dtm_name] = entries

    per_scene = {}
    for name, entries in scenes.items():
        complete = sum(1 for e in entries if e["status"] == "BOTH")
        partial = sum(1 for e in entries if e["status"] == "PARTIAL")
        per_scene[name] = {
            "acquisitions": len(entries),
            "acquisitions_both_cameras": complete,
            "acquisitions_partial": partial,
            "acquisitions_missing": len(entries) - complete - partial,
        }

    full = [name for name, value in per_scene.items()
            if value["acquisitions_both_cameras"] == value["acquisitions"]]
    summary = {
        "pool_scenes": len(scenes),
        "scenes_with_all_acquisitions_both_cameras": len(full),
        "scenes_complete": full,
        "acquisitions": {
            "total": sum(v["acquisitions"] for v in per_scene.values()),
            "both": sum(v["acquisitions_both_cameras"] for v in per_scene.values()),
            "partial": sum(v["acquisitions_partial"] for v in per_scene.values()),
            "missing": sum(v["acquisitions_missing"] for v in per_scene.values()),
        },
        "per_scene": per_scene,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_scene"}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
