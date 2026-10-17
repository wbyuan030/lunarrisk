#!/usr/bin/env python3
"""Export the stereo NAC EDR acquisition list for the policy-A candidate pool.

Inputs are the Gate 1 audit CSV (which already carries the README-verified stereo source
ids per scene) and the audit JSON (acquisition windows). Output is the shopping list for
the institutional EDR retrieval step described in docs/P1_9_EDR_ORTHO_RUNBOOK.md.
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
    parser.add_argument("--max-resolution-m", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/p19_edr_acquisition_list.csv"))
    args = parser.parse_args(argv)

    audit = pd.read_csv(args.audit)
    pool = audit[(audit["source_disjoint"].fillna(False)) & (audit["footprint_disjoint"].fillna(False))
                 & (audit["resolution_m"] <= args.max_resolution_m)].copy()
    rows = []
    for row in pool.itertuples():
        edrs = [token for token in str(row.all_stereo_EDR_IDs or "").split(";") if token]
        for index, edr in enumerate(edrs):
            rows.append({
                "DTM_name": row.DTM_name,
                "candidate_dtm_group": row.candidate_dtm_group,
                "edr_id": edr,
                "camera": "L/R",  # both cameras of the stereo pair are needed
                "role": "primary" if index == 0 else "secondary",
                "dtm_resolution_m": row.resolution_m,
                "dtm_map_projection": row.CRS if hasattr(row, "CRS") else "",
                "center_lat": round(row.center_lat, 5),
                "center_lon": round(row.center_lon, 5),
                "min_clearance_km": row.min_polygon_distance_km,
                "dtm_readme_sha256_group": row.source_readme_status,
            })
    frame = pd.DataFrame(rows).sort_values(["DTM_name", "edr_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = {
        "scenes": int(frame.DTM_name.nunique()),
        "edr_ids": int(frame.edr_id.nunique()),
        "rows": int(len(frame)),
        "max_resolution_m": args.max_resolution_m,
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
