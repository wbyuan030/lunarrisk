#!/usr/bin/env python3
"""Conservative pre-freeze tripwire for suspicious cross-group DTM locations.

This deliberately does NOT establish polygon-level independence: its job is to
reject a pending external heldout selection if nearby group centroids imply
the existing *nearest-neighbour-only* clustering may have split one area.
The correction must use ALL eligible polygon pairs, then regenerate pool and
versioned protocol. Never use this tripwire to certify 10 km separation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

MOON_RADIUS_KM = 1737.4


def spherical_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Central-angle distance; longitude wraparound works at +/-180 deg."""
    a, b = math.radians(lat1), math.radians(lat2)
    dlat = b - a
    dlon = math.radians((lon2 - lon1 + 180) % 360 - 180)
    h = math.sin(dlat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(dlon / 2) ** 2
    return 2 * MOON_RADIUS_KM * math.asin(min(1.0, math.sqrt(max(0.0, h))))


def parse_pool(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty group pool: fail closed")
    seen = set()
    for row in rows:
        gid = row.get("group_id", "").strip()
        if not gid or gid in seen:
            raise ValueError(f"missing or duplicate group ID {gid!r}")
        seen.add(gid)
        for field in ("center_lat", "center_lon"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{gid} lacks valid {field}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{gid} has nonfinite {field}")
            row[field] = value
        if not -90 <= row["center_lat"] <= 90:
            raise ValueError(f"invalid lunar latitude in {gid}")
    return rows


def collision_report(pool: list[dict], protocol: dict, threshold_km: float = 10.0) -> dict:
    if not (math.isfinite(threshold_km) and threshold_km > 0):
        raise ValueError("threshold must be finite and positive")
    groups = {row["group_id"] for row in pool}
    chosen = protocol.get("selection", {}).get("selected_group_ids", [])
    if not isinstance(chosen, list) or len(set(chosen)) != len(chosen):
        raise ValueError("selected groups must be a nonduplicated list")
    if any(item not in groups for item in chosen):
        raise ValueError("selection references missing group IDs; fail closed")
    chosen_set = set(chosen)
    pairs = []
    for i, left in enumerate(pool):
        for right in pool[i + 1:]:
            dist = spherical_distance_km(
                left["center_lat"], left["center_lon"],
                right["center_lat"], right["center_lon"],
            )
            if dist < threshold_km:
                pairs.append({
                    "group_a": left["group_id"],
                    "group_b": right["group_id"],
                    "group_a_members": left.get("members", ""),
                    "group_b_members": right.get("members", ""),
                    "center_distance_km": round(dist, 5),
                    "a_selected": left["group_id"] in chosen_set,
                    "b_selected": right["group_id"] in chosen_set,
                    "note": "centroid proximity only; actual polygon separation UNKNOWN",
                })
    risk = [p for p in pairs if p["a_selected"] or p["b_selected"]]
    return {
        "status": ("BLOCKED_NEEDS_FULL_PAIRWISE_POLYGON_AUDIT" if risk
                   else "UNCERTIFIED_NEEDS_FULL_PAIRWISE_POLYGON_AUDIT"),
        "group_count": len(pool),
        "selected_group_count": len(chosen),
        "threshold_km": threshold_km,
        "centroid_collision_pairs": len(pairs),
        "pairs_touching_selection": len(risk),
        "examples": pairs[:100],
        "warning": (
            "Centroids cannot prove polygon separation. Even if there are NO "
            "centroid collisions, final signoff REQUIRES all eligible polygon-"
            "to-polygon pairs, connected components and inter-group min-distance."
        ),
        "gate3": "NOT_FROZEN",
        "gate4": "HOLD",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("protocols/heldout_v2_group_pool.csv"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("protocols/heldout_v2_external_protocol.json"))
    parser.add_argument("--threshold-km", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/p21_group_collision_check.json"))
    args = parser.parse_args(argv)
    pool = parse_pool(args.pool)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = collision_report(pool, protocol, args.threshold_km)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    # Fail closed by design: no centroid-only check may authorize a freeze.
    return 2 if report["pairs_touching_selection"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
