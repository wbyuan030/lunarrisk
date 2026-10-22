#!/usr/bin/env python3
"""Offline tests for conservative group-pool pre-freeze collision tripwire."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_group_pool_collision", ROOT / "scripts" / "check_group_pool_collision.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def group(gid, lat, lon, members="X"):
    return {"group_id": gid, "center_lat": lat,
            "center_lon": lon, "members": members}


def proto(*selected):
    return {"selection": {"selected_group_ids": list(selected)},
            "gate3_status": "NOT_FROZEN", "gate4_status": "HOLD"}


class TestCollisionTripwire(unittest.TestCase):
    def test_spa_style_duplicate_centers_fail_pending_freeze(self):
        out = mod.collision_report([
            group("A", -51.0, 171.0, "SPARIM1"),
            group("B", -51.0, 171.0, "SPARIM2"),
            group("C", 20.0, -40.0),
        ], proto("A", "B"))
        self.assertEqual(out["status"], "BLOCKED_NEEDS_FULL_PAIRWISE_POLYGON_AUDIT")
        self.assertEqual(out["pairs_touching_selection"], 1)
        self.assertEqual(out["examples"][0]["center_distance_km"], 0)

    def test_one_selected_one_other_nearby_is_not_ignored(self):
        out = mod.collision_report([
            group("A", 0.0, 0.0),
            group("B", 0.02, 0.0),
        ], proto("A"))
        self.assertEqual(out["pairs_touching_selection"], 1)

    def test_dateline_short_path(self):
        self.assertLess(mod.spherical_distance_km(0, 179.99, 0, -179.99), 1.0)

    def test_far_centers_still_not_a_proof_of_polygon_clearance(self):
        out = mod.collision_report([
            group("A", 0.0, 0.0),
            group("B", 45.0, 130.0),
        ], proto("A"))
        self.assertEqual(out["pairs_touching_selection"], 0)
        self.assertIn("UNCERTIFIED", out["status"])
        self.assertIn("polygon-to-polygon", out["warning"])

    def test_reject_missing_or_duplicate_selection_groups(self):
        rows = [group("A", 0, 0)]
        with self.assertRaises(ValueError):
            mod.collision_report(rows, proto("A", "A"))
        with self.assertRaises(ValueError):
            mod.collision_report(rows, proto("MISSING"))

    def test_reject_bad_pool_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.csv"
            path.write_text(
                "group_id,center_lat,center_lon,members\n"
                "A,nan,0,X\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                mod.parse_pool(path)

    def test_bad_threshold_rejected(self):
        with self.assertRaises(ValueError):
            mod.collision_report([group("A", 0, 0)], proto("A"), -10)


if __name__ == "__main__":
    unittest.main()
