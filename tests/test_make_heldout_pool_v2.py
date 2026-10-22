#!/usr/bin/env python3
"""Offline tests for the v2 pool: group records and overlap-aware scene selection."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_heldout_pool_v2",
                                              ROOT / "scripts" / "make_heldout_pool_v2.py")
v2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)


def meta(images="M1, M2", resolution="2.0", footprint="True", source="True"):
    return {"images": images, "resolution": resolution, "footprint_disjoint": footprint,
            "source_disjoint": source, "center_lat": "10.0", "center_lon": "20.0"}


def groups_json(members, overlapping=()):
    return {"members": {f"r{i}": m for i, m in enumerate(members)},
            "zero_distance_overlapping": [[a, b, 0.0, area] for a, b, area in overlapping],
            "zero_distance_touching": [], "min_intergroup_distance_km": 40.0}


class TestBuildPool(unittest.TestCase):
    def test_records_carry_overlap_and_stereo_sources(self):
        doc = groups_json([["A", "B"]], overlapping=[("A", "B", 12.5)])
        records = v2.build_pool(doc, {"A": meta("M1, M2"), "B": meta("M3, M4")}, probes=())
        self.assertEqual(records[0]["max_internal_overlap_km2"], 12.5)
        self.assertEqual(len(records[0]["stereo_sources"]), 2)

    def test_probe_membership_makes_a_group_ineligible(self):
        doc = groups_json([["A", "RANGER"]])
        records = v2.build_pool(doc, {"A": meta(), "RANGER": meta()}, probes=("RANGER",))
        self.assertEqual(records[0]["contains_probe"], ["RANGER"])
        self.assertFalse(v2.eligible(records[0]))

    def test_disjoint_flag_is_required(self):
        doc = groups_json([["A"], ["B"]])
        records = v2.build_pool(doc, {"A": meta(), "B": meta(footprint="False")}, probes=())
        eligible = [r for r in records if v2.eligible(r)]
        self.assertEqual([r["members"] for r in eligible], [["A"]])

    def test_rank_is_hash_order_not_input_order(self):
        doc = groups_json([["A"], ["B"]])
        records = v2.build_pool(doc, {"A": meta(), "B": meta()}, probes=())
        reversed_records = v2.build_pool({"members": {"r0": ["B"], "r1": ["A"]},
                                          "zero_distance_overlapping": [],
                                          "min_intergroup_distance_km": 40.0},
                                         {"A": meta(), "B": meta()}, probes=())
        self.assertEqual([r["group_id"] for r in records],
                         [r["group_id"] for r in reversed_records])


class TestOverlapAwareSelect(unittest.TestCase):
    def records(self):
        doc = groups_json([["A", "B"], ["C"]])
        return v2.build_pool(doc, {"A": meta(), "B": meta(), "C": meta()}, probes=())

    def test_overlapping_scene_is_skipped(self):
        pairs = {("A", "B"): 20.0, ("B", "A"): 20.0}
        chosen = v2.select(self.records(), 2, 2, pairs)
        scenes = chosen["scenes"]
        self.assertEqual(len([s for s in scenes if s in ("A", "B")]), 1)
        self.assertTrue(chosen["skipped"])
        self.assertIn("overlaps", chosen["skipped"][0]["reason"])

    def test_non_overlapping_scenes_are_both_taken(self):
        chosen = v2.select(self.records(), 2, 2, {})       # both groups, two scenes allowed
        sizes = sorted(len(v) for v in chosen["scenes_per_group"].values())
        self.assertEqual(sizes, [1, 2])                    # the pair and the singleton
        pair = [v for v in chosen["scenes_per_group"].values() if len(v) == 2][0]
        self.assertEqual(sorted(pair), ["A", "B"])

    def test_cap_limits_scenes_per_group(self):
        chosen = v2.select(self.records(), 1, 1, {})
        self.assertEqual(len(chosen["scenes"]), 1)

    def test_eligibility_filters_before_the_cap(self):
        doc = groups_json([["A"], ["B"], ["C"]])
        records = v2.build_pool(doc, {"A": meta(footprint="False"), "B": meta(), "C": meta()},
                                probes=())
        chosen = v2.select(records, 2, 1, {})
        self.assertEqual(len(chosen["groups"]), 2)
        self.assertTrue(all(g in chosen["groups"] for g in chosen["groups"]))

    def test_selection_is_order_independent(self):
        a = v2.select(self.records(), 2, 2, {})
        b = v2.select(list(reversed(self.records())), 2, 2, {})
        self.assertEqual(a["scenes"], b["scenes"])
        self.assertEqual(a["groups"], b["groups"])
        self.assertEqual(a["scenes_per_group"], b["scenes_per_group"])


if __name__ == "__main__":
    unittest.main()
