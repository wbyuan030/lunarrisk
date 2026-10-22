#!/usr/bin/env python3
"""Offline tests for the preregistered held-out selection (determinism is the whole point)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pool = load("make_heldout_pool_p20", "make_heldout_pool_p20.py")


def row(name, nearest="", km="500", probe=False, footprint="True", source="True"):
    return {"DTM_name": name, "nearest_candidate_DTM": nearest, "min_candidate_distance_km": km,
            "center_lat": "10.0", "center_lon": "20.0", "resolution_m": "2.0",
            "footprint_disjoint": footprint, "source_disjoint": source}


class TestGrouping(unittest.TestCase):
    def test_close_scenes_share_one_group(self):
        rows = [row("A", "B", "3"), row("B", "A", "3"), row("C")]
        availability = {"scenes_complete": ["A", "B", "C"]}
        records = pool.build(rows, availability, 10.0)
        self.assertEqual(len(records), 2)
        sizes = sorted(r["n_scenes"] for r in records)
        self.assertEqual(sizes, [1, 2])

    def test_empty_distance_is_tolerated(self):
        rows = [row("A", km=""), row("B", km="")]
        records = pool.build(rows, {"scenes_complete": ["A", "B"]}, 10.0, probes=())
        self.assertTrue(all(r["min_candidate_distance_km"] is None for r in records))

    def test_group_id_is_order_independent(self):
        self.assertEqual(pool.group_id(["A", "B"]), pool.group_id(["B", "A"]))

    def test_group_id_changes_when_membership_changes(self):
        self.assertNotEqual(pool.group_id(["A", "B"]), pool.group_id(["A", "B", "C"]))


class TestSelectionRules(unittest.TestCase):
    def setUp(self):
        self.rows = [row("A"), row("B"), row("PROBE1"), row("D", footprint="False")]
        self.availability = {"scenes_complete": ["A", "B", "PROBE1", "D"]}
        self.records = pool.build(self.rows, self.availability, 10.0, probes=("PROBE1",))

    def test_probe_groups_are_marked_and_ineligible(self):
        probe_groups = [r for r in self.records if r["contains_probe"]]
        self.assertEqual(len(probe_groups), 1)
        self.assertFalse(pool.eligible(probe_groups[0]))

    def test_footprint_conflict_is_ineligible(self):
        conflict = [r for r in self.records if r["members"] == ["D"]][0]
        self.assertFalse(pool.eligible(conflict))

    def test_clean_group_is_eligible(self):
        clean = [r for r in self.records if r["members"] == ["A"]][0]
        self.assertTrue(pool.eligible(clean))

    def test_rank_order_is_deterministic_and_unique(self):
        ranks = [r["selection_rank"] for r in self.records]
        self.assertEqual(ranks, list(range(1, len(self.records) + 1)))
        again = pool.build(self.rows, self.availability, 10.0, probes=("PROBE1",))
        self.assertEqual([r["group_id"] for r in self.records], [r["group_id"] for r in again])

    def test_rank_order_does_not_depend_on_input_order(self):
        shuffled = pool.build(list(reversed(self.rows)), self.availability, 10.0,
                              probes=("PROBE1",))
        self.assertEqual([r["group_id"] for r in self.records], [r["group_id"] for r in shuffled])


class TestSelect(unittest.TestCase):
    def records(self):
        rows = [row(n) for n in ("A", "B", "C", "D", "E")]
        return pool.build(rows, {"scenes_complete": ["A", "B", "C", "D", "E"]}, 10.0, probes=())

    def test_caps_scenes_per_group_and_keeps_order(self):
        groups = [dict(r) for r in self.records()]
        groups[0]["members"] = ["A1", "A2", "A3", "A4"]
        groups[0]["n_scenes"] = 4
        chosen = pool.select(groups, target_groups=2, max_scenes_per_group=2)
        self.assertEqual(len(chosen["groups"]), 2)
        # first group clipped to its cap, the second is a singleton, so 2 + 1
        self.assertEqual(len(chosen["scenes"]), 3)
        self.assertEqual(len(chosen["scenes_per_group"][chosen["groups"][0]]), 2)

    def test_selection_is_independent_of_input_order(self):
        a = pool.select(self.records(), 3, 2)
        b = pool.select(list(reversed(self.records())), 3, 2)
        self.assertEqual(a["scenes"], b["scenes"])

    def test_ineligible_groups_are_skipped(self):
        groups = [dict(r) for r in self.records()]
        groups[0]["contains_probe"] = ["X"]
        chosen = pool.select(groups, target_groups=2, max_scenes_per_group=1)
        self.assertNotIn(groups[0]["group_id"], chosen["groups"])


class TestCommittedProtocol(unittest.TestCase):
    """The committed protocol must be reproducible from the committed pool and the stated rule."""

    def setUp(self):
        import csv
        import json
        root = ROOT
        self.doc = json.loads((root / "protocols" / "heldout_v2_external_protocol.json")
                              .read_text(encoding="utf-8"))
        rows = list(csv.DictReader((root / "protocols" / "heldout_v2_group_pool.csv")
                                  .open(newline="", encoding="utf-8")))
        records = [{"group_id": r["group_id"], "members": r["members"].split(";"),
                    "footprint_disjoint_all": r["footprint_disjoint_all"] == "True",
                    "source_disjoint_all": r["source_disjoint_all"] == "True",
                    "contains_probe": [x for x in r["contains_probe"].split(";") if x],
                    "selection_rank": int(r["selection_rank"])}
                   for r in rows]
        records.sort(key=lambda r: r["selection_rank"])
        selection = self.doc["selection"]
        self.expected = pool.select(records, selection["target_groups"],
                                    selection["max_scenes_per_group"])

    def test_groups_match_the_rule(self):
        self.assertEqual(self.doc["selection"]["selected_group_ids"], self.expected["groups"])

    def test_scenes_match_the_rule(self):
        self.assertEqual(self.doc["selection"]["selected_scenes"], self.expected["scenes"])
        self.assertEqual(self.doc["selection"]["expected_scenes"], len(self.expected["scenes"]))

    def test_hashes_match(self):
        import hashlib
        self.assertEqual(self.doc["selection"]["selection_hash"],
                         hashlib.sha256("|".join(self.expected["groups"]).encode()).hexdigest())
        self.assertEqual(self.doc["selection"]["scene_hash"],
                         hashlib.sha256("|".join(self.expected["scenes"]).encode()).hexdigest())

    def test_still_pending_signoff(self):
        self.assertEqual(self.doc["status"], "PENDING_SIGNOFF")
        self.assertEqual(self.doc["gate3_status"], "NOT_FROZEN")
        self.assertEqual(self.doc["gate4_status"], "HOLD")


if __name__ == "__main__":
    unittest.main()
