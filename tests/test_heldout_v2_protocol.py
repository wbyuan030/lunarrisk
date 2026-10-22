#!/usr/bin/env python3
"""Invariants for the v2 protocol: status, reproducibility, and geometric independence."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_heldout_pool_v2",
                                              ROOT / "scripts" / "make_heldout_pool_v2.py")
v2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)

PROTOCOL = ROOT / "protocols" / "heldout_v2_external_protocol_v2.json"
POOL = ROOT / "protocols" / "heldout_v2_group_pool_v2.csv"
PAIRS = ROOT / "protocols" / "heldout_v2_group_pairs_v2.csv"


class TestStatus(unittest.TestCase):
    def test_not_frozen_and_not_approved(self):
        doc = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(doc["status"], "PENDING_SIGNOFF")
        self.assertEqual(doc["gate3_status"], "NOT_FROZEN")
        self.assertEqual(doc["gate4_status"], "HOLD")
        self.assertIn("nearest-neighbour", doc["reason_for_v2"])

    def test_v1_is_still_present_and_untouched(self):
        v1 = json.loads((ROOT / "protocols" / "heldout_v2_external_protocol.json")
                        .read_text(encoding="utf-8"))
        self.assertEqual(v1["version"], 1)
        self.assertEqual(v1["selection"]["expected_scenes"], 18)


class TestReproducibility(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        rows = list(csv.DictReader(POOL.open(newline="", encoding="utf-8")))
        self.records = [{"group_id": r["group_id"], "members": r["members"].split(";"),
                         "footprint_disjoint_all": r["footprint_disjoint_all"] == "True",
                         "source_disjoint_all": r["source_disjoint_all"] == "True",
                         "contains_probe": [x for x in r["contains_probe"].split(";") if x],
                         "selection_rank": int(r["selection_rank"])} for r in rows]
        self.pairs = {}
        for row in csv.DictReader(PAIRS.open(newline="", encoding="utf-8")):
            area = row["overlap_area_km2"]
            if area:
                self.pairs[(row["scene_a"], row["scene_b"])] = float(area)
                self.pairs[(row["scene_b"], row["scene_a"])] = float(area)
        sel = self.doc["selection"]
        self.expected = v2.select(self.records, sel["target_groups"],
                                  sel["max_scenes_per_group"], self.pairs)

    def test_selection_matches_the_rule(self):
        sel = self.doc["selection"]
        self.assertEqual(sel["selected_group_ids"], self.expected["groups"])
        self.assertEqual(sel["selected_scenes"], self.expected["scenes"])
        self.assertEqual(sel["selection_hash"],
                         hashlib.sha256("|".join(self.expected["groups"]).encode()).hexdigest())
        self.assertEqual(sel["scene_hash"],
                         hashlib.sha256("|".join(self.expected["scenes"]).encode()).hexdigest())

    def test_no_two_selected_scenes_in_one_group_overlap(self):
        sel = self.doc["selection"]
        for group, members in sel["scenes_per_group"].items():
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    self.assertLessEqual(self.pairs.get((a, b), 0.0), 1.0,
                                         f"{a} and {b} overlap inside {group}")


class TestIndependence(unittest.TestCase):
    def test_every_cross_group_pair_is_beyond_the_threshold(self):
        doc = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        rows = list(csv.DictReader(POOL.open(newline="", encoding="utf-8")))
        group_of = {}
        for row in rows:
            for member in row["members"].split(";"):
                group_of[member] = row["group_id"]
        with PAIRS.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                a, b = row["scene_a"], row["scene_b"]
                if group_of.get(a) != group_of.get(b):
                    self.assertGreaterEqual(float(row["polygon_distance_km"]),
                                            doc["grouping"]["threshold_km"],
                                            f"{a} and {b} are in different groups but too close")

    def test_pair_table_is_complete_for_the_pool(self):
        doc = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        rows = list(csv.DictReader(POOL.open(newline="", encoding="utf-8")))
        scenes = {s for row in rows for s in row["members"].split(";")}
        n = len(scenes)
        self.assertEqual(doc["grouping"]["pairs_evaluated"], n * (n - 1) // 2)

    def test_sparim_products_share_one_group(self):
        rows = list(csv.DictReader(POOL.open(newline="", encoding="utf-8")))
        sparim = [row["group_id"] for row in rows
                  if "SPARIM1" in row["members"].split(";")]
        self.assertEqual(len(sparim), 1)
        members = [row["members"] for row in rows if row["group_id"] == sparim[0]][0].split(";")
        self.assertIn("SPARIM2", members)


if __name__ == "__main__":
    unittest.main()
