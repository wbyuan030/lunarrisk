#!/usr/bin/env python3
"""Offline tests for the P20 single-linkage group audit."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_groups_p20", ROOT / "scripts" / "audit_groups_p20.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def row(name, nearest, km):
    return {"DTM_name": name, "nearest_candidate_DTM": nearest,
            "min_candidate_distance_km": "" if km is None else str(km)}


class TestUnionFind(unittest.TestCase):
    def test_transitive_merge(self):
        uf = audit.UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        self.assertEqual(uf.find("a"), uf.find("c"))
        self.assertNotEqual(uf.find("a"), uf.find("d"))

    def test_singletons_are_kept(self):
        uf = audit.UnionFind()
        uf.find("solo")
        self.assertEqual(uf.groups(), {"solo": ["solo"]})


class TestCluster(unittest.TestCase):
    def test_chain_collapses_into_one_group(self):
        # a-b 4 km, b-c 9 km, c-d 25 km -> {a,b,c} and {d}
        rows = [row("a", "b", 4.0), row("b", "c", 9.0), row("c", "d", 25.0), row("d", "c", 25.0)]
        groups = audit.cluster(rows, 10.0, ["a", "b", "c", "d"])
        self.assertEqual(sorted(groups.values()), [["a", "b", "c"], ["d"]])

    def test_threshold_is_exclusive(self):
        rows = [row("a", "b", 10.0), row("b", "a", 10.0)]
        groups = audit.cluster(rows, 10.0, ["a", "b"])
        self.assertEqual(len(groups), 2)

    def test_edges_to_filtered_scenes_do_not_merge(self):
        # "gone" is not in the scene list, so a must stay alone even though its nearest is gone
        rows = [row("a", "gone", 1.0), row("a2", "a", 1.0)]
        groups = audit.cluster(rows, 10.0, ["a", "a2"])
        self.assertEqual(sorted(groups.values()), [["a", "a2"]])

    def test_missing_distance_is_not_an_edge(self):
        rows = [row("a", "b", None), row("b", "a", None)]
        self.assertEqual(len(audit.cluster(rows, 10.0, ["a", "b"])), 2)

    def test_unlisted_scene_still_reported(self):
        rows = [row("a", "", None)]
        groups = audit.cluster(rows, 10.0, ["a", "ghost"])
        self.assertEqual(sorted(groups.values()), [["a"]])


if __name__ == "__main__":
    unittest.main()
