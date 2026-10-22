#!/usr/bin/env python3
"""Offline tests for the pairwise 10 km grouping (the bug this script exists to fix)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_groups_pairwise",
                                              ROOT / "scripts" / "audit_groups_pairwise.py")
pw = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pw
spec.loader.exec_module(pw)

KM_PER_DEG_LAT = 30.3234          # 2*pi*R/360 at R = 1737.4 km


def east_deg(km, lat):
    """Longitude degrees that correspond to ``km`` of ground distance at latitude ``lat``."""
    import math
    return km / (KM_PER_DEG_LAT * math.cos(math.radians(lat)))


def square(lon, lat, size_km=1.0):
    """Small axis-aligned square polygon in lon/lat degrees."""
    half = size_km / 2 / KM_PER_DEG_LAT
    ring = [(lon - half, lat - half), (lon + half, lat - half),
            (lon + half, lat + half), (lon - half, lat + half),
            (lon - half, lat - half)]
    return [ring]


class TestDistance(unittest.TestCase):
    def test_identical_polygons_are_zero_apart(self):
        self.assertAlmostEqual(pw.distance_km(square(10, 5), square(10, 5)), 0.0, places=6)

    def test_offset_polygons_report_the_gap_not_the_centre_distance(self):
        gap_km = 3.0
        distance = pw.distance_km(square(10, 5), square(10 + east_deg(1.0 + gap_km, 5), 5))
        self.assertAlmostEqual(distance, gap_km, delta=0.2)

    def test_antimeridian_pair_is_close(self):
        # 0.5 deg either side of 180: centre distance via naive lon arithmetic would be ~half the
        # globe, the real gap is about 1 deg = 30 km
        left = square(179.5, 0.0)
        right = square(-179.5, 0.0)
        self.assertLess(pw.distance_km(left, right), 35.0)

    def test_polar_pair_is_close(self):
        # two points at lat 89.9 with opposite longitudes are 2 x 0.1 deg = 6.06 km apart, so the
        # polygon gap is that minus the two half-sizes -- naive lon arithmetic would say ~half a globe
        a = square(0.0, 89.9)
        b = square(180.0, 89.9)
        self.assertAlmostEqual(pw.distance_km(a, b), 6.06 - 1.0, delta=0.3)

    def test_multipart_takes_the_nearest_part(self):
        near = square(10.0, 5.0, size_km=1.0)
        far = square(40.0, 5.0, size_km=1.0)
        target = square(10 + east_deg(3.0, 5.0), 5.0, size_km=1.0)
        distance = pw.distance_km(near + far, target)
        self.assertAlmostEqual(distance, 2.0, delta=0.3)      # 3 km centre gap minus two halves

    def test_shift_lon_wraps_into_the_local_frame(self):
        shifted = pw.shift_lon([[(181.0, 0.0), (182.0, 0.0), (182.0, 1.0)]], lon0=-179.0)
        self.assertTrue(all(-181.0 <= lon <= 181.0 for lon, _ in shifted[0]))
        self.assertTrue(any(lon < 0 for lon, _ in shifted[0]))


class TestGrouping(unittest.TestCase):
    def test_bridging_scene_merges_even_when_nobody_points_at_it(self):
        # centres 9 km apart: A-B 8 km gap, B-C 8 km, A-C 17 km. Only the two short edges exist, so
        # A and C merge solely because B bridges them -- exactly what a nearest-neighbour-only union
        # can miss when B's nearest neighbour is not A.
        polygons = {"A": square(10.0, 5.0, size_km=1.0),
                    "B": square(10 + east_deg(9, 5), 5.0, size_km=1.0),
                    "C": square(10 + east_deg(18, 5), 5.0, size_km=1.0)}
        _edges, close = pw.build_edges(polygons, 10.0)
        self.assertEqual(len(close), 2)                        # A-B and B-C, not A-C
        groups = pw.group_from_edges(sorted(polygons), close)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(next(iter(groups.values()))), ["A", "B", "C"])

    def test_scenes_beyond_the_threshold_stay_separate(self):
        polygons = {"A": square(10.0, 5.0), "B": square(10 + east_deg(25, 5), 5.0)}
        _edges, close = pw.build_edges(polygons, 10.0)
        self.assertEqual(len(pw.group_from_edges(sorted(polygons), close)), 2)

    def test_sparim_like_zero_distance_pair_is_one_group(self):
        # the real pool had SPARIM1/SPARIM2 with identical centres and 0 km nearest distance, yet
        # they were split; with real polygons they must merge
        polygons = {"SPARIM1": square(171.0, -51.0, size_km=30.0),
                    "SPARIM2": square(171.0, -51.0, size_km=30.0)}
        _edges, close = pw.build_edges(polygons, 10.0)
        groups = pw.group_from_edges(sorted(polygons), close)
        self.assertEqual(len(groups), 1)

    def test_overlapping_pair_has_area_while_touching_pair_does_not(self):
        a = square(10.0, 5.0, size_km=10.0)
        overlap = square(10.0, 5.0, size_km=10.0)
        touching = square(10.0, 5.0 + 10 / KM_PER_DEG_LAT, size_km=10.0)   # shares an edge
        self.assertGreater(pw.overlap_km2(a, overlap), 50.0)
        self.assertLess(pw.overlap_km2(a, touching), 2.0)


class TestIntergroup(unittest.TestCase):
    def test_min_intergroup_is_the_smallest_cross_group_distance(self):
        groups = {"r1": ["A", "B"], "r2": ["C"]}
        distances = {("A", "B"): 1.0, ("A", "C"): 12.0, ("B", "C"): 7.5}
        result = pw.min_intergroup_km(groups, distances)
        self.assertAlmostEqual(result["A|C"], 7.5)


if __name__ == "__main__":
    unittest.main()
