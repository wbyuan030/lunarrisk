#!/usr/bin/env python3
"""Offline regression tests for the P19 LOC DTM footprint / source audit."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pyproj import CRS
from shapely.geometry import Point, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "audit_lroc_footprints", ROOT / "scripts" / "audit_lroc_footprints.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)

LUNAR_PRJ = ('GEOGCS["GCS_Moon_2000",DATUM["D_Moon_2000",'
             'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],PRIMEM["Greenwich",0.0],'
             'UNIT["Degree",0.0174532925199433]]')


def square(points):
    return audit.dtm_polygon(points, [0])


class TestLongitudes(unittest.TestCase):
    def test_normalisation_of_both_conventions(self):
        got = audit.normalise_lon([359.9, 0.0, -170.0, 180.0, 303.74])
        for value, expected in zip(got, [-0.1, 0.0, -170.0, -180.0, -56.26]):
            self.assertAlmostEqual(value, expected, places=6)

    def test_unwrap_keeps_vertices_contiguous(self):
        got = audit.unwrap_around([179.95, -179.95], 179.9)
        self.assertAlmostEqual(got[0], 179.95, places=6)
        self.assertAlmostEqual(got[1], 180.05, places=6)


class TestPolygons(unittest.TestCase):
    def test_valid_square(self):
        geometry, status = square([(10.0, 20.0), (10.1, 20.0), (10.1, 20.1), (10.0, 20.1)])
        self.assertEqual(status, "OK")
        self.assertAlmostEqual(geometry.area, 0.01, places=9)

    def test_short_and_degenerate_rings_are_rejected(self):
        self.assertEqual(square([(0.0, 0.0), (1.0, 1.0)])[1], "RING_TOO_SHORT")
        self.assertEqual(square([(1.0, 1.0)] * 4)[1], "DEGENERATE_RING")
        self.assertEqual(square([])[1], "EMPTY_GEOMETRY")

    def test_impossible_latitude_is_rejected(self):
        self.assertEqual(square([(0.0, 0.0), (1.0, 0.0), (1.0, 95.0), (0.0, 95.0)])[1],
                         "INVALID_COORDINATE")

    def test_antimeridian_polygon_stays_local(self):
        geometry, status = square([(179.99, -0.01), (180.01, -0.01), (180.01, 0.01),
                                   (179.99, 0.01)])
        self.assertEqual(status, "OK")
        # unwrapped around the circular mean (180) the box is ~0.02 deg wide, not a whole globe
        self.assertLess(geometry.bounds[2] - geometry.bounds[0], 0.05)

    def test_inner_rings_are_filled_conservatively(self):
        outer = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        hole = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]
        geometry, status = audit.dtm_polygon(outer + hole, [0, 4])
        self.assertEqual(status, "OK")
        self.assertAlmostEqual(geometry.area, 1.0, places=9)
        self.assertTrue(geometry.contains(Point(0.5, 0.5)))

    def test_self_intersecting_shell_is_reported(self):
        _, status = square([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)])
        self.assertTrue(status.startswith("INVALID_POLYGON"), status)

    def test_moon_2000_prj_is_accepted_and_other_bodies_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.prj"
            good.write_text(LUNAR_PRJ, encoding="utf-8")
            result = audit.crs_check(good)
            self.assertTrue(result["valid"], result)
            self.assertTrue(result["is_geographic"])
            bad = Path(tmp) / "bad.prj"
            bad.write_text(CRS.from_epsg(4326).to_wkt(), encoding="utf-8")
            self.assertFalse(audit.crs_check(bad)["valid"])
            self.assertFalse(audit.crs_check(Path(tmp) / "missing.prj")["valid"])


class TestDistances(unittest.TestCase):
    def test_separated_squares_distance_matches_arc_length(self):
        target = box(0.0, 0.0, 0.01, 0.01)
        other = box(0.06, 0.0, 0.07, 0.01)
        distance, nearest = audit.distance_km(target, [("tile", other)], 0.0, 0.0)
        expected = (0.06 - 0.01) * 1737400.0 * 3.141592653589793 / 180.0 / 1000.0
        self.assertAlmostEqual(distance, expected, delta=0.02)
        self.assertEqual(nearest, "tile")

    def test_overlapping_polygons_have_zero_distance(self):
        target = box(0.0, 0.0, 0.02, 0.02)
        distance, _ = audit.distance_km(target, [("tile", box(0.01, 0.01, 0.03, 0.03))],
                                        0.0, 0.0)
        self.assertEqual(distance, 0.0)

    def test_distance_across_the_antimeridian_is_short(self):
        target = Polygon([(179.995, 0.0), (180.005, 0.0), (180.005, 0.02), (179.995, 0.02)])
        tile = box(-180.0, 0.0, -179.995, 0.02)
        distance, _ = audit.distance_km(target, [("tile", tile)], 0.0, 180.0)
        self.assertLess(distance, 2.0)

    def test_polar_separation_is_not_inflated(self):
        target = box(0.0, 84.9, 0.5, 85.0)
        tile = box(0.0, 84.8, 0.5, 84.85)
        distance, _ = audit.distance_km(target, [("tile", tile)], 84.95, 0.25)
        self.assertAlmostEqual(distance, 1.516, delta=0.05)

    def test_no_tiles_gives_infinite_distance(self):
        distance, nearest = audit.distance_km(box(0.0, 0.0, 0.01, 0.01), [], 0.0, 0.0)
        self.assertEqual(distance, float("inf"))
        self.assertEqual(nearest, "")

    def test_tile_footprints_use_geographic_corners_and_split_antimeridian(self):
        manifest = pd.DataFrame([
            {"sample_id": "A", "upper_left_lon": 10.0, "upper_left_lat": 1.0,
             "lower_right_lon": 10.01, "lower_right_lat": 0.99},
            {"sample_id": "B", "upper_left_lon": 179.99, "upper_left_lat": 1.0,
             "lower_right_lon": -179.99, "lower_right_lat": 0.99},
        ])
        footprints = dict(audit.tile_footprints(manifest, {"A", "B"}))
        self.assertAlmostEqual(footprints["A"].area, 0.0001, places=9)
        self.assertTrue(footprints["B"].intersects(box(179.995, 0.99, 180.0, 1.0)))
        self.assertTrue(footprints["B"].intersects(box(-180.0, 0.99, -179.995, 1.0)))
        self.assertFalse(footprints["B"].intersects(box(0.0, 0.99, 0.01, 1.0)))


class TestCandidateSpacing(unittest.TestCase):
    def test_nearest_neighbour_is_a_polygon_distance(self):
        first = box(0.0, 0.0, 0.01, 0.01)
        second = box(0.06, 0.0, 0.07, 0.01)
        far = box(40.0, 0.0, 40.01, 0.01)
        entries = [("A", first, (0.005, 0.005)), ("B", second, (0.005, 0.065)),
                   ("C", far, (0.005, 40.005))]
        nearest = audit.candidate_spacing(entries)
        self.assertEqual(nearest["A"][1], "B")
        self.assertAlmostEqual(nearest["A"][0], 1.516, delta=0.05)
        self.assertEqual(nearest["B"][1], "A")
        self.assertEqual(nearest["C"], (float("inf"), ""))

    def test_candidate_groups_merge_only_close_candidates(self):
        first = box(0.0, 0.0, 0.01, 0.01)
        second = box(0.02, 0.0, 0.03, 0.01)
        far = box(1.0, 0.0, 1.01, 0.01)
        entries = [("A", first, (0.005, 0.005)), ("B", second, (0.005, 0.025)),
                   ("C", far, (0.005, 1.005))]
        structure = audit.candidate_structure(entries)
        self.assertEqual(structure["groups"]["A"], structure["groups"]["B"])
        self.assertNotEqual(structure["groups"]["A"], structure["groups"]["C"])
        self.assertEqual(structure["group_count"], 2)
        self.assertEqual(structure["largest_group"], 2)

    def test_pair_distance_across_the_antimeridian(self):
        first = Polygon([(179.99, 0.0), (180.0, 0.0), (180.0, 0.02), (179.99, 0.02)])
        second = Polygon([(-180.0, 0.0), (-179.99, 0.0), (-179.99, 0.02), (-180.0, 0.02)])
        distance = audit.pair_distance_km(first, second, (0.01, 179.995), (0.01, -179.995))
        self.assertAlmostEqual(distance, 0.0, places=6)


class TestReadmeParsing(unittest.TestCase):
    def test_source_block_is_parsed_and_deduplicated(self):
        text = ("Title\n\nLROC NAC SOURCE IMAGES\nM1503229434 L/R\nM1503250470 L/R\n"
                "\nNext section\nM9999999999\n")
        result = audit.parse_readme_sources(text)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["sources"], ["M1503229434", "M1503250470"])

    def test_camera_suffixes_are_ignored_but_malformed_ids_are_not(self):
        result = audit.parse_readme_sources("LROC NAC SOURCE IMAGES\nM1136306535L M1136330937R\n")
        self.assertEqual(result["sources"], ["M1136306535", "M1136330937"])
        result = audit.parse_readme_sources("LROC NAC SOURCE IMAGES\nM12345\n")
        self.assertEqual(result["status"], "MALFORMED_TOKEN")

    def test_missing_and_empty_sections_are_fail_closed(self):
        self.assertEqual(audit.parse_readme_sources("nothing here")["status"], "NO_SOURCE_SECTION")
        self.assertEqual(audit.parse_readme_sources("LROC NAC SOURCE IMAGES\n\nmore text")["status"],
                         "EMPTY_SECTION")


class TestVerdict(unittest.TestCase):
    def test_pass_requires_every_check(self):
        self.assertEqual(audit.audit_reasons("OK", True, "", "OK", [], True), [])
        reasons = audit.audit_reasons("INVALID_POLYGON", False, "bad crs", "OUTSIDE_SECTION",
                                      ["M1"], False)
        self.assertEqual(reasons, ["GEOMETRY:INVALID_POLYGON", "CRS:bad crs",
                                   "SOURCE_README:OUTSIDE_SECTION",
                                   "SOURCE_OVERLAP_WITH_SOMBENCH", "FOOTPRINT_NOT_DISJOINT"])


class TestEndToEnd(unittest.TestCase):
    def _build_inputs(self, tmp: str):
        import shapefile

        base = Path(tmp)
        (base / "dtms.prj").write_text(LUNAR_PRJ, encoding="utf-8")
        writer = shapefile.Writer(str(base / "dtms"), shapeType=shapefile.POLYGON)
        writer.field("DTM_NAME", "C", size=24)
        writer.field("images", "C", size=60)
        writer.field("center_lat", "N", size=12, decimal=6)
        writer.field("center_lon", "N", size=12, decimal=6)
        writer.field("resolution", "N", size=8, decimal=1)
        writer.field("cov_sqkm", "N", size=12, decimal=2)
        # candidate far from every used tile
        writer.poly([[(100.0, 10.0), (100.2, 10.0), (100.2, 10.2), (100.0, 10.2), (100.0, 10.0)]])
        writer.record("FAR_DTM", "M1503229434, M1503250470", 10.1, 100.1, 5.0, 100.0)
        # overlapping product: one of its sources is already in SomBench
        writer.poly([[(20.0, 0.0), (20.2, 0.0), (20.2, 0.2), (20.0, 0.2), (20.0, 0.0)]])
        writer.record("SEEN_DTM", "M111856750", 0.1, 20.1, 5.0, 100.0)
        # malformed source list
        writer.poly([[(30.0, 0.0), (30.2, 0.0), (30.2, 0.2), (30.0, 0.2), (30.0, 0.0)]])
        writer.record("BAD_DTM", "M12", 0.1, 30.1, 5.0, 100.0)
        # candidate whose footprint touches a used tile
        writer.poly([[(40.0, 0.0), (40.2, 0.0), (40.2, 0.2), (40.0, 0.2), (40.0, 0.0)]])
        writer.record("NEAR_DTM", "M1503229439", 0.1, 40.1, 5.0, 100.0)
        writer.close()

        manifest = pd.DataFrame([
            {"sample_id": "T1", "NAC_EDR_ID": "M111856750",
             "upper_left_lon": 20.0, "upper_left_lat": 0.2,
             "lower_right_lon": 20.2, "lower_right_lat": 0.0},
            {"sample_id": "T2", "NAC_EDR_ID": "M199999999",
             "upper_left_lon": 40.0, "upper_left_lat": 0.2,
             "lower_right_lon": 40.2, "lower_right_lat": 0.0},
        ])
        manifest.to_parquet(base / "manifest.parquet", index=False)
        pd.DataFrame({"sample_id": ["T1", "T2"]}).to_csv(base / "labels.csv", index=False)
        return base

    def test_audit_selects_only_independent_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._build_inputs(tmp)
            code = audit.main([
                "--shapefile", str(base / "dtms.shp"),
                "--manifest", str(base / "manifest.parquet"),
                "--labels", str(base / "labels.csv"),
                "--protected", str(base / "absent.csv"),
                "--readme-cache", str(base / "readmes"),
                "--no-readme",
                "--out-csv", str(base / "audit.csv"),
                "--out-json", str(base / "audit.json"),
            ])
            self.assertEqual(code, 0)
            frame = pd.read_csv(base / "audit.csv").set_index("DTM_name")
            self.assertEqual(set(frame.index), {"FAR_DTM", "NEAR_DTM"})
            self.assertTrue(frame.loc["FAR_DTM", "footprint_disjoint"])
            self.assertFalse(frame.loc["NEAR_DTM", "footprint_disjoint"])
            self.assertAlmostEqual(frame.loc["NEAR_DTM", "min_polygon_distance_km"], 0.0, places=6)
            self.assertFalse(frame.loc["FAR_DTM", "source_disjoint"])  # README skipped
            self.assertIn("min_candidate_distance_km", frame.columns)
            self.assertTrue(bool(frame.loc["FAR_DTM", "candidate_spacing_disjoint"]))
            self.assertEqual(frame.loc["FAR_DTM", "n_parts"], 1)
            report = json.loads((base / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["candidates_screened"], 2)
            self.assertEqual(report["pass_both"], 0)
            self.assertTrue(report["crs_check"]["valid"])
            self.assertEqual(report["verdict"], "BLOCKED_OR_UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
