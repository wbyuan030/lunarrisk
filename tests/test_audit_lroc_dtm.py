#!/usr/bin/env python3
"""Offline regression tests for provisional LROC DTM source and footprint screening."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_lroc_dtm", ROOT / "scripts" / "audit_lroc_dtm.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


class TestSourceIDs(unittest.TestCase):
    def test_empty_ids_are_not_auditable(self):
        for raw in ("", None, "None", "no source IDs"):
            self.assertFalse(audit.source_id_metadata_valid(raw))
            self.assertEqual(audit.normalise_products(raw), [])

    def test_short_or_long_numeric_ids_are_rejected(self):
        for raw in ("M123", "M123456789012", "M1234567890 M123"):
            self.assertFalse(audit.source_id_metadata_valid(raw))

    def test_legitimate_sources_with_camera_suffix(self):
        raw = "M1136306535L, M1136330937R"
        self.assertTrue(audit.source_id_metadata_valid(raw))
        self.assertEqual(audit.normalise_products(raw),
                         ["M1136306535", "M1136330937"])


class TestPolygonScreen(unittest.TestCase):
    def test_thin_footprint_is_not_approximated_by_equal_area_radius(self):
        # A thin 20-degree-long footprint has a small area but very distant
        # endpoint vertices, which the old sqrt(area/pi) method could miss.
        shape = [(-10.0, -0.01), (-10.0, 0.01),
                 (10.0, 0.01), (10.0, -0.01), (-10.0, -0.01)]
        radius = audit.vertex_outer_radius_km(0.0, 0.0, shape)
        self.assertIsNotNone(radius)
        self.assertGreater(radius, 200.0)

    def test_longitude_wraparound_uses_moon_centered_vectors(self):
        shape = [(179.9, 0), (180.1, 0), (180.1, 0.1),
                 (179.9, 0.1), (179.9, 0)]
        radius = audit.vertex_outer_radius_km(0.0, 180.0, shape)
        self.assertIsNotNone(radius)
        self.assertLess(radius, 10.0)

    def test_missing_or_invalid_polygon_never_passes(self):
        self.assertIsNone(audit.vertex_outer_radius_km(0, 0, []))
        self.assertIsNone(audit.vertex_outer_radius_km(0, 0, [(0, 0), (1, 0)]))
        self.assertIsNone(audit.vertex_outer_radius_km(0, 0, [(0, 0), (1, 0), (2, 95)]))
        self.assertIsNone(audit.vertex_outer_radius_km(float("nan"), 0, [(0, 0), (1, 0), (1, 1)]))


class TestEndToEndVerdict(unittest.TestCase):
    def test_metadata_screen_never_claims_strict_ood(self):
        import shapefile  # pyshp, present in requirements-extra.txt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "nac_dtms"
            with shapefile.Writer(str(base), shapeType=shapefile.POLYGON) as writer:
                writer.field("images", "C", size=100)
                writer.field("DTM_NAME", "C", size=40)
                writer.field("center_lon", "F", size=15, decimal=5)
                writer.field("center_lat", "F", size=15, decimal=5)
                writer.field("cov_sqkm", "F", size=15, decimal=5)
                for name, source, lon in [
                    ("candidate", "M1136306535L M1136330937R", 30.0),
                    ("empty_source", "", 40.0),
                ]:
                    writer.poly([[(lon - .1, -.1), (lon + .1, -.1),
                                  (lon + .1, .1), (lon - .1, .1), (lon - .1, -.1)]])
                    writer.record(images=source, DTM_NAME=name, center_lon=lon,
                                  center_lat=0.0, cov_sqkm=0.1)

            manifest = root / "manifest.parquet"
            pd.DataFrame({"NAC_EDR_ID": ["M9999999999"]}).to_parquet(manifest, index=False)
            labels = root / "labels.csv"
            pd.DataFrame({"center_lat": [0.0], "center_lon": [0.0]}).to_csv(labels, index=False)
            target = root / "out.json"
            self.assertEqual(audit.main([
                "--shapefile", str(base) + ".shp", "--manifest", str(manifest),
                "--labels", str(labels), "--out", str(target),
                "--buffer-km", "10"]), 0)
            payload = json.loads(target.read_text())
            self.assertEqual(payload["dtm_coverage"]["all_products_unseen"], 1)
            self.assertEqual(payload["dtm_coverage"]["empty_or_malformed_source_records"], 1)
            self.assertEqual(payload["spatial_isolation"]["provisional_vertex_screen_clear"], 1)
            self.assertIsNone(payload["strict_candidates_after_spatial"])
            self.assertFalse(payload["spatial_isolation"]["footprint_verified"])
            self.assertEqual(payload["verdict"], "CANDIDATES_REQUIRE_POLYGON_AND_SOURCE_QA")


if __name__ == "__main__":
    unittest.main()
