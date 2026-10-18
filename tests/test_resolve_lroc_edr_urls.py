#!/usr/bin/env python3
"""No-network tests for the LROC official observation-index EDR URL resolver."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
import shapefile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("resolve_lroc_edr_urls", ROOT / "scripts" / "resolve_lroc_edr_urls.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class TestIDs(unittest.TestCase):
    def test_requires_edr_camera_suffix(self):
        self.assertEqual(mod.parse_product_id("M139939938LE"), "M139939938LE")
        self.assertEqual(mod.parse_product_id("m139939938re"), "M139939938RE")
        self.assertIsNone(mod.parse_product_id("M139939938LC"))
        self.assertIsNone(mod.parse_product_id("M139939938"))

    def test_9_and_10_digits(self):
        self.assertEqual(mod.expected_products(["M139939938", "M1399399380"]),
                         {"M139939938LE", "M139939938RE", "M1399399380LE", "M1399399380RE"})
        with self.assertRaises(ValueError):
            mod.expected_products(["M139939938LE"])


class TestIndexes(unittest.TestCase):
    def test_zip_dbf_extract_and_match(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)
            stem = path / "NAC_EQ_NOMINAL_MISSION_360"
            with shapefile.Writer(str(stem), shapeType=shapefile.POINT) as w:
                w.field("PRODUCT_ID", "C", size=20)
                w.field("PDS_VOLUME", "C", size=30)
                w.field("FILE_SPECI", "C", size=100)
                w.field("URL", "C", size=250)
                for camera in ("LE", "RE"):
                    image_id = "M139939938" + camera
                    w.point(0, 0)
                    w.record(image_id, "LROLRC_0005",
                             "DATA/SCI/2010267/NAC/" + image_id + ".IMG",
                             "https://archive.example/" + image_id + ".IMG")
                w.point(0, 0)
                w.record("M000000000LE", "LROLRC_0001", "ignored", "https://ignore.invalid/a")
            z = path / "NAC_EQ_NOMINAL_MISSION_360.ZIP"
            with zipfile.ZipFile(z, "w") as archive:
                archive.write(stem.with_suffix(".dbf"), arcname=stem.with_suffix(".dbf").name)
            targets = mod.expected_products(["M139939938", "M139872173"])
            result, stats = mod.resolve([z], targets)
            self.assertEqual(stats["records_scanned"], 3)
            self.assertEqual(result["M139939938LE"]["status"], "UNVERIFIED_URL")
            self.assertEqual(result["M139939938RE"]["volume"], "LROLRC_0005")
            self.assertEqual(result["M139872173LE"]["status"], "NOT_IN_INDEX")
            self.assertIn("/NAC/", result["M139939938LE"]["file_spec"])

    def test_csv_conflicting_duplicate_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "index.csv"
            pd.DataFrame([
                {"PRODUCT_ID": "M139939938LE", "PDS_VOLUME": "V1", "FILE_SPECI": "A", "URL": "https://a/1.IMG"},
                {"PRODUCT_ID": "M139939938LE", "PDS_VOLUME": "V2", "FILE_SPECI": "B", "URL": "https://b/1.IMG"},
            ]).to_csv(path, index=False)
            result, _ = mod.resolve([path], {"M139939938LE"})
            self.assertEqual(result["M139939938LE"]["status"], "AMBIGUOUS_INDEX")

    def test_head_refuses_non_url(self):
        self.assertEqual(mod.verify_head("file:///tmp/x")[0], "NO_HTTP_URL")
        self.assertEqual(mod.verify_head("")[0], "NO_HTTP_URL")


if __name__ == "__main__":
    unittest.main()
