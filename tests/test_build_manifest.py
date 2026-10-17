#!/usr/bin/env python3
"""Tests for the P1-01 manifest grouping/split logic (synthetic inputs only)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_manifest", ROOT / "scripts" / "build_manifest.py")
bm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bm
SPEC.loader.exec_module(bm)


def frame(rows):
    base = {
        "PRODUCT_ID": "P",
        "ROW": 0,
        "COL": 0,
        "LTM_CODE": "LTM_1N",
        "DATASET": "train",
        "BOUNDS_XMIN": 0.0,
        "BOUNDS_XMAX": 512.0,
        "BOUNDS_YMIN": 0.0,
        "BOUNDS_YMAX": 512.0,
        "DTM_3M_TILE": "dtm/a.nc",
        "SLOPE_3M_TILE": "slope/a.nc",
        "DTM_60M_TILE": "dtm60/a.nc",
        "METADATA_TILE": "meta/a.txt",
        "NAC_TILE": "nac/a.nc",
        "DTM_3M_FRACTION_NULL": 0.0,
        "SLOPE_3M_FRACTION_NULL": 0.0,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


class TestGrouping(unittest.TestCase):
    def test_blocks_are_zone_aware(self):
        df = frame([
            {"PRODUCT_ID": "A", "LTM_CODE": "LTM_1N", "BOUNDS_XMIN": 10.0},
            {"PRODUCT_ID": "B", "LTM_CODE": "LTM_2N", "BOUNDS_XMIN": 10.0},
        ])
        blocks = bm.assign_blocks(df, 5000)
        self.assertNotEqual(blocks.iloc[0], blocks.iloc[1])

    def test_products_sharing_a_block_share_a_group(self):
        df = frame([
            {"PRODUCT_ID": "A"},
            {"PRODUCT_ID": "B"},
            {"PRODUCT_ID": "C", "BOUNDS_XMIN": 6000.0},
        ])
        _, groups = bm.build_groups(df, 5000)
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertNotEqual(groups.iloc[0], groups.iloc[2])

    def test_product_spanning_blocks_connects_them(self):
        df = frame([
            {"PRODUCT_ID": "P"},
            {"PRODUCT_ID": "P", "BOUNDS_XMIN": 6000.0},
            {"PRODUCT_ID": "Q", "BOUNDS_XMIN": 6000.0},
            {"PRODUCT_ID": "R"},
        ])
        _, groups = bm.build_groups(df, 5000)
        self.assertEqual(len(set(groups)), 1)

    def test_clean_split_keeps_group_whole(self):
        df = frame([
            {"PRODUCT_ID": "A", "DATASET": "train"},
            {"PRODUCT_ID": "B", "DATASET": "val"},
            {"PRODUCT_ID": "C", "DATASET": "test", "BOUNDS_XMIN": 10000.0},
        ])
        df["block_5km"], df["group_id"] = bm.build_groups(df, 5000)
        cleaned, reassigned = bm.clean_split(df)
        self.assertEqual(len(reassigned), 1)
        self.assertEqual(cleaned.iloc[0], cleaned.iloc[1])
        self.assertNotEqual(cleaned.iloc[0], cleaned.iloc[2])

    def test_clean_split_has_no_group_overlap(self):
        df = frame([
            {"PRODUCT_ID": "A", "DATASET": "train"},
            {"PRODUCT_ID": "B", "DATASET": "val"},
            {"PRODUCT_ID": "C", "DATASET": "test", "BOUNDS_XMIN": 10000.0},
        ])
        df["block_5km"], df["group_id"] = bm.build_groups(df, 5000)
        df["split_group_clean"], _ = bm.clean_split(df)
        audit = bm.split_audit(df, "split_group_clean")
        for overlap in audit["overlap"].values():
            self.assertEqual(overlap["groups"], 0)
            self.assertEqual(overlap["products"], 0)
            self.assertEqual(overlap["blocks_5km"], 0)

    def test_footprint_wkt(self):
        row = frame([{}]).iloc[0]
        wkt = bm.footprint_wkt(row)
        self.assertTrue(wkt.startswith("POLYGON (("))
        self.assertIn("0.000 0.000", wkt)
        self.assertIn("512.000 512.000", wkt)


class TestManifest(unittest.TestCase):
    def test_manifest_columns_and_flags(self):
        df = frame([
            {"PRODUCT_ID": "A", "DATASET": "train"},
            {"PRODUCT_ID": "B", "DATASET": "train", "SLOPE_3M_TILE": None},
        ])
        verification = {("A", 0, 0): {
            "assets": {"NAC_TILE": {"relative": "nac/a.nc", "sha256": "abc", "bytes": 1}},
            "slope_finite_fraction": 1.0,
            "bounds_aligned": True,
            "units_ok": True,
        }}
        manifest = bm.build_manifest(df, verification, "CRS_WKT", "index_sha", 5000)
        self.assertEqual(len(manifest), 2)
        self.assertEqual(int(manifest["has_slope3m"].sum()), 1)
        row = manifest[manifest["NAC_EDR_ID"] == "A"].iloc[0]
        self.assertEqual(row["nac_sha256"], "abc")
        self.assertEqual(row["valid_fraction"], 1.0)
        self.assertEqual(row["co_registration_status"], "SAMPLED_VERIFIED")
        other = manifest[manifest["NAC_EDR_ID"] == "B"].iloc[0]
        self.assertIsNone(other["nac_sha256"])
        self.assertEqual(other["co_registration_status"], "INDEX_BOUNDS_ONLY")
        self.assertEqual(manifest["geo_join_status"].unique().tolist(), ["GEO_JOIN_PENDING"])


if __name__ == "__main__":
    unittest.main()
