#!/usr/bin/env python3
"""Offline tests for the heldout_v2 freeze (P18-05)."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("freeze_heldout", ROOT / "scripts" / "freeze_heldout.py")
fh = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fh
SPEC.loader.exec_module(fh)


def row(sample_id, product, group, split="test", lat=0.0, lon=0.0, zone="LTM_A"):
    return {
        "sample_id": sample_id,
        "NAC_EDR_ID": product,
        "group_id": group,
        "LTM_CODE": zone,
        "split_final": split,
        "center_lat": lat,
        "center_lon": lon,
        "bounds_xmin": 0.0,
        "bounds_ymin": 0.0,
        "footprint_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
        "image_path": f"nac/{sample_id}.nc",
        "dtm3m_path": f"dtm_3m/{sample_id}.nc",
        "slope3m_path": f"slope_3m/{sample_id}.nc",
        "has_slope3m": True,
    }


def used_frame():
    # 1 degree of latitude is ~30.3 km on the Moon.
    return pd.DataFrame([
        {"sample_id": "used_1", "NAC_EDR_ID": "P_USED", "group_id": "G_USED",
         "center_lat": 0.0, "center_lon": 0.0},
    ])


class TestSelection(unittest.TestCase):
    def test_excludes_used_products_groups_and_near_tiles(self):
        manifest = pd.DataFrame([
            row("a", "P_USED", "G_NEW", lat=1.0),       # reused product
            row("b", "P_NEW", "G_USED", lat=1.0),       # reused group
            row("c", "P_NEAR", "G_NEAR", lat=0.05),     # 1.5 km away -> inside buffer
            row("d", "P_OK1", "G_OK1", lat=1.0, zone="LTM_A"),
            row("e", "P_OK2", "G_OK2", lat=2.0, zone="LTM_B"),
            row("f", "P_TRAIN", "G_TRAIN", lat=3.0, split="train"),  # not in the test split
        ])
        selected, audit = fh.select_heldout(manifest, used_frame(), n=10, buffer_km=10.0, seed=7)
        self.assertEqual(set(selected["sample_id"]), {"d", "e"})
        self.assertGreaterEqual(float(selected["buffer_km_to_used"].min()), 10.0)
        self.assertEqual(audit["used_tiles"], 1)

    def test_product_cap_is_respected(self):
        manifest = pd.DataFrame([
            row("a1", "P_ONE", "G_A", lat=1.0, zone="LTM_A"),
            row("a2", "P_ONE", "G_A", lat=1.1, zone="LTM_A"),
            row("b1", "P_TWO", "G_B", lat=2.0, zone="LTM_B"),
        ])
        capped, _ = fh.select_heldout(manifest, used_frame(), n=10, buffer_km=10.0, seed=1,
                                      max_per_product=1)
        self.assertEqual(len(capped["NAC_EDR_ID"]), len(set(capped["NAC_EDR_ID"])))
        self.assertEqual(len(capped), 2)
        # Spacing also holds inside the frozen cohort itself.
        spaced, _ = fh.select_heldout(manifest, used_frame(), n=10, buffer_km=10.0, seed=1,
                                      max_per_product=5)
        self.assertLessEqual(len(spaced), 2)  # a1/a2 are ~3 km apart, one is rejected by spacing

    def test_respects_requested_count(self):
        manifest = pd.DataFrame([
            row(f"s{i}", f"P{i}", f"G{i}", lat=1.0 + i * 0.5, zone=f"LTM_{i % 3}")
            for i in range(20)
        ])
        selected, _ = fh.select_heldout(manifest, used_frame(), n=5, buffer_km=10.0, seed=3)
        self.assertEqual(len(selected), 5)

    def test_seed_is_deterministic(self):
        manifest = pd.DataFrame([
            row(f"s{i}", f"P{i}", f"G{i}", lat=1.0 + i * 0.1, zone=f"LTM_{i % 4}")
            for i in range(30)
        ])
        first, _ = fh.select_heldout(manifest, used_frame(), n=8, buffer_km=10.0, seed=11)
        second, _ = fh.select_heldout(manifest, used_frame(), n=8, buffer_km=10.0, seed=11)
        third, _ = fh.select_heldout(manifest, used_frame(), n=8, buffer_km=10.0, seed=12)
        self.assertEqual(list(first["sample_id"]), list(second["sample_id"]))
        self.assertNotEqual(list(first["sample_id"]), list(third["sample_id"]))

    def test_no_candidates_raises(self):
        manifest = pd.DataFrame([row("a", "P_USED", "G_NEW", lat=1.0)])
        with self.assertRaises(fh.FreezeError):
            fh.select_heldout(manifest, used_frame(), n=5, buffer_km=10.0, seed=1)

    def test_empty_used_set_is_refused(self):
        with self.assertRaises(fh.FreezeError):
            fh.select_heldout(pd.DataFrame([row("a", "P", "G", lat=1.0)]), pd.DataFrame(),
                              n=5, buffer_km=10.0, seed=1)

    def test_all_candidates_inside_buffer_raises(self):
        manifest = pd.DataFrame([row("a", "P_NEW", "G_NEW", lat=0.0, lon=0.02)])
        with self.assertRaises(fh.FreezeError):
            fh.select_heldout(manifest, used_frame(), n=5, buffer_km=10.0, seed=1)


class TestFreezeArtifacts(unittest.TestCase):
    def test_main_writes_frozen_manifest_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = pd.DataFrame([
                row(f"s{i}", f"P{i}", f"G{i}", lat=1.0 + i * 0.2, zone=f"LTM_{i % 3}")
                for i in range(40)
            ])
            manifest_path = root / "manifest.parquet"
            manifest.to_parquet(manifest_path, index=False)
            labels_path = root / "labels.csv"
            used_frame().to_csv(labels_path, index=False)
            code = fh.main(["--manifest", str(manifest_path), "--labels", str(labels_path),
                            "--out-dir", str(root / "protocols"), "--n", "10",
                            "--min-tiles", "5", "--buffer-km", "10", "--seed", "5"])
            self.assertEqual(code, 0)
            frozen = pd.read_csv(root / "protocols" / "heldout_v2_manifest.csv")
            protocol = json.loads((root / "protocols" / "heldout_v2_protocol.json").read_text())
            self.assertEqual(len(frozen), 10)
            self.assertEqual(protocol["status"], "FROZEN")
            self.assertEqual(protocol["counts"]["frozen_tiles"], 10)
            self.assertEqual(protocol["sources"]["heldout_manifest_sha256"],
                             fh.sha256_file(root / "protocols" / "heldout_v2_manifest.csv"))
            self.assertGreaterEqual(protocol["audit"]["min_buffer_km_selected"], 10.0)
            self.assertEqual(protocol["overlap_checks"]["hydrated_sample_ids"], 0)

    def test_insufficient_candidates_reports_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = pd.DataFrame([row("a", "P1", "G1", lat=1.0),
                                     row("b", "P2", "G2", lat=2.0)])
            manifest_path = root / "manifest.parquet"
            manifest.to_parquet(manifest_path, index=False)
            labels_path = root / "labels.csv"
            used_frame().to_csv(labels_path, index=False)
            code = fh.main(["--manifest", str(manifest_path), "--labels", str(labels_path),
                            "--out-dir", str(root / "protocols"), "--n", "5"])
            self.assertEqual(code, 2)
            self.assertFalse((root / "protocols" / "heldout_v2_manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
