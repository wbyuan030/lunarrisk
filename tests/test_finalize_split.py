#!/usr/bin/env python3
"""Tests for P1-02 split finalization (synthetic inputs only)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("finalize_split", ROOT / "scripts" / "finalize_split.py")
fs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fs
SPEC.loader.exec_module(fs)


class TestCalibration(unittest.TestCase):
    def test_deterministic_and_group_level(self):
        groups = pd.Series(["G_a", "G_b", "G_c", "G_d", "G_e"])
        first = fs.cal_assignment(groups, seed=42)
        second = fs.cal_assignment(groups, seed=42)
        self.assertEqual(first, second)
        self.assertTrue(first.issubset(set(groups)))

    def test_different_seed_can_change_selection(self):
        groups = pd.Series([f"G_{i}" for i in range(100)])
        self.assertNotEqual(fs.cal_assignment(groups, seed=1), fs.cal_assignment(groups, seed=2))


class TestBuffer(unittest.TestCase):
    def frame(self, rows):
        return pd.DataFrame(rows)

    def test_train_near_test_is_dropped_test_is_kept(self):
        frame = self.frame([
            {"split_pre_buffer": "train", "nearest_other_split": "test", "nearest_other_split_km": 5.0},
            {"split_pre_buffer": "test", "nearest_other_split": "train", "nearest_other_split_km": 5.0},
        ])
        final = fs.apply_buffer(frame, 10.0)
        self.assertEqual(final.tolist(), ["drop_buffer", "test"])

    def test_cal_and_val_rules(self):
        frame = self.frame([
            {"split_pre_buffer": "cal", "nearest_other_split": "val", "nearest_other_split_km": 3.0},
            {"split_pre_buffer": "val", "nearest_other_split": "test", "nearest_other_split_km": 3.0},
            {"split_pre_buffer": "val", "nearest_other_split": "train", "nearest_other_split_km": 3.0},
            {"split_pre_buffer": "train", "nearest_other_split": "val", "nearest_other_split_km": 20.0},
            {"split_pre_buffer": "train", "nearest_other_split": "cal", "nearest_other_split_km": 3.0},
        ])
        final = fs.apply_buffer(frame, 10.0)
        self.assertEqual(final.tolist(), ["drop_buffer", "drop_buffer", "val", "train", "drop_buffer"])

    def test_sphere_distance(self):
        points = fs.unit_sphere(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        distance = np.linalg.norm(points[0] - points[1])
        expected = 2 * fs.LUNAR_RADIUS_KM * np.sin(np.radians(1.0) / 2)
        self.assertAlmostEqual(float(distance), float(expected), places=6)


class TestAudit(unittest.TestCase):
    def test_group_disjoint_final_split(self):
        frame = pd.DataFrame([
            {"split_final": "train", "group_id": "G1", "NAC_EDR_ID": "A", "block_5km": "b1",
             "has_slope3m": True, "LTM_CODE": "L1"},
            {"split_final": "val", "group_id": "G2", "NAC_EDR_ID": "B", "block_5km": "b2",
             "has_slope3m": True, "LTM_CODE": "L1"},
            {"split_final": "test", "group_id": "G3", "NAC_EDR_ID": "C", "block_5km": "b3",
             "has_slope3m": True, "LTM_CODE": "L2"},
        ])
        audit = fs.split_audit(frame, "split_final")
        for overlap in audit["overlap"].values():
            self.assertEqual(overlap["groups"], 0)
            self.assertEqual(overlap["products"], 0)
            self.assertEqual(overlap["blocks_5km"], 0)


if __name__ == "__main__":
    unittest.main()
