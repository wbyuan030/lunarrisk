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
    """Buffer drops are coordinate driven: no single nearest-neighbour label is consulted."""

    def frame(self, splits, lats, lons=None):
        return pd.DataFrame({
            "split_pre_buffer": splits,
            "center_lat": lats,
            "center_lon": lons if lons is not None else [0.0] * len(splits),
        })

    def buffer(self, frame, buffer_km=10.0):
        points = fs.unit_sphere(frame["center_lat"].to_numpy(), frame["center_lon"].to_numpy())
        return fs.apply_buffer(frame, points, buffer_km), points

    def test_shadowing_regression_val_is_dropped_even_when_train_is_closer(self):
        # Reviewer counterexample: train(0.00) val(0.05) test(0.20).
        # train-val 1.52 km, val-test 4.55 km. The old code kept val because its nearest
        # neighbour was train, leaving a surviving val-test pair under the buffer.
        frame = self.frame(["train", "val", "test"], [0.00, 0.05, 0.20])
        final, points = self.buffer(frame)
        self.assertEqual(final.tolist(), ["drop_buffer", "drop_buffer", "test"])
        contract = fs.surviving_pair_distances(final, points)
        self.assertNotIn("val_test", contract)

    def test_train_near_test_is_dropped_test_is_kept(self):
        frame = self.frame(["train", "test"], [0.00, 0.10])
        final, _ = self.buffer(frame)
        self.assertEqual(final.tolist(), ["drop_buffer", "test"])

    def test_test_tiles_are_never_dropped(self):
        frame = self.frame(["test", "test"], [0.00, 0.01])
        final, _ = self.buffer(frame)
        self.assertEqual(final.tolist(), ["test", "test"])

    def test_cal_near_val_is_dropped_but_val_survives(self):
        # cal(0.00) val(0.05) test(0.60): cal is 1.52 km from val -> dropped, val survives.
        frame = self.frame(["cal", "val", "test"], [0.00, 0.05, 0.60])
        final, _ = self.buffer(frame)
        self.assertEqual(final.tolist(), ["drop_buffer", "val", "test"])

    def test_cal_near_val_is_dropped_even_when_train_is_closest(self):
        # Shadowing variant for cal: train sits right on top of cal.
        frame = self.frame(["cal", "val", "test", "train"], [0.00, 0.05, 0.60, 0.001])
        final, _ = self.buffer(frame)
        self.assertEqual(final[0], "drop_buffer")
        self.assertEqual(final[3], "drop_buffer")

    def test_train_far_from_everything_survives(self):
        frame = self.frame(["train", "cal", "val", "test"], [0.0, 0.5, 2.0, 5.0])
        final, _ = self.buffer(frame)
        self.assertEqual(final.tolist(), ["train", "cal", "val", "test"])

    def test_buffer_contract_property(self):
        frame = self.frame(
            ["train", "train", "val", "cal", "test", "test", "val", "cal"],
            [0.000, 0.002, 0.001, 0.004, 0.30, 0.50, 0.28, 0.31],
        )
        final, points = self.buffer(frame, 10.0)
        contract = fs.surviving_pair_distances(final, points)
        for pair, entry in contract.items():
            self.assertGreaterEqual(entry["min_km"], 10.0, f"{pair} violates the buffer contract")
        self.assertTrue((final == "test").sum() == 2)

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
