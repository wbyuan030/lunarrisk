#!/usr/bin/env python3
"""Unit tests for slope label boundary semantics (synthetic inputs only)."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("make_slope_labels", ROOT / "scripts" / "make_slope_labels.py")
msl = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = msl
SPEC.loader.exec_module(msl)


class TestComputeLabels(unittest.TestCase):
    def test_exact_threshold_is_not_above(self):
        values = np.array([10.0, 10.0, 10.0, 10.0001])
        out = msl.compute_labels(values)
        self.assertEqual(out["n_valid"], 4)
        self.assertAlmostEqual(out["frac_gt_10deg_fraction"], 1 / 4)
        # 10.0 belongs to the [10, 15) bin; all four values are in that bin.
        self.assertEqual(out["bin_4"], 4)
        self.assertEqual(out["bin_5"], 0)

    def test_all_ten_bin_vs_strict(self):
        out = msl.compute_labels(np.full(8, 10.0))
        self.assertEqual(out["g10"], 0.0)
        self.assertEqual(out["bin_4_fraction"], 1.0)
        self.assertIs(out["unsafe_proxy_g10_gt_0_05"], False)

    def test_all_nan(self):
        out = msl.compute_labels(np.full(4, np.nan))
        self.assertEqual(out["n_valid"], 0)
        self.assertEqual(out["n_nan"], 4)
        self.assertEqual(out["valid_fraction"], 0.0)
        self.assertIsNone(out["g10"])
        self.assertIsNone(out["unsafe_proxy_g10_gt_0_05"])
        self.assertEqual(sum(out[f"bin_{i}"] for i in range(11)), 0)

    def test_out_of_range_values_are_invalid(self):
        out = msl.compute_labels(np.array([-1.0, -90.0, 91.0, 100.0, 0.0, 45.0, 90.0]))
        self.assertEqual(out["n_valid"], 3)
        self.assertEqual(out["n_negative"], 2)
        self.assertEqual(out["n_gt90"], 2)
        self.assertEqual(out["min_valid_deg"], 0.0)
        self.assertEqual(out["max_valid_deg"], 90.0)

    def test_valid_fraction_exactly_095(self):
        values = np.concatenate([np.zeros(19), [np.nan]])
        out = msl.compute_labels(values)
        self.assertAlmostEqual(out["valid_fraction"], 0.95)
        self.assertIs(out["valid_fraction_ge_0_95"], True)

    def test_bins_partition_valid_pixels(self):
        values = np.array([0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 60, 90, 10.0,
                           np.nan, -1, 91])
        out = msl.compute_labels(values)
        self.assertEqual(sum(out[f"bin_{i}"] for i in range(11)), out["n_valid"])
        self.assertEqual(out["bin_0"], 1)
        self.assertEqual(out["bin_10"], 2)  # 60 and 90
        self.assertAlmostEqual(out["g10"], 7 / 13)
        self.assertAlmostEqual(out["g15"], 6 / 13)

    def test_unsafe_proxy_uses_strict_rule(self):
        # Exactly 5% unsafe is NOT above the strict >0.05 rule.
        boundary = msl.compute_labels(np.array([11.0] * 5 + [0.0] * 95))
        self.assertFalse(boundary["unsafe_proxy_g10_gt_0_05"])
        above = msl.compute_labels(np.array([11.0] * 6 + [0.0] * 94))
        self.assertTrue(above["unsafe_proxy_g10_gt_0_05"])
        out2 = msl.compute_labels(np.array([10.0] * 5 + [0.0] * 95))
        self.assertFalse(out2["unsafe_proxy_g10_gt_0_05"])


class TestVariableSelection(unittest.TestCase):
    def test_unique_slope_name(self):
        ds = xr.Dataset({
            "slope_deg": (("y", "x"), np.zeros((2, 2))),
            "aspect": (("y", "x"), np.zeros((2, 2))),
            "foo": (("y", "x"), np.zeros((2, 2))),
        })
        self.assertEqual(msl.find_slope_variable(ds), "slope_deg")

    def test_multiple_slope_names_fail(self):
        ds = xr.Dataset({
            "slope_a": (("y", "x"), np.zeros((2, 2))),
            "slope_b": (("y", "x"), np.zeros((2, 2))),
        })
        with self.assertRaises(msl.SlopeLabelError):
            msl.find_slope_variable(ds)

    def test_single_2d_variable_fallback(self):
        ds = xr.Dataset({"band_data": (("y", "x"), np.zeros((2, 2)))})
        self.assertEqual(msl.find_slope_variable(ds), "band_data")

    def test_no_variable_fails(self):
        ds = xr.Dataset({"time": ("time", np.arange(3))})
        with self.assertRaises(msl.SlopeLabelError):
            msl.find_slope_variable(ds)

    def test_ambiguous_without_slope_name_fails(self):
        ds = xr.Dataset({
            "a": (("y", "x"), np.zeros((2, 2))),
            "b": (("y", "x"), np.zeros((2, 2))),
        })
        with self.assertRaises(msl.SlopeLabelError):
            msl.find_slope_variable(ds)

    def test_requested_variable(self):
        ds = xr.Dataset({"a": (("y", "x"), np.zeros((2, 2)))})
        self.assertEqual(msl.find_slope_variable(ds, "a"), "a")
        with self.assertRaises(msl.SlopeLabelError):
            msl.find_slope_variable(ds, "missing")


class TestUnitsAndFiles(unittest.TestCase):
    def test_units(self):
        ds = xr.Dataset({"slope": (("y", "x"), np.zeros((2, 2)), {"units": "degrees"})})
        ok, units, source = msl.angle_units_status(ds, "slope")
        self.assertTrue(ok)
        self.assertEqual(units, "degrees")
        self.assertIn("units", source)
        rad = xr.Dataset({"slope": (("y", "x"), np.zeros((2, 2)), {"units": "radian"})})
        ok, units, _ = msl.angle_units_status(rad, "slope")
        self.assertFalse(ok)
        missing = xr.Dataset({"slope": (("y", "x"), np.zeros((2, 2)))})
        self.assertFalse(msl.angle_units_status(missing, "slope")[0])

    def test_empty_file_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.nc"
            path.write_bytes(b"")
            with self.assertRaises(msl.SlopeLabelError):
                msl.label_one_file(path, None, allow_unverified_units=False)

    def test_label_one_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "M123_r1_c2.nc"
            ds = xr.Dataset(
                {"slope": (("y", "x"), np.array([[0.0, 10.0], [11.0, np.nan]]),
                           {"units": "degrees"})},
                attrs={"crs": "TEST_CRS", "pix_res_x": 3.0},
            )
            ds.to_netcdf(path)
            ds.close()
            row = msl.label_one_file(path, None, allow_unverified_units=False)
            self.assertEqual(row["sample_id"], "M123_r1_c2")
            self.assertEqual(row["source_id"], "M123")
            self.assertEqual(row["geo_join_status"], "GEO_JOIN_PENDING")
            self.assertEqual(row["n_valid"], 3)
            self.assertAlmostEqual(row["g10"], 1 / 3)
            self.assertTrue(row["unit_verified"])
            self.assertEqual(len(row["sha256"]), 64)

    def test_unverified_units_refused_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "M1_r1_c1.nc"
            xr.Dataset({"slope": (("y", "x"), np.zeros((2, 2)))}).to_netcdf(path)
            with self.assertRaises(msl.SlopeLabelError):
                msl.label_one_file(path, None, allow_unverified_units=False)
            row = msl.label_one_file(path, None, allow_unverified_units=True)
            self.assertFalse(row["unit_verified"])
            row2 = msl.label_one_file(
                path, None, allow_unverified_units=False,
                units_override="degrees", units_source="TEST_DOC",
            )
            self.assertTrue(row2["unit_verified"])
            self.assertEqual(row2["units_source"], "TEST_DOC")


if __name__ == "__main__":
    unittest.main()
