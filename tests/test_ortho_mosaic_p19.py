#!/usr/bin/env python3
"""Offline tests for the P20 mosaic policy (synthetic masks and weights only)."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ortho_mosaic_p19", ROOT / "scripts" / "ortho_mosaic_p19.py")
mos = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mos
spec.loader.exec_module(mos)

SHAPE = (8, 8)


def image(value, valid):
    out = np.full(SHAPE, np.nan)
    out[valid] = value
    return out


class TestMosaicSelection(unittest.TestCase):
    def test_single_valid_image_fills_exactly_its_footprint(self):
        valid = np.zeros(SHAPE, dtype=bool)
        valid[:4] = True
        images = [("a_L_ortho.raw", image(1.0, valid))]
        stats = mos.weights(images)
        out, source = mos.mosaic(images, stats)
        self.assertEqual(int(np.isfinite(out).sum()), valid.sum())
        self.assertTrue(np.allclose(out[valid], 1.0))
        self.assertTrue(np.all(source[valid] == 0))
        self.assertTrue(np.all(source[~valid] == -1))

    def test_overlap_goes_to_the_higher_weight_not_the_average(self):
        overlap = np.zeros(SHAPE, dtype=bool)
        overlap[:4] = True
        left = image(10.0, overlap)
        right = image(20.0, overlap)
        stats = mos.weights([("a", left), ("b", right)])
        stats["a"]["priority"], stats["b"]["priority"] = [0.9, 1.0], [0.1, 1.0]
        out, source = mos.mosaic([("a", left), ("b", right)], stats)
        self.assertTrue(np.all(out[overlap] == 10.0))     # hard pick, never (10+20)/2
        self.assertTrue(np.all(source[overlap] == 0))

    def test_complementary_footprints_union_without_seam_error(self):
        top = np.zeros(SHAPE, dtype=bool)
        top[:4] = True
        bottom = np.zeros(SHAPE, dtype=bool)
        bottom[4:] = True
        images = [("a", image(5.0, top)), ("b", image(7.0, bottom))]
        stats = mos.weights(images)
        out, source = mos.mosaic(images, stats)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.all(out[:4] == 5.0))
        self.assertTrue(np.all(out[4:] == 7.0))

    def test_seam_length_counts_one_boundary(self):
        source = np.zeros(SHAPE, dtype=np.int16)
        source[4:, :] = 1
        self.assertEqual(mos.seam_length(source), SHAPE[1])   # 8 horizontal boundaries
        self.assertEqual(mos.seam_length(np.zeros(SHAPE, dtype=np.int16)), 0)

    def test_ties_keep_the_first_input(self):
        valid = np.ones(SHAPE, dtype=bool)
        images = [("a", image(3.0, valid)), ("b", image(4.0, valid))]
        stats = mos.weights(images)
        stats["a"]["priority"] = stats["b"]["priority"] = [1.0, 1.0]
        out, source = mos.mosaic(images, stats)
        self.assertTrue(np.all(out == 3.0))
        self.assertTrue(np.all(source == 0))


class TestSaturation(unittest.TestCase):
    def test_saturated_image_loses_regardless_of_sharpness(self):
        valid = np.ones((4, 4), dtype=bool)
        sharp = np.zeros((4, 4))
        sharp[:, ::2] = 32000.0            # high laplacian, but saturated
        soft = np.full((4, 4), 0.02)
        images = [("sharp_but_saturated", sharp), ("soft", soft)]
        stats = mos.weights(images)
        self.assertGreater(stats["sharp_but_saturated"]["sharpness"], stats["soft"]["sharpness"])
        self.assertEqual(mos.priority_order(images, stats)[0], 1)   # the unsaturated one wins
        _, source = mos.mosaic(images, stats)
        self.assertTrue(np.all(source == 1))

    def test_saturation_share_counts_only_valid_pixels(self):
        img = np.full((4, 4), np.nan)
        img[0, :] = 32000.0
        img[1, :2] = 0.5
        self.assertAlmostEqual(mos.saturated_share(img), 4 / 6)


class TestIlluminationGuard(unittest.TestCase):
    def test_same_sun_is_accepted(self):
        spread = mos.illumination_spread({
            "a": {"sun_azimuth_deg": 100.0, "sun_elevation_deg": 42.6},
            "b": {"sun_azimuth_deg": 100.3, "sun_elevation_deg": 43.5}})
        self.assertLess(spread["azimuth_spread_deg"], 1.0)
        self.assertLess(spread["elevation_spread_deg"], 1.0)

    def test_wrapped_azimuth_is_not_340_degrees(self):
        spread = mos.illumination_spread({
            "a": {"sun_azimuth_deg": 350.0, "sun_elevation_deg": 20.0},
            "b": {"sun_azimuth_deg": 10.0, "sun_elevation_deg": 20.0}})
        self.assertAlmostEqual(spread["azimuth_spread_deg"], 20.0)

    def test_different_sun_is_flagged(self):
        spread = mos.illumination_spread({
            "a": {"sun_azimuth_deg": 100.0, "sun_elevation_deg": 42.6},
            "b": {"sun_azimuth_deg": 250.0, "sun_elevation_deg": 20.0}})
        self.assertGreater(spread["azimuth_spread_deg"], 10.0)
        self.assertGreater(spread["elevation_spread_deg"], 5.0)

    def test_no_illumination_reports_no_images(self):
        self.assertEqual(mos.illumination_spread({})["images"], 0)


class TestLoadRaw(unittest.TestCase):
    def test_fill_value_becomes_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.raw"
            grid = np.full((4, 4), 0.3, dtype="<f4")
            grid[0, 0] = 0.9
            grid.tofile(path)
            values, fill_share = mos.load_raw(path, 4, 4)
            self.assertAlmostEqual(fill_share, 15 / 16)
            self.assertEqual(int(np.isfinite(values).sum()), 1)


if __name__ == "__main__":
    unittest.main()
