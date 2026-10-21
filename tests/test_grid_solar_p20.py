#!/usr/bin/env python3
"""Offline tests for the P20 grid-arithmetic and solar-geometry helpers."""
from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


grid = load("grid_consistency_p20", "grid_consistency_p20.py")
solar = load("solar_azimuth_p20", "solar_azimuth_p20.py")

# RANGER left ortho, values copied from the real cam2map output label
MAPPING = {
    "MinimumLatitude": "8.6909262", "MaximumLatitude": "9.6304002",
    "MinimumLongitude": "21.4103879", "MaximumLongitude": "21.5649788",
}
LINES, SAMPLES = 14244, 2315


class TestGridArithmetic(unittest.TestCase):
    def test_corners_map_to_raster_corners(self):
        x, y = grid.latlon_to_pixel(MAPPING, LINES, SAMPLES, 9.6304002, 21.4103879)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        x, y = grid.latlon_to_pixel(MAPPING, LINES, SAMPLES, 8.6909262, 21.5649788)
        self.assertAlmostEqual(x, float(SAMPLES), places=6)
        self.assertAlmostEqual(y, float(LINES), places=6)

    def test_inverse_round_trip_at_pixel_centres(self):
        for i, j in ((0, 0), (1157, 7122), (2314, 14243)):
            lat, lon = grid.pixel_to_latlon(MAPPING, LINES, SAMPLES, i + 0.5, j + 0.5)
            x, y = grid.latlon_to_pixel(MAPPING, LINES, SAMPLES, lat, lon)
            self.assertAlmostEqual(x, i + 0.5, places=6)
            self.assertAlmostEqual(y, j + 0.5, places=6)

    def test_pixel_size_is_two_metres_in_millidegrees(self):
        # 2 m on R = 1737.4 km is 6.5955e-5 deg of latitude per pixel
        dy = (9.6304002 - 8.6909262) / LINES
        self.assertAlmostEqual(dy * 3600, 0.2374, places=3)


class TestSolarAzimuth(unittest.TestCase):
    def test_sun_to_the_east_is_90(self):
        self.assertAlmostEqual(solar.solar_azimuth(0.0, 0.0, 0.0, 90.0), 90.0, places=6)

    def test_sun_to_the_west_is_270(self):
        self.assertAlmostEqual(solar.solar_azimuth(0.0, 0.0, 0.0, -90.0), 270.0, places=6)

    def test_sun_to_the_north_and_south(self):
        self.assertAlmostEqual(solar.solar_azimuth(0.0, 0.0, 10.0, 0.0), 0.0, places=6)
        self.assertAlmostEqual(solar.solar_azimuth(0.0, 0.0, -10.0, 0.0), 180.0, places=6)

    def test_ranger_scene_matches_the_spice_solution(self):
        # campt for M144483945LC gives lat 9.2663, lon 21.5387, sub-solar (-1.0578, 67.9604);
        # the empirical hillshade search preferred 105 deg, so the analytic value must be near it.
        az = solar.solar_azimuth(9.2662598746462, 21.53866549593, -1.0577739236887, 67.960425825467)
        self.assertAlmostEqual(az, 100.1, delta=0.5)

    def test_azimuth_is_always_normalised(self):
        for lon in (-350.0, -180.0, 180.0, 359.0, 720.0):
            self.assertGreaterEqual(solar.solar_azimuth(12.0, 0.0, -2.0, lon), 0.0)
            self.assertLess(solar.solar_azimuth(12.0, 0.0, -2.0, lon), 360.0)


if __name__ == "__main__":
    unittest.main()
