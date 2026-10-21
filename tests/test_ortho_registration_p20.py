#!/usr/bin/env python3
"""Offline tests for the P20 registration check (synthetic shifts only)."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ortho_registration_p20",
                                              ROOT / "scripts" / "ortho_registration_p20.py")
reg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reg
spec.loader.exec_module(reg)


def texture(n=256, seed=0):
    """Band-limited random texture: closer to lunar regolith than white noise."""
    rng = np.random.default_rng(seed)
    coarse = rng.normal(size=(n // 4, n // 4))
    return np.kron(coarse, np.ones((4, 4)))


def shift(a, dy, dx):
    """Integer shift that moves content down/right (np.roll is a wrap: crop back)."""
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


class TestNccShift(unittest.TestCase):
    def test_recovers_integer_shift_direction(self):
        a = texture()
        # b holds the same texture displaced by (dy=3, dx=-5) relative to a
        b = shift(a, 3, -5)
        dy, dx, peak = reg.ncc_shift(a[40:-40, 40:-40], b[40:-40, 40:-40], radius=16)
        self.assertAlmostEqual(dy, -3, delta=0.6)
        self.assertAlmostEqual(dx, 5, delta=0.6)
        self.assertGreater(peak, 0.9)

    def test_identical_patches_give_zero(self):
        a = texture(seed=3)
        dy, dx, peak = reg.ncc_shift(a, a, radius=8)
        self.assertAlmostEqual(dy, 0.0, delta=0.05)
        self.assertAlmostEqual(dx, 0.0, delta=0.05)
        self.assertAlmostEqual(peak, 1.0, delta=1e-6)

    def test_flat_patch_is_reported_without_crashing(self):
        a = np.zeros((32, 32))
        dy, dx, peak = reg.ncc_shift(a, a, radius=4)
        self.assertEqual((dy, dx, peak), (0.0, 0.0, 0.0))


class TestWindows(unittest.TestCase):
    def test_validity_threshold_is_enforced(self):
        mask = np.zeros((300, 300), dtype=bool)
        mask[:200, :] = True                      # one clean 128 block, one speckled one
        mask[200:, :] = False
        spots = reg.windows(mask, 128, 64, 0.999, 100)
        self.assertIn((0, 0), spots)
        self.assertTrue(all(y + 128 <= 200 for y, _ in spots))

    def test_hole_rejects_at_strict_threshold_only(self):
        mask = np.ones((200, 200), dtype=bool)
        mask[60:70, 60:70] = False                # 100 px hole = 0.39% of a 128 window
        self.assertEqual(reg.windows(mask, 128, 64, 0.999, 100), [])
        # 100/16384 = 0.9939 covered, so 0.999 rejects and 0.99 accepts
        self.assertIn((0, 0), reg.windows(mask, 128, 64, 0.99, 100))


class TestHillshade(unittest.TestCase):
    def test_flat_dem_is_uniform(self):
        shade = reg.hillshade(np.zeros((16, 16)), azimuth=315.0, altitude=45.0)
        self.assertTrue(np.allclose(shade, np.sin(np.deg2rad(45.0))))

    def test_west_facing_slope_is_dark_under_eastern_sun(self):
        yy, xx = np.mgrid[0:32, 0:32]
        ramp = (xx * 0.1).astype(float)           # elevation grows to +x, so the surface faces -x
        east = reg.hillshade(ramp, azimuth=90.0, altitude=30.0)
        west = reg.hillshade(ramp, azimuth=270.0, altitude=30.0)
        self.assertGreater(west.mean(), east.mean())


class TestLoadRaw(unittest.TestCase):
    def test_null_sentinel_becomes_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ortho.raw"
            grid = np.full((4, 4), 0.25, dtype="<f4")
            grid[0, 0] = 0.9
            grid[1, 1] = 0.8
            grid[2, 2] = 0.7
            grid[3, 3] = 0.6
            grid.tofile(path)
            image = reg.load_raw(path, 4, 4)
            self.assertEqual(int(np.isfinite(image).sum()), 4)
            self.assertTrue(np.isnan(image[0, 1]))


if __name__ == "__main__":
    unittest.main()
