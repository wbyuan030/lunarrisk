#!/usr/bin/env python3
"""Offline tests for the crater-centroid GCP matcher (synthetic blobs only)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("crater_gcp_p20", ROOT / "scripts" / "crater_gcp_p20.py")
gcp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gcp
spec.loader.exec_module(gcp)


def crater_field(centres, size=256, radius=5.0, depth=1.0):
    """Dark-centred rings on a flat background."""
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size), dtype=float)
    for (cy, cx) in centres:
        r = np.hypot(yy - cy, xx - cx)
        img += depth * np.exp(-((r - radius) ** 2) / (2 * 1.2 ** 2))      # bright rim
        img -= depth * 0.7 * np.exp(-(r ** 2) / (2 * 1.5 ** 2))           # dark floor
    return img


CENTRES = [(40.0, 40.0), (40.0, 150.0), (150.0, 40.0), (150.0, 150.0), (95.0, 95.0)]


class TestDetect(unittest.TestCase):
    def test_finds_synthetic_craters(self):
        img = crater_field(CENTRES)
        blobs = gcp.detect(img, [2, 3, 5], threshold=0.02, min_sep=7)
        self.assertGreaterEqual(len(blobs), 4)
        found = np.array([(b[0], b[1]) for b in blobs], dtype=float)
        for cy, cx in CENTRES:
            distance = np.hypot(found[:, 0] - cy, found[:, 1] - cx).min()
            self.assertLess(distance, 3.0, f"crater at {(cy, cx)} missed")

    def test_flat_image_yields_no_strong_blobs(self):
        blobs = gcp.detect(np.zeros((64, 64)), [2, 3], threshold=0.05, min_sep=7)
        self.assertEqual(blobs, [])


class TestMatch(unittest.TestCase):
    def test_recovers_a_known_shift(self):
        a = crater_field(CENTRES)
        shifted = crater_field([(cy + 2.0, cx - 3.0) for cy, cx in CENTRES])
        pairs = gcp.match(gcp.detect(a, [2, 3, 5], 0.02, 7),
                          gcp.detect(shifted, [2, 3, 5], 0.02, 7), max_dist=8.0)
        stats = gcp.dominant_offset(pairs)
        self.assertGreaterEqual(stats["matches"], 4)
        self.assertAlmostEqual(stats["dominant_dy_px"], 2.0, delta=1.0)
        self.assertAlmostEqual(stats["dominant_dx_px"], -3.0, delta=1.0)
        self.assertEqual(stats["dominant_dy_px"], 2.0)
        self.assertEqual(stats["dominant_dx_px"], -3.0)

    def test_identical_input_gives_zero_offset(self):
        a = crater_field(CENTRES)
        pairs = gcp.match(gcp.detect(a, [2, 3, 5], 0.02, 7),
                          gcp.detect(a, [2, 3, 5], 0.02, 7), max_dist=4.0)
        stats = gcp.dominant_offset(pairs)
        self.assertEqual((stats["dominant_dy_px"], stats["dominant_dx_px"]), (0.0, 0.0))
        self.assertEqual(stats["dominant_share"], 1.0)

    def test_far_apart_points_do_not_match(self):
        a = gcp.detect(crater_field(CENTRES), [2, 3, 5], 0.02, 7)
        b = gcp.detect(crater_field([(cy + 40, cx) for cy, cx in CENTRES]), [2, 3, 5], 0.02, 7)
        self.assertEqual(gcp.match(a, b, max_dist=8.0), [])


class TestBowlDetection(unittest.TestCase):
    def test_dem_bowls_are_detected_without_any_illumination_model(self):
        # --dtm-mode dem feeds elevation straight to the detector, so a bowl must register
        yy, xx = np.mgrid[0:192, 0:192]
        dem = np.zeros((192, 192))
        bowls = [(48.0, 48.0), (48.0, 140.0), (140.0, 48.0), (140.0, 140.0)]
        for cy, cx in bowls:
            r = np.hypot(yy - cy, xx - cx)
            dem -= 10.0 * np.exp(-(r ** 2) / (2 * 4.0 ** 2))
        blobs = gcp.detect(dem, [3, 5, 8], threshold=0.02, min_sep=9)
        found = np.array([(b[0], b[1]) for b in blobs], dtype=float)
        self.assertGreaterEqual(len(blobs), 4)
        for cy, cx in bowls:
            self.assertLess(np.hypot(found[:, 0] - cy, found[:, 1] - cx).min(), 2.5)


class TestHillshade(unittest.TestCase):
    def test_flat_dem_is_constant(self):
        shade = gcp.hillshade(np.zeros((8, 8)), azimuth=100.0, altitude=42.6)
        self.assertTrue(np.allclose(shade, np.sin(np.deg2rad(42.6))))


if __name__ == "__main__":
    unittest.main()
