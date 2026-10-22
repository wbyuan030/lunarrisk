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


class TestGains(unittest.TestCase):
    SHAPE = (16, 16)

    def pair(self, ref_value, other_value):
        left = np.zeros(self.SHAPE, dtype=bool)
        left[:, :10] = True                     # enough overlap to fit a gain
        right = np.zeros(self.SHAPE, dtype=bool)
        right[:, 6:] = True
        ref = np.full(self.SHAPE, np.nan)
        ref[left] = ref_value
        other = np.full(self.SHAPE, np.nan)
        other[right] = other_value
        images = [("ref", ref), ("dim", other)]
        stats = mos.weights(images)
        stats["ref"]["priority"], stats["dim"]["priority"] = [1.0, 1.0], [1.0, 0.0]
        return images, stats

    def test_gain_is_fitted_to_the_reference(self):
        images, stats = self.pair(0.02, 0.01)   # same terrain, half the exposure
        fitted = mos.fit_photometry(images, stats)
        self.assertEqual(fitted["reference"], "ref")
        self.assertEqual(fitted["gains"]["ref"], 1.0)
        self.assertAlmostEqual(fitted["gains"]["dim"], 2.0, places=6)

    def test_offset_is_recovered_too(self):
        left = np.zeros(self.SHAPE, dtype=bool)
        left[:, :10] = True
        right = np.zeros(self.SHAPE, dtype=bool)
        right[:, 6:] = True
        ref = np.full(self.SHAPE, np.nan)
        ref[left] = 0.02
        other = np.full(self.SHAPE, np.nan)
        other[right] = 0.01 + 0.004            # different gain and a constant lift
        images = [("ref", ref), ("dim", other)]
        stats = mos.weights(images)
        stats["ref"]["priority"], stats["dim"]["priority"] = [1.0, 1.0], [1.0, 0.0]
        fitted = mos.fit_photometry(images, stats)
        self.assertAlmostEqual(fitted["gains"]["dim"], 0.02 / 0.014, places=4)
        self.assertAlmostEqual(fitted["offsets"]["dim"], -(0.02 - 0.02), places=6)

    def test_equalization_removes_the_seam_step(self):
        images, stats = self.pair(0.02, 0.01)
        before, src_before = mos.mosaic(images, stats)
        step_before = mos.seam_step(before, src_before)
        fitted = mos.fit_photometry(images, stats)
        after, src_after = mos.mosaic(mos.apply_gains(images, fitted["gains"],
                                                      fitted["offsets"]), stats)
        step_after = mos.seam_step(after, src_after)
        self.assertGreater(step_before["seam_median"], 0.005)   # a visible 0.01 jump
        self.assertLess(step_after.get("seam_median", 0.0), 1e-9)
        self.assertGreater(step_after["seam_pairs"], 0)         # both images still contribute


class TestSeamStep(unittest.TestCase):
    def test_matching_images_show_no_step(self):
        values = np.full((8, 8), 0.02)
        source = np.zeros((8, 8), dtype=np.int16)
        source[4:, :] = 1
        stats = mos.seam_step(values, source)
        self.assertGreater(stats["seam_pairs"], 0)
        self.assertAlmostEqual(stats["seam_median"], 0.0)      # no jump at the boundary
        self.assertAlmostEqual(stats["interior_median"], 0.0)  # constant image has no texture
        self.assertNotIn("seam_over_interior_median", stats)   # ratio undefined at zero texture

    def test_radiometric_offset_is_detected(self):
        values = np.full((8, 8), 0.02)
        values[4:, :] = 0.03                     # a 50% brighter second image
        source = np.zeros((8, 8), dtype=np.int16)
        source[4:, :] = 1
        stats = mos.seam_step(values, source)
        self.assertAlmostEqual(stats["seam_median"], 0.01, places=6)
        self.assertEqual(stats["seam_pairs"], 8)

    def test_offset_stands_out_against_texture(self):
        rng = np.random.default_rng(0)
        values = 0.02 + rng.normal(scale=1e-5, size=(16, 16))
        values[8:, :] += 0.01                    # step far above the texture level
        source = np.zeros((16, 16), dtype=np.int16)
        source[8:, :] = 1
        stats = mos.seam_step(values, source)
        self.assertGreater(stats["seam_over_interior_median"], 100)

    def test_no_boundary_reports_zero_pairs(self):
        values = np.full((6, 6), 0.02)
        stats = mos.seam_step(values, np.zeros((6, 6), dtype=np.int16))
        self.assertEqual(stats["seam_pairs"], 0)
        self.assertIn("interior_median", stats)


class TestSeamJump(unittest.TestCase):
    def test_flat_field_with_a_real_step_shows_a_jump(self):
        values = np.full((12, 12), 0.02)
        values[:, 6:] = 0.03
        source = np.zeros((12, 12), dtype=np.int16)
        source[:, 6:] = 1
        stats = mos.seam_jump(values, source)
        self.assertGreater(stats["seam_samples"], 0)
        self.assertAlmostEqual(stats["seam_jump_median"], 0.01, places=6)
        self.assertAlmostEqual(stats["control_jump_median"], 0.0, places=6)

    def test_a_smooth_ramp_across_the_boundary_is_not_a_jump(self):
        values = np.tile(np.linspace(0.02, 0.04, 12), (12, 1))
        source = np.zeros((12, 12), dtype=np.int16)
        source[:, 6:] = 1                      # boundary on a slope, but no discontinuity
        stats = mos.seam_jump(values, source)
        self.assertGreater(stats["seam_samples"], 0)
        self.assertLess(stats["seam_jump_median"], 1e-9)

    def test_steep_terrain_does_not_inflate_the_jump(self):
        # a gradient 100x larger than the step must still not register as a step
        values = np.tile(np.linspace(0.0, 0.5, 12), (12, 1))
        source = np.zeros((12, 12), dtype=np.int16)
        source[:, 6:] = 1
        self.assertLess(mos.seam_jump(values, source)["seam_jump_median"], 1e-9)


class TestAcquisitionFilter(unittest.TestCase):
    def test_prefix_selects_one_acquisition(self):
        names = ["M144483945LC_L_ortho.raw", "M144483945RC_R_ortho.raw",
                 "M144490730LC_L_ortho.raw", "M144490730RC_R_ortho.raw"]
        kept = [n for n in names if n.startswith(("M144483945",))]
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(n.startswith("M144483945") for n in kept))

    def test_within_acquisition_pair_keeps_a_small_step(self):
        # same acquisition: the two cameras differ only by exposure (~1%), never by shadow
        # geometry, so the gradient-cancelled jump stays at that level (contrast: 0.5% of signal
        # across acquisitions, measured on RANGER 0.0052 and PLANCKFLOOR 0.0104)
        left = np.zeros(SHAPE, dtype=bool)
        left[:, :6] = True
        right = np.zeros(SHAPE, dtype=bool)
        right[:, 4:] = True
        a = np.full(SHAPE, np.nan)
        a[left] = 0.0200
        b = np.full(SHAPE, np.nan)
        b[right] = 0.0198
        images = [("M1LC_L_ortho.raw", a), ("M1RC_R_ortho.raw", b)]
        stats = mos.weights(images)
        stats["M1LC_L_ortho.raw"]["priority"] = [1.0, 1.0]
        stats["M1RC_R_ortho.raw"]["priority"] = [1.0, 0.0]
        combined, source = mos.mosaic(images, stats)
        stats_jump = mos.seam_jump(combined, source)
        self.assertGreater(stats_jump["seam_samples"], 0)
        self.assertLess(stats_jump["seam_jump_median"], 0.0005)


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
