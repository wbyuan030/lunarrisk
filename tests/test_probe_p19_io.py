#!/usr/bin/env python3
"""Offline tests for the P19 Gate 2 native-DTM I/O probe (synthetic label + GeoTIFF only)."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_p19_io", ROOT / "scripts" / "probe_p19_io.py")
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)

LABEL = """PDS_VERSION_ID                  = PDS3
RECORD_TYPE                     = FIXED_LENGTH
RECORD_BYTES                    = 10000
FILE_RECORDS                    = 2500
DATA_SET_ID                     = "LRO-L-LROC-5-RDR-V1.0"
PRODUCT_ID                      = "NAC_DTM_TEST01"
PRODUCT_VERSION_ID              = "v1.2"
INSTRUMENT_NAME                 = "LUNAR RECONNAISSANCE ORBITER CAMERA"
^IMAGE                          = ("NAC_DTM_TEST01.TIF", 20675 <BYTES>)

OBJECT                          = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE           = "POLARSTEREOGRAPHIC"
  A_AXIS_RADIUS                 = 1737.4 <KM>
  CENTER_LATITUDE               = -90
  MAP_SCALE                     = 2.0 <METERS/PIXEL>
  MINIMUM_LATITUDE              = -89.84
  MAXIMUM_LATITUDE              = -89.63
  WESTERNMOST_LONGITUDE         = 180.82
  EASTERNMOST_LONGITUDE         = 226.62
END_OBJECT                      = IMAGE_MAP_PROJECTION

OBJECT                          = IMAGE
  LINES                         = 2500
  LINE_SAMPLES                  = 2500
  SAMPLE_TYPE                   = PC_REAL
  SAMPLE_BITS                   = 32
  BANDS                         = 1
  DESCRIPTION                   = "Height [m] equals DN. The NoDATA value is
    -3.40282266e+038 with a missing constant of -1e+32."
END_OBJECT                      = IMAGE
END
"""


def write_geotiff(path: Path, array: np.ndarray, pixel_scale=(2.0, 2.0), nodata="-3.40282266e+38"):
    import tifffile

    # GeoKey values live either inline (location 0) or in the double-params tag (34736).
    geo_keys = np.array([
        1, 1, 0, 4,
        1024, 0, 1, 1,        # GTModelTypeGeoKey = projected
        1025, 0, 1, 1,        # GTRasterTypeGeoKey = pixel is area
        4096, 0, 1, 15,       # ProjCoordTransGeoKey = polar stereographic
        2057, 34736, 1, 0,    # GeogSemiMajorAxisGeoKey -> doubles[0]
    ], dtype="<u2")
    doubles = np.array([1737400.0], dtype="<f8")
    nodata_bytes = (nodata + "\x00").encode()
    tifffile.imwrite(path, array.astype("<f4"), extratags=[
        (33550, "d", 3, (pixel_scale[0], pixel_scale[1], 0.0), False),
        (34735, "H", geo_keys.size, geo_keys.tolist(), False),
        (34736, "d", doubles.size, doubles.tolist(), False),
        (42113, "s", len(nodata_bytes), nodata_bytes, False),
    ])


class TestLabelParsing(unittest.TestCase):
    def test_keywords_and_objects(self):
        label = probe.parse_pds_label(LABEL)
        self.assertEqual(label["PRODUCT_ID"], "NAC_DTM_TEST01")
        self.assertEqual(label["PROJECTION"], "POLARSTEREOGRAPHIC")
        self.assertEqual(label["MAP_SCALE_M"], 2.0)
        self.assertEqual(label["SPHERE_RADIUS_KM"], 1737.4)
        self.assertEqual(label["CENTER_LAT"], -90)
        self.assertEqual(label["LINES"], 2500)
        self.assertEqual(label["SAMPLE_BITS"], 32)
        self.assertEqual(label["SAMPLE_TYPE"], "PC_REAL")
        # real labels describe the sentinel in prose, not as a keyword
        self.assertAlmostEqual(label["NODATA"], -3.40282266e38)
        self.assertEqual(label["NODATA_SOURCE"], "IMAGE.DESCRIPTION")
        self.assertAlmostEqual(label["MISSING"], -1e32)
        self.assertEqual(label["IMAGE_FILE"], "NAC_DTM_TEST01.TIF")
        self.assertEqual(label["IMAGE_OFFSET_BYTES"], 20675)
        self.assertEqual(probe.image_pointer(label), ("NAC_DTM_TEST01.TIF", 20675))


class TestLabelNoDataKeyword(unittest.TestCase):
    def test_explicit_nodata_keyword_wins(self):
        text = LABEL.replace('DESCRIPTION                   = "Height', 'NoDATA = -9.5\n  DESCRIPTION = "Height')
        label = probe.parse_pds_label(text)
        self.assertAlmostEqual(label["NODATA"], -9.5)
        self.assertEqual(label["NODATA_SOURCE"], "IMAGE.NoDATA")


class TestLayoutCheck(unittest.TestCase):
    def test_one_based_pointer_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.tif"
            payload = np.arange(4, dtype="<f4").tobytes()
            path.write_bytes(b"\x00" * 16 + payload)
            info = probe.layout_check(path, 17, 16, 2, 2)
            self.assertTrue(info["label_pointer_is_1_based"])
            self.assertEqual(info["zero_based_start"], 16)
            self.assertEqual(info["bytes_after_offset"], 16)
            self.assertIn("raw_offset_view", info)
            self.assertTrue(np.array_equal(info["raw_offset_view"], np.arange(4, dtype="<f4").reshape(2, 2)))

    def test_mismatched_layout_reports_without_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.tif"
            path.write_bytes(b"\x00" * 16 + np.arange(4, dtype="<f4").tobytes())
            info = probe.layout_check(path, 17, 999, 2, 2)
            self.assertNotIn("raw_offset_view", info)
            self.assertEqual(info["bytes_after_offset"], 16)
            self.assertEqual(info["expected_bytes"], 999)


class TestMasksAndSlope(unittest.TestCase):
    def test_nodata_and_sentinels_are_invalid(self):
        elevation = np.array([[1.0, 2.0], [-3.40282266e38, -1.0e32]])
        mask = probe.valid_mask(elevation, -3.40282266e38)
        self.assertEqual(mask.tolist(), [[True, True], [False, False]])

    def test_interior_mask_excludes_edges_and_holes(self):
        mask = np.ones((7, 7), dtype=bool)
        mask[3, 3] = False
        interior = probe.interior_mask(mask)
        self.assertFalse(interior[0].any())
        self.assertFalse(interior[:, 0].any())
        self.assertFalse(interior[6, 6])   # outer ring has no full neighbourhood
        self.assertFalse(interior[3, 3])   # the hole itself
        self.assertFalse(interior[2, 2])   # 8-neighbour of the hole
        self.assertTrue(interior[5, 5])    # far from hole and border
        self.assertEqual(int(interior.sum()), 16)

    def test_slope_of_a_known_plane(self):
        rows, cols = np.mgrid[0:6, 0:6].astype(np.float64)
        plane = rows  # 1 m rise per row
        self.assertAlmostEqual(float(probe.slope_degrees(plane, 1.0).mean()), 45.0, places=4)
        self.assertAlmostEqual(float(probe.slope_degrees(plane, 2.0).mean()), 26.565, places=3)
        flat = np.zeros((6, 6))
        self.assertAlmostEqual(float(probe.slope_degrees(flat, 2.0).max()), 0.0, places=6)


class TestLabelStatistics(unittest.TestCase):
    def test_fraction_and_proxy_rule(self):
        flat = np.full((20, 20), 5.0)
        stats = probe.label_statistics(flat, 2.0, nodata=-3.40282266e38)
        self.assertAlmostEqual(stats["valid_fraction"], 1.0)
        self.assertAlmostEqual(stats["g10"], 0.0)
        self.assertEqual(stats["unsafe_proxy_rule"], False)
        self.assertLess(stats["evaluated_fraction_of_total"], 1.0)  # 1-px border dropped

        steep = np.add.outer(np.arange(40.0), np.zeros(40)) * 1.0  # 1 m rise per 1 m row
        stats = probe.label_statistics(steep, 1.0)
        self.assertAlmostEqual(stats["g10"], 1.0)
        self.assertAlmostEqual(stats["g15"], 1.0)
        self.assertEqual(stats["unsafe_proxy_rule"], True)

    def test_holes_reduce_valid_fraction(self):
        elevation = np.full((20, 20), 1.0)
        elevation[:10, :] = -3.40282266e38
        stats = probe.label_statistics(elevation, 2.0, nodata=-3.40282266e38)
        self.assertAlmostEqual(stats["valid_fraction"], 0.5, places=3)
        self.assertLess(stats["evaluated_pixels"], 20 * 20 * 0.5)

    def test_block_mean_aggregates_and_keeps_nodata(self):
        elevation = np.arange(16.0).reshape(4, 4)
        grid, factor = probe.block_mean(elevation, 2)
        self.assertEqual(factor, 2)
        self.assertAlmostEqual(grid[0, 0], 2.5)  # mean of 0,1,4,5
        self.assertAlmostEqual(grid[1, 1], 12.5)
        holed = elevation.copy()
        holed[0, 0] = probe.MISSING_CONSTANT
        # strict default: a block with any missing pixel is dropped, never interpolated
        grid, _ = probe.block_mean(holed, 2)
        self.assertTrue(np.isnan(grid[0, 0]))
        grid, _ = probe.block_mean(holed, 2, min_valid_fraction=0.5)
        self.assertAlmostEqual(grid[0, 0], 10.0 / 3.0)


class TestGeoTiffProbe(unittest.TestCase):
    def _scene(self, tmp: str):
        base = Path(tmp)
        rows, cols = np.mgrid[0:64, 0:64].astype(np.float64)
        array = (rows + cols) * 0.05  # gentle slope: far below the 10 deg rule
        array[:8, :8] = -3.40282266e38
        tif = base / "NAC_DTM_TEST01.TIF"
        write_geotiff(tif, array)
        (base / "NAC_DTM_TEST01.LBL").write_text(LABEL.replace("2500", "64"), encoding="utf-8")
        return tif, base

    def test_metadata_and_probe_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            tif, base = self._scene(tmp)
            meta = probe.geotiff_metadata(tif)
            self.assertEqual(meta["shape"], [64, 64])
            self.assertEqual(meta["model_pixel_scale"][:2], [2.0, 2.0])
            self.assertEqual(meta["geo_keys"]["ProjCoordTransGeoKey"], 15)
            self.assertEqual(meta["geo_keys"]["GeogSemiMajorAxisGeoKey"], [1737400.0])
            self.assertEqual(meta["gdal_nodata"], "-3.40282266e+38")

            record = probe.probe_scene(tif, base / "NAC_DTM_TEST01.LBL", scales=(1, 2))
            self.assertEqual(record["product_id"], "NAC_DTM_TEST01")
            self.assertTrue(record["shape_matches_label"])
            self.assertTrue(record["dtype_matches_label"])
            self.assertLess(record["scales"]["factor_1"]["valid_fraction"], 1.0)
            self.assertEqual(record["scales"]["factor_2"]["post_spacing_m"], 4.0)
            self.assertFalse(record["scales"]["factor_1"]["unsafe_proxy_rule"])
            # the synthetic file does not reproduce the real 20675-byte PDS offset layout,
            # so the byte-layout cross-check must stay unset rather than guess
            self.assertNotIn("raw_offset_view_identical_to_tiff", record)
            self.assertNotEqual(record["bytes_after_offset"], record["expected_data_bytes"])
            self.assertEqual(record["effective_nodata"], -3.40282266e38)
            self.assertEqual(record["nodata_source"], "IMAGE.DESCRIPTION")

    def test_geotiff_nodata_is_the_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tif, base = self._scene(tmp)
            stripped = LABEL.replace("NoDATA value is", "value is").replace("2500", "64")
            (base / "NAC_DTM_TEST01.LBL").write_text(stripped, encoding="utf-8")
            record = probe.probe_scene(tif, base / "NAC_DTM_TEST01.LBL", scales=(1,))
            self.assertEqual(record["nodata_source"], "GEOTIFF_GDAL_NODATA")
            self.assertAlmostEqual(record["effective_nodata"], -3.40282266e38, places=20)

    def test_main_writes_json_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, base = self._scene(tmp)
            out = base / "probe.json"
            code = probe.main(["--data", str(base), "--out", str(out), "--scales", "1", "2"])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("NAC_DTM_TEST01", report["scenes"])
            self.assertEqual(report["scenes"]["NAC_DTM_TEST01"]["label_map_scale_m"], 2.0)
            self.assertIn("tif_sha256", report["scenes"]["NAC_DTM_TEST01"])
            self.assertTrue(report["scenes"]["NAC_DTM_TEST01"]["shape_matches_label"])


if __name__ == "__main__":
    unittest.main()
