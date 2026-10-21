#!/usr/bin/env python3
"""Offline tests for the mosaic GeoTIFF export (georeferencing must be copied, not re-derived)."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mosaic_to_geotiff_p20",
                                              ROOT / "scripts" / "mosaic_to_geotiff_p20.py")
geo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = geo
spec.loader.exec_module(geo)

SCALE = (2.0, 2.0, 0.0)
TIEPOINT = (0.0, 0.0, 0.0, -4749762.0, 292026.0, 0.0)
GEOKEYS = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3075, 0, 1, 17)


def make_dtm(path: Path, shape=(16, 12)):
    tags = geo.build_extratags({"scale": SCALE, "tiepoint": TIEPOINT, "geokeys": GEOKEYS})
    data = np.zeros(shape, dtype=np.float32)
    tifffile.imwrite(path, data, extratags=tags, photometric="minisblack")


class TestExport(unittest.TestCase):
    def test_georeferencing_is_copied_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            dtm = Path(tmp) / "dtm.tif"
            raw = Path(tmp) / "mosaic.raw"
            out = Path(tmp) / "mosaic.tif"
            make_dtm(dtm, shape=(16, 12))
            grid = np.full((16, 12), 0.02, dtype="<f4")
            grid[0, 0] = 0.0                      # uncovered sentinel
            grid.tofile(raw)
            report = geo.export(raw, dtm, out, 16, 12)
            self.assertTrue(report["georeferencing_matches_dtm"])
            self.assertEqual(report["shape"], [16, 12])
            self.assertIn("nodata", report["tags_written"])
            values = tifffile.imread(out)
            self.assertTrue(np.isnan(values[0, 0]))       # 0 became NaN
            self.assertAlmostEqual(float(values[1, 1]), 0.02, places=6)

    def test_grid_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            dtm = Path(tmp) / "dtm.tif"
            raw = Path(tmp) / "mosaic.raw"
            make_dtm(dtm, shape=(16, 12))
            np.zeros((10, 10), dtype="<f4").tofile(raw)
            with self.assertRaises(ValueError):
                geo.export(raw, dtm, Path(tmp) / "o.tif", 10, 10)

    def test_missing_georeferencing_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "plain.tif"
            tifffile.imwrite(plain, np.zeros((8, 8), dtype=np.float32))
            with self.assertRaises(ValueError):
                geo.read_geo_tags(plain)

    def test_nodata_string_is_terminated(self):
        tags = geo.build_extratags({"scale": SCALE, "tiepoint": TIEPOINT, "geokeys": GEOKEYS})
        nodata = [t for t in tags if t[0] == 42113][0]
        self.assertEqual(nodata[3], "nan\x00")


if __name__ == "__main__":
    unittest.main()
