#!/usr/bin/env python3
"""Offline tests for the P20 DTM map-template builder."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_dtm_template", ROOT / "scripts" / "make_dtm_template.py")
tpl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tpl
spec.loader.exec_module(tpl)

GRID = {
    "dtm": "PLANCKFLOOR", "projection": "equirectangular",
    "clat": "-55.0", "clon": "180.0", "scale": "2.0000000000006", "radius": "1737.4",
    "lines": "17556", "samples": "3290", "ppd": "15161.67521207",
    "minlat": "-56.0701234", "maxlat": "-54.9122039",
    "minlon": "137.063397", "maxlon": "137.4417154",
}


class TestBuild(unittest.TestCase):
    def test_label_uses_dtm_values_verbatim(self):
        text = tpl.build(GRID)
        self.assertIn("ProjectionName     = Equirectangular", text)
        self.assertIn("CenterLatitude     = -55.0", text)
        self.assertIn("MinimumLatitude    = -56.0701234", text)
        self.assertIn("MaximumLongitude   = 137.4417154", text)
        self.assertIn("Scale              = 15161.67521207 <pixels/degree>", text)
        self.assertIn("EquatorialRadius   = 1737.4 <meters>", text)
        self.assertTrue(text.endswith("End_Group\nEnd\n"))
        # a template must never carry a raster or a Dimensions group
        self.assertNotIn("Dimensions", text)
        self.assertNotIn("TileSamples", text)

    def test_idempotent_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            grid = Path(tmp) / "PLANCKFLOOR.grid.json"
            grid.write_text(json.dumps(GRID), encoding="utf-8")
            out = Path(tmp) / "dtm_map.cub"
            self.assertEqual(tpl.main([str(grid)]), 0)
            first = out.read_text()
            self.assertEqual(tpl.main([str(grid)]), 0)          # keeps the existing file
            self.assertEqual(out.read_text(), first)
            changed = dict(GRID, clat="-54.0")
            grid.write_text(json.dumps(changed), encoding="utf-8")
            self.assertEqual(tpl.main([str(grid), "--force"]), 0)
            self.assertIn("CenterLatitude     = -54.0", out.read_text())

    def test_missing_grid_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                tpl.main([str(Path(tmp) / "nope.grid.json")])


if __name__ == "__main__":
    unittest.main()
