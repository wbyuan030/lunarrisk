#!/usr/bin/env python3
"""Offline tests for the LROC CDR preparation helpers (no ISIS, no SPICE calls)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = load("prepare_lroc_cdr_cube", ROOT / "scripts" / "prepare_lroc_cdr_cube.py")
ckdb = load("build_lro_ck_db", ROOT / "scripts" / "build_lro_ck_db.py")

LABEL = b"""PDS_VERSION_ID                     = PDS3
/* FILE CHARACTERISTICS */
RECORD_BYTES                       = 5064
FRAME_ID                           = LEFT
START_TIME                         = 2011-03-01T17:30:05.040
STOP_TIME                          = 2011-03-01T17:30:22.750
LRO:SPACECRAFT_CLOCK_PREROLL_COUNT = "1/320693404:44564"
SPACECRAFT_CLOCK_START_COUNT       = "1/320693405:01785"
SPACECRAFT_CLOCK_STOP_COUNT        = "1/320693422:48326"
LINE_EXPOSURE_DURATION             = 0.337600 <ms>
CROSSTRACK_SUMMING                 = 1
LRO:TEMPERATURE_SCS                = 11.62 <degC>
LRO:TEMPERATURE_FPA                = 21.21 <degC>
LRO:TEMPERATURE_FPGA               = -5.93 <degC>
LRO:TEMPERATURE_TELESCOPE          = 18.37 <degC>
END
"""


class TestLabelValues(unittest.TestCase):
    def test_isis_named_values_are_extracted(self):
        values = prep.label_values(LABEL)
        self.assertEqual(values["FrameId"], "LEFT")
        self.assertEqual(values["SpacecraftClockStartCount"], "1/320693405:01785")
        self.assertEqual(values["SpacecraftClockPrerollCount"], "1/320693404:44564")
        self.assertEqual(values["LineExposureDuration"], "0.337600")  # unit stripped
        self.assertEqual(values["SpatialSumming"], "1")
        self.assertEqual(values["StartTime"], "2011-03-01T17:30:05.040")

    def test_absent_keys_are_simply_omitted(self):
        values = prep.label_values(b"PDS_VERSION_ID = PDS3\nEND\n")
        self.assertEqual(values, {})

    def test_frame_codes_match_the_camera_models(self):
        self.assertEqual(prep.FRAME_CODES["LEFT"], -85600)
        self.assertEqual(prep.FRAME_CODES["RIGHT"], -85610)


class TestKernelDbTimestamps(unittest.TestCase):
    def test_doy_to_date_and_isis_format(self):
        import datetime as dt

        moment = ckdb.doy_to_date(2011, 60)
        self.assertEqual(moment, dt.datetime(2011, 3, 1))
        self.assertEqual(ckdb.isis_time(moment), "2011 MAR 01 00:00:00.000000 TDB")

    def test_filename_bounds_add_a_margin(self):
        bounds = ckdb.filename_bounds(Path("moc42r_2011031_2011060_v05.bc"))
        self.assertEqual(bounds, ("2011 JAN 30 00:00:00.000000 TDB",
                                  "2011 MAR 02 00:00:00.000000 TDB"))
        self.assertIsNone(ckdb.filename_bounds(Path("not-a-kernel.bc")))


if __name__ == "__main__":
    unittest.main()
