#!/usr/bin/env python3
"""Offline tests for the P20-03 CDR availability join (no network, no data files)."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_p20_availability",
                                              ROOT / "scripts" / "check_p20_availability.py")
check = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = check
spec.loader.exec_module(check)


def write_inputs(base: Path):
    audit = base / "audit.csv"
    pd.DataFrame([
        # strict pool members
        {"DTM_name": "AAA", "source_disjoint": True, "footprint_disjoint": True, "resolution_m": 2.0},
        {"DTM_name": "BBB", "source_disjoint": True, "footprint_disjoint": True, "resolution_m": 3.0},
        # excluded by resolution, source, footprint respectively
        {"DTM_name": "CCC", "source_disjoint": True, "footprint_disjoint": True, "resolution_m": 5.0},
        {"DTM_name": "DDD", "source_disjoint": False, "footprint_disjoint": True, "resolution_m": 2.0},
        {"DTM_name": "EEE", "source_disjoint": True, "footprint_disjoint": False, "resolution_m": 2.0},
    ]).to_csv(audit, index=False)

    requests = base / "requests.csv"
    pd.DataFrame([
        {"DTM_name": "AAA", "edr_id": "M100000001"},
        {"DTM_name": "AAA", "edr_id": "M100000002"},
        {"DTM_name": "BBB", "edr_id": "M100000003"},
        {"DTM_name": "CCC", "edr_id": "M100000004"},  # not in the pool
    ]).to_csv(requests, index=False)

    resolved = base / "resolved.csv"
    pd.DataFrame([
        {"product_id": "M100000001LC", "status": "UNVERIFIED_URL"},
        {"product_id": "M100000001RC", "status": "UNVERIFIED_URL"},
        {"product_id": "M100000002LC", "status": "UNVERIFIED_URL"},   # only left
        {"product_id": "M100000003LC", "status": "UNVERIFIED_URL"},
        {"product_id": "M100000003RC", "status": "NOT_IN_INDEX"},     # unresolved right
    ]).to_csv(resolved, index=False)
    return audit, requests, resolved


class TestAvailability(unittest.TestCase):
    def test_pool_filter_and_camera_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            audit, requests, resolved = write_inputs(base)
            out = base / "avail.json"
            code = check.main(["--audit", str(audit), "--requests", str(requests),
                               "--resolved", str(resolved), "--out", str(out)])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            # only pool scenes are reported, and CCC is outside the pool
            self.assertEqual(set(report["per_scene"]), {"AAA", "BBB"})
            self.assertEqual(report["per_scene"]["AAA"],
                             {"acquisitions": 2, "acquisitions_both_cameras": 1,
                              "acquisitions_partial": 1, "acquisitions_missing": 0})
            self.assertEqual(report["per_scene"]["BBB"],
                             {"acquisitions": 1, "acquisitions_both_cameras": 0,
                              "acquisitions_partial": 1, "acquisitions_missing": 0})
            self.assertEqual(report["scenes_with_all_acquisitions_both_cameras"], 0)
            self.assertEqual(report["acquisitions"], {"total": 3, "both": 1, "partial": 2,
                                                      "missing": 0})

    def test_scene_with_both_cameras_counts_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            audit, requests, resolved = write_inputs(base)
            resolved = base / "resolved2.csv"
            pd.DataFrame([
                {"product_id": "M100000001LC", "status": "UNVERIFIED_URL"},
                {"product_id": "M100000001RC", "status": "UNVERIFIED_URL"},
                {"product_id": "M100000002LC", "status": "UNVERIFIED_URL"},
                {"product_id": "M100000002RC", "status": "UNVERIFIED_URL"},
            ]).to_csv(resolved, index=False)
            out = base / "avail.json"
            check.main(["--audit", str(audit), "--requests", str(requests),
                        "--resolved", str(resolved), "--out", str(out)])
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["scenes_complete"], ["AAA"])


if __name__ == "__main__":
    unittest.main()
