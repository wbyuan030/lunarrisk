#!/usr/bin/env python3
"""Offline tests for the A0 target audit (synthetic rows only)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_targets_p21",
                                              ROOT / "scripts" / "audit_targets_p21.py")
au = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = au
spec.loader.exec_module(au)


def row(sid="s1", split="train", g10="0.1", g15="0.02", rough="0.01", relief="12.0",
        group="g1", source="M1", verified="True"):
    return {"sample_id": sid, "split": split, "slope3m_g10": g10, "slope3m_g15": g15,
            "dtm3m_roughness_9m_m": rough, "dtm3m_relief_p95_p5_m": relief,
            "group_id": group, "NAC_EDR_ID": source, "slope3m_variable": "band_data",
            "units_source": "README degrees", "slope3m_units_verified": verified}


class TestParsing(unittest.TestCase):
    def test_missing_is_none_not_zero(self):
        self.assertIsNone(au.to_float(""))
        self.assertIsNone(au.to_float("nan"))
        self.assertIsNone(au.to_float("null"))
        self.assertEqual(au.to_float("0"), 0.0)

    def test_garbage_is_treated_as_missing(self):
        self.assertIsNone(au.to_float("n/a"))


class TestViolations(unittest.TestCase):
    def test_clean_rows_have_no_violations(self):
        report = au.audit([row(), row("s2", g10="0.3", g15="0.1")])
        self.assertEqual(report["violations"], [])

    def test_g15_above_g10_is_flagged(self):
        report = au.audit([row(g10="0.02", g15="0.05")])
        self.assertEqual(len(report["violations"]), 1)
        self.assertEqual(report["violations"][0]["reason"], "g15 > g10")

    def test_probability_out_of_range_is_flagged(self):
        report = au.audit([row(g10="1.2")])
        self.assertTrue(any(v["reason"] == "outside [0,1]" for v in report["violations"]))

    def test_negative_roughness_is_flagged(self):
        report = au.audit([row(rough="-0.01")])
        self.assertTrue(any(v["reason"] == "negative" and v["target"] == "dtm3m_roughness_9m_m"
                            for v in report["violations"]))


class TestAvailability(unittest.TestCase):
    def test_missing_targets_are_counted_per_split(self):
        rows = [row("a", split="train"), row("b", split="test", g15=""),
                row("c", split="test", rough="")]
        report = au.audit(rows)
        self.assertEqual(report["targets"]["slope3m_g15"]["available_per_split"]["test"], 1)
        self.assertEqual(report["targets"]["slope3m_g15"]["missing_per_split"]["test"], 1)
        self.assertEqual(report["targets"]["dtm3m_roughness_9m_m"]["missing"], 1)
        # a missing value must never be reported as a zero-valued target
        self.assertEqual(report["targets"]["slope3m_g15"]["distribution"]["min"], 0.02)


class TestSplitIntegrity(unittest.TestCase):
    def test_group_spanning_splits_is_reported(self):
        report = au.audit([row("a", split="train", group="G"), row("b", split="test", group="G")])
        self.assertEqual(report["split_integrity"]["group_spans_splits"], ["G"])

    def test_duplicate_ids_are_reported(self):
        report = au.audit([row("a"), row("a")])
        self.assertFalse(report["split_integrity"]["unique_sample_ids"])

    def test_clean_split_has_no_spans(self):
        report = au.audit([row("a", split="train", group="G1"), row("b", split="test", group="G2")])
        self.assertEqual(report["split_integrity"]["group_spans_splits"], [])


class TestLeakage(unittest.TestCase):
    def test_declared_target_in_inputs_is_flagged(self):
        findings = au.leakage_check({"COARSE": '"dtm3m_roughness_9m_m"'})
        self.assertEqual(findings[0]["target"], "dtm3m_roughness_9m_m")

    def test_project_inputs_are_clean(self):
        self.assertEqual(au.leakage_check(au.input_lists()), [])

    def test_markdown_lists_violations_and_leakage(self):
        report = au.audit([row(g10="0.02", g15="0.05")])
        report["leakage"] = []
        text = au.render_markdown(report)
        self.assertIn("g15 > g10", text)
        self.assertIn("未出现在", text)


if __name__ == "__main__":
    unittest.main()
