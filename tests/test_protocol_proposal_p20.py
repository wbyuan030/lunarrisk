#!/usr/bin/env python3
"""Invariants that keep the CDR amendment a proposal rather than an approval."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "protocols" / "p19_cdr_route_amendment_proposal.json"
ORIGINAL = ROOT / "protocols" / "p19_external_policy.json"


class TestProposalInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(PROPOSAL.read_text(encoding="utf-8"))

    def test_status_stays_unapproved(self):
        self.assertEqual(self.doc["status"], "PROPOSED_NOT_APPROVED")
        self.assertNotEqual(self.doc.get("gate3_status"), "FROZEN")
        self.assertFalse(self.doc.get("supersedes_original_record") is True)

    def test_original_policy_is_not_rewritten(self):
        original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
        text = json.dumps(original)
        self.assertIn("INSTITUTIONAL_EDR_ISIS", text)
        self.assertNotIn("PUBLIC_CDR_PDS2ISIS_SPICE_ORTHO", text)

    def test_pending_decisions_are_listed_for_the_researcher(self):
        pending = self.doc["pending_rules_requiring_approval"]
        for key in ("replacement_acquisition", "group_stratification", "coverage_gate",
                    "mosaic_policy", "sub_pixel_claim"):
            self.assertIn(key, pending)
            self.assertTrue(pending[key].strip())

    def test_evidence_matches_the_reports(self):
        evidence = self.doc["p20_evidence_2026_10_20"]
        self.assertEqual(evidence["grid_consistency"]["exact_match_fraction"], 1.0)
        gcp = evidence["crater_gcp"]
        self.assertEqual(gcp["RANGER"]["dominant_offset_px"], [0.0, 0.0])
        self.assertIn("NOT_ESTABLISHED", gcp["sub_pixel_claim"])
        self.assertEqual(evidence["independence"]["independent_groups_10km"], 51)
        self.assertEqual(evidence["availability"]["scenes_complete"], 69)


if __name__ == "__main__":
    unittest.main()
