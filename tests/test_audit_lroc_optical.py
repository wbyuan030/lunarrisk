#!/usr/bin/env python3
"""Offline tests for the LROC optical-availability audit (cached pages only)."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_lroc_optical", ROOT / "scripts" / "audit_lroc_optical.py")
optical = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = optical
spec.loader.exec_module(optical)

BASE = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001"
DTM_ONLY_HTML = f'''<html><body>
<a href="{BASE}/DATA/SDP/NAC_DTM/ALPHA/NAC_DTM_ALPHA.TIF">dtm</a>
<a href="{BASE}/DATA/SDP/NAC_DTM/ALPHA/NAC_DTM_ALPHA.LBL">label</a>
<a href="{BASE}/EXTRAS/BROWSE/NAC_DTM/ALPHA/NAC_DTM_ALPHA.PYR.TIF">browse</a>
</body></html>'''
FULL_HTML = f'''<html><body>
<a href="{BASE}/DATA/SDP/NAC_DTM/BETA/NAC_DTM_BETA.TIF">dtm</a>
<a href="{BASE}/DATA/SDP/NAC_DTM/BETA/NAC_DTM_BETA.LBL">label</a>
<a href="{BASE}/DATA/SDP/NAC_DTM/BETA/NAC_DTM_BETA_MOSAIC_200CM.IMG">mosaic</a>
<a href="{BASE}/DATA/SDP/NAC_DTM/BETA/NAC_DTM_BETA_SHADE.TIF">shade</a>
<a href="{BASE}/DATA/SDP/NAC_DTM/BETA/NAC_DTM_BETA_SLOPE.TIF">slope</a>
<a href="{BASE}/EXTRAS/BROWSE/NAC_DTM/BETA/NAC_DTM_BETA.PYR.TIF">browse</a>
<a href="{BASE}/DATA/SDP/NAC_DTM/BETA/README.TXT">readme</a>
</body></html>'''


SCENE_DTM_ONLY = '<html><body><a href="/lroc/view_rdr_product/NAC_DTM_ALPHA">self</a></body></html>'
SCENE_WITH_SIBLINGS = ('<html><body>'
                       '<a href="/lroc/view_rdr_product/NAC_DTM_BETA">self</a>'
                       '<a href="/lroc/view_rdr_product/NAC_DTM_BETA_MOSAIC_200CM">mosaic</a>'
                       '<a href="/lroc/view_rdr_product/NAC_DTM_BETA_SHADE">shade</a>'
                       '<a href="/lroc/view_rdr_product/NAC_DTM_BETA_E763N0085">quad</a>'
                       '</body></html>')


class TestSiblingDiscovery(unittest.TestCase):
    def test_sibling_products_exclude_the_base_id(self):
        products = optical.sibling_products(SCENE_WITH_SIBLINGS, "BETA")
        self.assertEqual(products, ["NAC_DTM_BETA_E763N0085", "NAC_DTM_BETA_MOSAIC_200CM",
                                    "NAC_DTM_BETA_SHADE"])

    def test_sibling_classification(self):
        groups = optical.classify_siblings(
            optical.sibling_products(SCENE_WITH_SIBLINGS, "BETA"), "BETA")
        self.assertEqual(groups["optical_ids"], ["NAC_DTM_BETA_MOSAIC_200CM"])
        self.assertEqual(groups["derived_ids"], ["NAC_DTM_BETA_SHADE"])
        self.assertEqual(groups["quad_ids"], ["NAC_DTM_BETA_E763N0085"])
        self.assertEqual(optical.classify_siblings([], "ALPHA")["optical_ids"], [])


class TestLinkHandling(unittest.TestCase):
    def test_links_are_extracted_deduplicated_and_sorted(self):
        html = FULL_HTML + FULL_HTML.replace("BETA", "BETA")
        links = optical.extract_links(html)
        self.assertEqual(len(links), len(set(links)))
        self.assertEqual(links, sorted(links))
        self.assertTrue(all(link.startswith("https://pds.lroc.im-ldi.com/data/") for link in links))


class TestClassification(unittest.TestCase):
    def test_dtm_only_scene_has_no_optical(self):
        result = optical.classify_products(optical.extract_links(DTM_ONLY_HTML), "ALPHA")
        self.assertTrue(result["has_dtm"])
        self.assertFalse(result["has_optical"])
        self.assertEqual(result["browse"], ["NAC_DTM_ALPHA.PYR.TIF"])
        self.assertEqual(result["dtm"], ["NAC_DTM_ALPHA.LBL", "NAC_DTM_ALPHA.TIF"])

    def test_orthomosaic_scene_is_detected(self):
        result = optical.classify_products(optical.extract_links(FULL_HTML), "BETA")
        self.assertTrue(result["has_optical"])
        self.assertEqual(result["optical"], ["NAC_DTM_BETA_MOSAIC_200CM.IMG"])
        # DTM-derived products must never be reported as optical input
        self.assertEqual(result["derived"], ["NAC_DTM_BETA_SHADE.TIF", "NAC_DTM_BETA_SLOPE.TIF"])
        self.assertNotIn("NAC_DTM_BETA.TIF", result["optical"])
        self.assertEqual(result["aux"], ["README.TXT"])


class TestMainWithCache(unittest.TestCase):
    def test_summary_counts_from_cached_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cache = base / "pages"
            cache.mkdir()
            (cache / "NAC_DTM_ALPHA.html").write_text(DTM_ONLY_HTML, encoding="utf-8")
            (cache / "NAC_DTM_BETA.html").write_text(FULL_HTML, encoding="utf-8")
            (cache / "NAC_DTM_GAMMA.html").write_text("<html>no links here</html>", encoding="utf-8")
            (cache / "scene_NAC_DTM_ALPHA.html").write_text(SCENE_DTM_ONLY, encoding="utf-8")
            (cache / "scene_NAC_DTM_BETA.html").write_text(SCENE_WITH_SIBLINGS, encoding="utf-8")
            (cache / "scene_NAC_DTM_GAMMA.html").write_text(SCENE_DTM_ONLY, encoding="utf-8")
            (cache / "scene_NAC_DTM_DELTA.html").write_text(SCENE_DTM_ONLY, encoding="utf-8")
            audit_csv = base / "audit.csv"
            pd.DataFrame([
                {"DTM_name": "ALPHA", "source_disjoint": True, "footprint_disjoint": True},
                {"DTM_name": "BETA", "source_disjoint": True, "footprint_disjoint": True},
                {"DTM_name": "GAMMA", "source_disjoint": True, "footprint_disjoint": True},
                {"DTM_name": "DELTA", "source_disjoint": True, "footprint_disjoint": False},
            ]).to_csv(audit_csv, index=False)
            out_csv, out_json = base / "out.csv", base / "out.json"
            code = optical.main(["--audit", str(audit_csv), "--cache", str(cache), "--scope", "passing",
                                 "--no-head", "--out-csv", str(out_csv), "--out-json", str(out_json)])
            self.assertEqual(code, 0)
            frame = pd.read_csv(out_csv).set_index("DTM_name")
            self.assertEqual(set(frame.index), {"ALPHA", "BETA", "GAMMA"})  # DELTA excluded
            self.assertTrue(bool(frame.loc["BETA", "has_optical"]))
            self.assertIn("NAC_DTM_BETA_MOSAIC_200CM", frame.loc["BETA", "optical_sibling_ids"])
            self.assertEqual(frame.loc["BETA", "optical_verified"], "not_checked")
            self.assertFalse(bool(frame.loc["ALPHA", "has_optical"]))
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["audited_scenes"], 3)
            self.assertEqual(report["scope"], "passing")

            code = optical.main(["--audit", str(audit_csv), "--cache", str(cache), "--scope", "all",
                                 "--no-head", "--out-csv", str(base / "all.csv"),
                                 "--out-json", str(base / "all.json")])
            self.assertEqual(code, 0)
            all_report = json.loads((base / "all.json").read_text(encoding="utf-8"))
            self.assertEqual(all_report["audited_scenes"], 4)  # includes the footprint-failing scene
            self.assertEqual(report["with_optical_product"], 1)
            self.assertEqual(report["with_dtm_product"], 2)


if __name__ == "__main__":
    unittest.main()
