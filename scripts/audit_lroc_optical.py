#!/usr/bin/env python3
"""P20 pre-flight: which audited LROC DTM scenes publish optical imagery at all?

The PDS 5-RDR bundle ships only the DTM (`.TIF`/`.LBL`/browse) for most scenes; a subset
(landing-site style products) additionally publishes a NAC-derived orthomosaic
(``*_MOSAIC_200CM.IMG``). Raw EDR volumes are not published on the mirrors we can reach, so
the per-scene product page of the LROC im-ldi archive is the authoritative list of what is
actually downloadable.

This script only reads product pages (cached locally) and classifies their file links; it
never downloads imagery and never reports model results.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_PAGE = "https://data.lroc.im-ldi.com/lroc/view_rdr_product/{product}"
LINK_RE = re.compile(r"https://pds\.lroc\.im-ldi\.com/data/[^\"'<>\s]+")
OPTICAL_RE = re.compile(r"(MOSAIC|ORTHO|ORTHOPHOTO|NAC_IMG|_PHOTO)", re.IGNORECASE)
DERIVED_RE = re.compile(r"(_SHADE|_SLOPE|_CLRDISC|_CLRGRAD|_CONF|_LEGEND)", re.IGNORECASE)


def extract_links(html: str) -> list[str]:
    return sorted(set(LINK_RE.findall(html)))


def classify_products(links: list[str], dtm_name: str) -> dict:
    """Split published files into DTM / browse / optical / derived / auxiliary."""
    out: dict[str, list[str]] = {"dtm": [], "browse": [], "optical": [], "derived": [], "aux": []}
    for url in links:
        path = url.split("/data/", 1)[-1]
        filename = path.rsplit("/", 1)[-1]
        if not filename:
            continue
        if "EXTRAS/BROWSE/" in path or filename.endswith(".PYR.TIF"):
            out["browse"].append(filename)
        elif DERIVED_RE.search(filename):
            out["derived"].append(filename)
        elif OPTICAL_RE.search(filename):
            out["optical"].append(filename)
        elif filename.upper().startswith(f"NAC_DTM_{dtm_name.upper()}") and \
                filename.upper().endswith((".TIF", ".LBL", ".IMG")):
            out["dtm"].append(filename)
        else:
            out["aux"].append(filename)
    for key in out:
        out[key] = sorted(set(out[key]))
    out["has_optical"] = bool(out["optical"])
    out["has_dtm"] = bool(out["dtm"])
    return out


def fetch_page(product: str, cache_dir: Path, timeout: int = 40) -> tuple[str | None, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{product}.html"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace"), "CACHED"
    try:
        with urllib.request.urlopen(PRODUCT_PAGE.format(product=product), timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
        path.write_text(text, encoding="utf-8")
        return text, "FETCHED"
    except urllib.error.HTTPError as exc:
        return None, f"HTTP_{exc.code}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path,
                        default=Path("outputs/p19_lroc_footprint_audit.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/lroc_product_pages"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/p19_optical_audit.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/p19_optical_audit.json"))
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N products")
    args = parser.parse_args(argv)

    audit = pd.read_csv(args.audit)
    audit = audit[audit["source_disjoint"].fillna(False) & audit["footprint_disjoint"].fillna(False)]
    names = sorted(audit["DTM_name"].astype(str))
    if args.limit:
        names = names[: args.limit]
    rows = []
    for name in names:
        html, status = fetch_page(f"NAC_DTM_{name}", args.cache)
        if html is None:
            rows.append({"DTM_name": name, "page_status": status, "has_dtm": False,
                         "has_optical": False, "optical_files": "", "n_links": 0})
            continue
        classified = classify_products(extract_links(html), name)
        rows.append({
            "DTM_name": name,
            "page_status": status,
            "has_dtm": classified["has_dtm"],
            "has_optical": classified["has_optical"],
            "optical_files": ";".join(classified["optical"]),
            "dtm_files": ";".join(classified["dtm"]),
            "derived_files": ";".join(classified["derived"]),
            "n_links": sum(len(v) for v in classified.values() if isinstance(v, list)),
        })
    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    report = {
        "audited_scenes": int(len(frame)),
        "pages_found": int((frame["page_status"] != f"HTTP_404").sum()),
        "with_dtm_product": int(frame["has_dtm"].sum()),
        "with_optical_product": int(frame["has_optical"].sum()),
        "optical_examples": frame.loc[frame["has_optical"], ["DTM_name", "optical_files"]]
        .head(15).values.tolist(),
        "notes": [
            "Product pages are the authoritative list of downloadable files on this mirror.",
            "Raw NAC EDR volumes returned 404 on every mirror path tried; the published "
            "optical product is a NAC-derived site orthomosaic (*_MOSAIC_200CM.IMG).",
            "_SHADE/_SLOPE products are DTM-derived and are not independent optical input.",
            "Only scenes with an optical product can serve as frozen external samples.",
        ],
    }
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:2500])
    print(f"wrote {args.out_csv} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
