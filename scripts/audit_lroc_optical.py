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
SCENE_PAGE = "https://data.lroc.im-ldi.com/lroc/view_rdr/{product}"
PRODUCT_PAGE = "https://data.lroc.im-ldi.com/lroc/view_rdr_product/{product}"
DATA_HOST = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001"
LINK_RE = re.compile(r"https://pds\.lroc\.im-ldi\.com/data/[^\"'<>\s]+")
SIBLING_RE = re.compile(r"/lroc/view_rdr_product/(NAC_DTM_[A-Za-z0-9_]+)")
OPTICAL_RE = re.compile(r"(MOSAIC|ORTHO|ORTHOPHOTO|NAC_IMG|_PHOTO)", re.IGNORECASE)
# used both on file names (NAC_DTM_X_SHADE.TIF) and on sibling-id suffixes (SHADE)
DERIVED_RE = re.compile(r"(SHADE|SLOPE|CLRDISC|CLRGRAD|CONF|LEGEND)", re.IGNORECASE)


def extract_links(html: str) -> list[str]:
    return sorted(set(LINK_RE.findall(html)))


def sibling_products(html: str, dtm_name: str) -> list[str]:
    """Scene pages link sibling products (orthomosaic, shade, quads) relatively."""
    base = f"NAC_DTM_{dtm_name}"
    found = {match.group(1) for match in SIBLING_RE.finditer(html)}
    return sorted(product for product in found if product != base)


def classify_siblings(products: list[str], dtm_name: str) -> dict:
    """Group sibling product ids into optical-candidate / derived / quad / other."""
    out: dict[str, list[str]] = {"optical_ids": [], "derived_ids": [], "quad_ids": [], "other_ids": []}
    base = f"NAC_DTM_{dtm_name}"
    for product in products:
        suffix = product[len(base):].strip("_")
        if DERIVED_RE.search(suffix):
            out["derived_ids"].append(product)
        elif OPTICAL_RE.search(suffix):
            out["optical_ids"].append(product)
        elif re.match(r"^E?\d+[NS]\d+", suffix):
            out["quad_ids"].append(product)
        else:
            out["other_ids"].append(product)
    return out


def head_size(url: str, timeout: int = 40) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.headers.get("Content-Length", 0))
    except Exception:  # noqa: BLE001 - a missing product is a normal outcome
        return None


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


def fetch_scene(product: str, cache_dir: Path, timeout: int = 40) -> tuple[str | None, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"scene_{product}.html"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace"), "CACHED"
    try:
        with urllib.request.urlopen(SCENE_PAGE.format(product=product), timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
        path.write_text(text, encoding="utf-8")
        return text, "FETCHED"
    except urllib.error.HTTPError as exc:
        return None, f"HTTP_{exc.code}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}"


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
    parser.add_argument("--no-head", dest="head", action="store_false", default=True,
                        help="skip HEAD verification of sibling optical products")
    parser.add_argument("--scope", choices=("passing", "all"), default="passing",
                        help="passing = source+footprint PASS only; all = every audited candidate")
    args = parser.parse_args(argv)

    audit = pd.read_csv(args.audit)
    if args.scope == "passing":
        audit = audit[audit["source_disjoint"].fillna(False) & audit["footprint_disjoint"].fillna(False)]
    names = sorted(audit["DTM_name"].astype(str))
    if args.limit:
        names = names[: args.limit]
    rows = []
    for name in names:
        scene_html, scene_status = fetch_scene(f"NAC_DTM_{name}", args.cache)
        html, status = fetch_page(f"NAC_DTM_{name}", args.cache)
        classified = (classify_products(extract_links(html), name) if html is not None
                      else {"dtm": [], "browse": [], "optical": [], "derived": [], "aux": [],
                            "has_dtm": False, "has_optical": False})
        siblings = classify_siblings(sibling_products(scene_html, name) if scene_html else [], name)
        # a sibling optical product only counts if its data file is really published
        optical_published = []
        for product in siblings["optical_ids"]:
            if not args.head:
                break
            size = head_size(f"{DATA_HOST}/DATA/SDP/NAC_DTM/{name}/{product}.IMG")
            optical_published.append(f"{product}.IMG:{size if size else 'MISSING'}")
        rows.append({
            "DTM_name": name,
            "scene_page_status": scene_status,
            "page_status": status,
            "has_dtm": classified["has_dtm"],
            "has_optical": bool(siblings["optical_ids"]) or bool(classified["optical"]),
            "optical_verified": ("yes" if any(not entry.endswith("MISSING")
                                              for entry in optical_published)
                                 else ("no" if optical_published else "not_checked")),
            "optical_files": ";".join(classified["optical"]),
            "optical_sibling_ids": ";".join(siblings["optical_ids"]),
            "optical_sibling_heads": ";".join(optical_published),
            "derived_files": ";".join(classified["derived"]),
            "dtm_files": ";".join(classified["dtm"]),
            "n_links": sum(len(v) for v in classified.values() if isinstance(v, list)),
            "n_siblings": len(siblings["optical_ids"] + siblings["derived_ids"]
                              + siblings["quad_ids"] + siblings["other_ids"]),
        })
    frame = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    report = {
        "scope": args.scope,
        "audited_scenes": int(len(frame)),
        "pages_found": int((frame["page_status"] != f"HTTP_404").sum()),
        "with_dtm_product": int(frame["has_dtm"].sum()),
        "with_optical_product": int(frame["has_optical"].sum()),
        "with_optical_sibling_id": int((frame["optical_sibling_ids"].fillna("") != "").sum()),
        "optical_verified_published": int((frame["optical_verified"] == "yes").sum()),
        "scenes_with_siblings": int((frame["n_siblings"] > 0).sum()),
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
