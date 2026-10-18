#!/usr/bin/env python3
"""Resolve LROC NAC EDR identifiers against official PDS RDR observation indexes.

The LROC NAC observing-phase shapefiles (published in RDR / EXTRAS/SHAPEFILE)
contain PRODUCT_ID, PDS_VOLUME, FILE_SPECI and URL attributes. They are a
trustworthy *path discovery* input, unlike guessing volume/day directories.
The URL may still point to a retired host: --verify-head tests accessibility
without downloading any large image. This script never downloads image data.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

BASE_ID = re.compile(r"M\d{9,10}$", re.I)
CAMERA_ID = re.compile(r"(M\d{9,10})([LR])([EC])$", re.I)
# Official observation indexes publish *_LC/_RC (calibrated). READMEs and the DTM database
# use *_LE/_RE (engineering) for the same frames. A request is expanded to all four forms so
# that either product resolves; the resolved suffix is recorded as ``data_set``.
CAMERA_SUFFIXES = ("LE", "RE", "LC", "RC")
REQUIRED = ("PRODUCT_ID", "PDS_VOLUME", "FILE_SPECI", "URL")


def expected_products(base_ids: list[str], cameras: str = "both") -> set[str]:
    """Expand plain acquisition IDs into the camera product IDs the EDR indexes use."""
    cams = CAMERA_SUFFIXES if cameras == "both" else (CAMERA_SUFFIXES[0],)
    result: set[str] = set()
    for base in base_ids:
        base = base.strip().upper()
        if not BASE_ID.fullmatch(base):
            raise ValueError(f"invalid source NAC acquisition ID {base!r}")
        result.update(base + suffix for suffix in cams)
    return result


def parse_product_id(raw: object) -> str | None:
    value = str(raw or "").strip().upper()
    match = CAMERA_ID.fullmatch(value)
    if not match:
        return None
    return match.group(0)


def acq_id(product_id: str) -> str:
    """Strip the two-character camera suffix back to the acquisition id used in requests."""
    return CAMERA_ID.fullmatch(product_id.upper()).group(1)


def _read_dbf_file(file: Path):
    import shapefile
    with file.open("rb") as handle:
        reader = shapefile.Reader(dbf=handle)
        for record in reader.iterRecords():
            yield record.as_dict()


def _read_dbf_zip(file: Path):
    import shapefile
    with zipfile.ZipFile(file) as archive:
        dbfs = sorted(n for n in archive.namelist() if n.upper().endswith(".DBF"))
        if not dbfs:
            raise ValueError(f"{file} has no .DBF index")
        for member in dbfs:
            reader = shapefile.Reader(dbf=io.BytesIO(archive.read(member)))
            for record in reader.iterRecords():
                yield record.as_dict()


def _read_csv_file(file: Path):
    # CSV exports can exceed 100MB; only keep matching rows in memory.
    for chunk in pd.read_csv(file, dtype=str, chunksize=10000, keep_default_na=False):
        for record in chunk.to_dict(orient="records"):
            yield record


def iter_records(file: Path):
    suffix = file.suffix.lower()
    if suffix == ".zip":
        yield from _read_dbf_zip(file)
    elif suffix == ".dbf":
        yield from _read_dbf_file(file)
    elif suffix == ".csv":
        yield from _read_csv_file(file)
    else:
        raise ValueError(f"unsupported index format {file}; supply .zip/.dbf/.csv")


def resolve(indexes: list[Path], targets: set[str]) -> tuple[dict, dict]:
    matches: dict[str, list[dict]] = defaultdict(list)
    stats = {"index_files": [], "records_scanned": 0, "bad_index_fields": []}
    for file in indexes:
        stats["index_files"].append(str(file))
        if not file.is_file():
            raise FileNotFoundError(file)
        for row in iter_records(file):
            stats["records_scanned"] += 1
            canonical = {str(key).strip().upper(): value for key, value in row.items()}
            pid = parse_product_id(canonical.get("PRODUCT_ID"))
            if pid not in targets:
                continue
            missing = [k for k in REQUIRED if k not in canonical]
            if missing:
                stats["bad_index_fields"].append({"file": str(file), "id": pid, "missing": missing})
                continue
            matches[pid].append({
                "product_id": pid,
                "volume": str(canonical["PDS_VOLUME"] or "").strip(),
                "file_spec": str(canonical["FILE_SPECI"] or "").strip(),
                "url": str(canonical["URL"] or "").strip(),
                "index": str(file),
            })
    resolved = {}
    for pid in sorted(targets):
        candidates = matches.get(pid, [])
        # Conflicting volume/path/URL evidence must not be silently chosen.
        unique = {(r["volume"], r["file_spec"], r["url"]) for r in candidates}
        if not candidates:
            resolved[pid] = {"product_id": pid, "status": "NOT_IN_INDEX", "url": "",
                             "volume": "", "file_spec": "", "index": ""}
        elif len(unique) != 1:
            resolved[pid] = {"product_id": pid, "status": "AMBIGUOUS_INDEX", "url": "", "volume": "", "file_spec": "", "index": ";".join(sorted({r["index"] for r in candidates}))}
        else:
            row = candidates[0].copy()
            row["status"] = "UNVERIFIED_URL" if row["url"] else "NO_URL_IN_INDEX"
            row["data_set"] = "EDR" if row["product_id"].upper().endswith(("LE", "RE")) else "CDR"
            resolved[pid] = row
    return resolved, stats


def verify_head(url: str, timeout: float = 15.0) -> tuple[str, str, int | None]:
    if not url or urlparse(url).scheme not in ("http", "https"):
        return "NO_HTTP_URL", "", None
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "LunarRisk-EDR-metadata-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            final = response.url
            content_type = response.headers.get("Content-Type", "").lower()
            content_len = response.headers.get("Content-Length", "")
            if "text/html" in content_type or not urlparse(final).path.upper().endswith(".IMG"):
                return "HTML_OR_BROWSER_REDIRECT", final, int(content_len) if content_len.isdigit() else None
            if status != 200:
                return f"HTTP_{status}", final, None
            return "HEAD_BINARY_CANDIDATE", final, int(content_len) if content_len.isdigit() else None
    except urllib.error.HTTPError as exc:
        return f"HTTP_{exc.code}", url, None
    except Exception as exc:
        return f"NETWORK_ERROR:{type(exc).__name__}", url, None


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=Path("outputs/p19_edr_acquisition_list.csv"))
    parser.add_argument("--index", type=Path, action="append", required=True,
                        help="Official NAC_* observation index .ZIP, .DBF or .csv (repeat for phases/latitudes)")
    parser.add_argument("--out", type=Path, default=Path("outputs/p19_edr_resolved_urls.csv"))
    parser.add_argument("--verify-head", action="store_true", help="HEAD only, never download IMG")
    parser.add_argument("--max-head", type=int, default=5, help="maximum URL checks for initial probe (0=all)")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.requests, dtype=str)
    if "edr_id" not in frame.columns:
        parser.error("--requests must contain column edr_id")
    expected = expected_products(frame.edr_id.dropna().unique().tolist())
    resolved, stats = resolve(args.index, expected)
    output = [resolved[key] for key in sorted(resolved)]
    if args.verify_head:
        candidates = [row for row in output if row.get("url") and row["status"] == "UNVERIFIED_URL"]
        candidates = candidates[:args.max_head] if args.max_head else candidates
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(verify_head, row["url"], args.timeout): row for row in candidates}
            for future in as_completed(futures):
                row = futures[future]
                row["status"], row["final_url"], row["content_length"] = future.result()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(args.out, index=False)
    counts = pd.DataFrame(output).status.value_counts().to_dict()
    summary = {"expected_camera_products": len(expected), "status_counts": counts,
               "found_urls": sum(bool(row.get("url")) for row in output), **stats,
               "verified_binary_candidates": sum(row["status"] == "HEAD_BINARY_CANDIDATE" for row in output),
               "warning": "HEAD_BINARY_CANDIDATE is not file-content verified; run Range/IMG+LBL validation before bulk retrieval."}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["found_urls"] else 2


if __name__ == "__main__":
    raise SystemExit(run())
