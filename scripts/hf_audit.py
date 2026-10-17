#!/usr/bin/env python3
"""Audit a public SomBench Hugging Face dataset, optionally downloading it.

The audit distinguishes:
  * listed files in the repository (API metadata),
  * files actually present/readable on disk after download.

It never treats Parquet index rows as downloadable assets and never creates
synthetic lunar data.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DATASETS = {
    "sample": {
        "repo_id": "nasa-ibm-ai4science/Sombench-pretraining-data",
        "allow_patterns": ["NAC_HighRes/**", "README.md"],
        "modalities": ["nac", "dtm_3m", "slope_3m", "aspect_3m"],
    },
    "crater": {
        "repo_id": "nasa-ibm-ai4science/Sombench-NAC-Crater-Detection",
        "allow_patterns": ["images/**", "annotations*.json", "README.md", "crater_catalog.parquet"],
        "modalities": [],
    },
}


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


def list_repo(repo_id: str) -> tuple[dict, list[dict]]:
    info = HfApi().dataset_info(repo_id, files_metadata=True)
    files = []
    for sibling in info.siblings:
        files.append({"path": sibling.rfilename, "bytes": sibling.size or 0})
    meta = {
        "repo_id": repo_id,
        "revision": info.sha,
        "private": bool(info.private),
        "gated": info.gated,
        "listed_files": len(files),
        "listed_bytes": sum(f["bytes"] for f in files),
        "listed_by_top_dir": dict(sorted(Counter(
            f["path"].split("/")[0] if "/" in f["path"] else "<root>" for f in files
        ).items())),
    }
    return meta, files


def scan_local(root: Path) -> dict:
    if not root.exists():
        return {"root": str(root), "exists": False, "files": 0, "bytes": 0}
    files = [
        p for p in root.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(root).parts
    ]
    return {
        "root": str(root),
        "exists": True,
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "zero_byte_files": [str(p.relative_to(root)) for p in files if p.stat().st_size == 0],
    }


def can_read_netcdf(path: Path) -> tuple[bool, str]:
    try:
        import xarray as xr
    except ImportError as exc:
        return False, f"xarray missing: {exc}"
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    errors = []
    for engine in ("h5netcdf", "netcdf4", None):
        try:
            with xr.open_dataset(path, engine=engine) as ds:
                variables = [n for n, v in ds.data_vars.items() if v.ndim == 2]
                if not variables:
                    return False, "no 2-D data variable"
                ds[variables[0]].values  # force decompression/read
                return True, ""
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{engine}: {type(exc).__name__}: {exc}")
    return False, " | ".join(errors)


def audit_tile_alignment(base: Path, rows: list[dict]) -> list[dict]:
    """Compare NAC and slope grid bounds/CRS; not a pixel-level georeference."""
    import xarray as xr
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    results = []
    for row in rows:
        try:
            nac_path = base / str(row["NAC_TILE"])
            slope_path = base / str(row["SLOPE_3M_TILE"])
            with xr.open_dataset(nac_path, engine="h5netcdf") as nac, \
                    xr.open_dataset(slope_path, engine="h5netcdf") as slope:
                dx = [
                    abs(float(nac.x.min()) - float(slope.x.min())),
                    abs(float(nac.x.max()) - float(slope.x.max())),
                ]
                dy = [
                    abs(float(nac.y.min()) - float(slope.y.min())),
                    abs(float(nac.y.max()) - float(slope.y.max())),
                ]
                nac_gsd = abs(float(nac.attrs.get("pix_res_x", 0)))
                slope_gsd = abs(float(slope.attrs.get("pix_res_x", 0)))
                crs_equal = nac.attrs.get("crs") == slope.attrs.get("crs")
                tolerance = max(nac_gsd, slope_gsd, 1.0) * 1.5
                results.append({
                    "product_id": str(row.get("PRODUCT_ID", "")),
                    "nac_gsd_m": nac_gsd,
                    "slope_gsd_m": slope_gsd,
                    "crs_equal": crs_equal,
                    "max_abs_dx_m": round(max(dx), 4),
                    "max_abs_dy_m": round(max(dy), 4),
                    "bounds_aligned_within_tolerance": bool(crs_equal and max(dx + dy) <= tolerance),
                    "tolerance_m": tolerance,
                })
        except Exception as exc:  # noqa: BLE001
            results.append({
                "product_id": str(row.get("PRODUCT_ID", "")),
                "error": f"{type(exc).__name__}: {exc}",
                "bounds_aligned_within_tolerance": False,
            })
    return results


def audit_sample(root: Path) -> dict:
    base = root / "NAC_HighRes"
    result = {
        "parquet_index_rows": None,
        "parquet_columns": [],
        "modalities": {},
        "complete_triplets": 0,
        "independent_product_ids": [],
        "dtm_source_id_known": False,
        "geo_join_status": "GEO_JOIN_PENDING",
    }
    parquet = base / "NAC_HighRes.parquet"
    rows = []
    if parquet.exists():
        try:
            import pandas as pd
            frame = pd.read_parquet(parquet)
            result["parquet_index_rows"] = int(len(frame))
            result["parquet_columns"] = list(map(str, frame.columns))
            rows = frame.to_dict("records")
        except Exception as exc:  # noqa: BLE001
            result["parquet_error"] = f"{type(exc).__name__}: {exc}"
    for modality in DATASETS["sample"]["modalities"]:
        directory = base / modality
        listed = sorted(directory.glob("*.nc")) if directory.exists() else []
        readable = 0
        errors = {}
        for path in listed:
            ok, err = can_read_netcdf(path)
            readable += int(ok)
            if not ok:
                errors[path.name] = err[:300]
        result["modalities"][modality] = {
            "downloaded": len(listed),
            "readable": readable,
            "bytes": sum(p.stat().st_size for p in listed),
            "read_errors": errors,
        }
    if rows:
        columns = set(rows[0])
        key = {
            "nac": "NAC_TILE",
            "dtm_3m": "DTM_3M_TILE",
            "slope_3m": "SLOPE_3M_TILE",
        }
        if all(value in columns for value in key.values()):
            triplets = 0
            for row in rows:
                if all((base / str(row[column])).exists() for column in key.values()):
                    triplets += 1
            result["complete_triplets"] = triplets
        if "PRODUCT_ID" in columns:
            result["independent_product_ids"] = sorted({str(v) for v in (r["PRODUCT_ID"] for r in rows)})
        if "DATASET" in columns:
            result["parquet_split_counts"] = dict(Counter(str(r["DATASET"]) for r in rows))
        result["grid_alignment"] = audit_tile_alignment(base, rows)
        result["n_bounds_aligned"] = sum(
            bool(item.get("bounds_aligned_within_tolerance"))
            for item in result["grid_alignment"]
        )
    return result


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_crater(root: Path) -> dict:
    images_dir = root / "images"
    image_files = sorted(images_dir.glob("*.npy")) if images_dir.exists() else []
    result = {
        "npy_files": len(image_files),
        "npy_bytes": sum(p.stat().st_size for p in image_files),
        "protocols": dict(Counter(
            "standard" if "__standard_" in p.name else ("large" if "__large_" in p.name else "unknown")
            for p in image_files
        )),
        "splits": {},
        "annotation_files": sorted(p.name for p in root.glob("annotations*.json")),
        "crs": None,
        "note": "Official detection split; not a site-level OOD split.",
    }
    for split in ("train", "val", "test"):
        path = root / f"annotations_min5px_{split}.json"
        variant = "min5px"
        if not path.exists():
            path = root / f"annotations_{split}.json"
            variant = "full"
        if not path.exists():
            result["splits"][split] = {"found": False}
            continue
        data = _load_json(path)
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        counts: Counter = Counter()
        missing = []
        for image in images:
            name = Path(image.get("file_name", "")).name
            counts[name] += 1
        ann_by_image: Counter = Counter()
        for ann in annotations:
            ann_by_image[ann.get("image_id")] += 1
        standard_counts = []
        for image in images:
            name = Path(image.get("file_name", "")).name
            if "__standard_" not in name:
                continue
            standard_counts.append(ann_by_image.get(image.get("id"), 0))
            if not (root / "images" / name).exists():
                missing.append(name)
        result["splits"][split] = {
            "found": True,
            "annotation_file": path.name,
            "variant": variant,
            "images": len(images),
            "standard_images": sum("__standard_" in Path(i.get("file_name", "")).name for i in images),
            "large_images": sum("__large_" in Path(i.get("file_name", "")).name for i in images),
            "annotations": len(annotations),
            "images_missing_on_disk": len(missing),
            "standard_annotation_count_median": (
                sorted(standard_counts)[len(standard_counts) // 2] if standard_counts else None
            ),
            "standard_annotation_count_min": min(standard_counts) if standard_counts else None,
            "standard_annotation_count_max": max(standard_counts) if standard_counts else None,
        }
    sample = image_files[:20]
    readable, errors = 0, {}
    for path in sample:
        try:
            import numpy as np
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 2:
                raise ValueError(f"ndim={array.ndim}")
            readable += 1
        except Exception as exc:  # noqa: BLE001
            errors[path.name] = f"{type(exc).__name__}: {exc}"
    result["npy_readable_sample"] = {"checked": len(sample), "readable": readable, "errors": errors}
    for path in image_files[:1]:
        try:
            import numpy as np
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            result["npy_shape"] = list(array.shape)
            result["npy_dtype"] = str(array.dtype)
        except Exception:
            pass
    return result


def write_outputs(dataset: str, payload: dict) -> Path:
    out = Path("outputs") / f"hf_audit_{dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--max-download-gb", type=float, default=2.0)
    parser.add_argument("--proxy", default=None, help="e.g. http://127.0.0.1:7890")
    args = parser.parse_args(argv)

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    spec = DATASETS[args.dataset]
    dest = args.dest or Path("data") / args.dataset
    repo_meta, files = list_repo(spec["repo_id"])
    selected = [f for f in files if any(
        Path(f["path"]).match(pattern) or (pattern.endswith("/**") and f["path"].startswith(pattern[:-3]))
        for pattern in spec["allow_patterns"]
    )]
    selected_bytes = sum(f["bytes"] for f in selected)

    print(f"repo={repo_meta['repo_id']} revision={repo_meta['revision']}")
    print(f"listed: {repo_meta['listed_files']} files, {human_bytes(repo_meta['listed_bytes'])}")
    print(f"selected by {spec['allow_patterns']}: {len(selected)} files, {human_bytes(selected_bytes)}")
    for top, count in repo_meta["listed_by_top_dir"].items():
        print(f"  {top}: {count} files")

    free = shutil.disk_usage(dest.parent if dest.parent.exists() else Path(".")).free
    print(f"destination: {dest} | free: {human_bytes(free)}")
    if selected_bytes > args.max_download_gb * 2**30:
        print(f"ERROR: selection exceeds --max-download-gb={args.max_download_gb}", file=sys.stderr)
        return 2
    if args.download and free - selected_bytes < 2**30:
        print("ERROR: refusing download, less than 1 GiB headroom would remain", file=sys.stderr)
        return 2

    payload = {
        "dataset": args.dataset,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "listed": repo_meta,
        "selected": {
            "allow_patterns": spec["allow_patterns"],
            "files": len(selected),
            "bytes": selected_bytes,
        },
    }
    if args.download:
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec["repo_id"], repo_type="dataset", local_dir=dest,
            allow_patterns=spec["allow_patterns"], max_workers=8,
        )
        payload["local"] = scan_local(dest)
        if args.dataset == "sample":
            payload["sample_audit"] = audit_sample(dest)
        else:
            payload["crater_audit"] = audit_crater(dest)
    else:
        payload["local"] = scan_local(dest)
        payload["note"] = "listing only; add --download to materialize files"

    out = write_outputs(args.dataset, payload)
    print(f"wrote {out}")
    if args.download:
        local = payload["local"]
        print(f"downloaded: {local['files']} files, {human_bytes(local['bytes'])}")
        if args.dataset == "sample":
            audit = payload["sample_audit"]
            print(f"parquet index rows: {audit['parquet_index_rows']}")
            for modality, info in audit["modalities"].items():
                print(f"  {modality}: downloaded={info['downloaded']} readable={info['readable']}")
            print(f"complete NAC+DTM3m+slope3m triplets: {audit['complete_triplets']}")
            print(f"independent PRODUCT_ID groups: {len(audit['independent_product_ids'])}")
        else:
            audit = payload["crater_audit"]
            print(f"npy: {audit['npy_files']} files, {human_bytes(audit['npy_bytes'])}, protocols={audit['protocols']}")
            for split, info in audit["splits"].items():
                print(f"  {split}: {info}")
        if local["zero_byte_files"]:
            print(f"WARNING: zero-byte files: {len(local['zero_byte_files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
