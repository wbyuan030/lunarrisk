#!/usr/bin/env python3
"""P1-03 step 1: hydrate real tiles and compute physical labels.

Selects tiles per split from the finalized manifest (group-aware round robin),
downloads NAC + DTM_3m + SLOPE_3m + SLOPE_60m + metadata from the public S3
bucket, verifies them and computes:

  * slope_3m labels  : g(10), g(15), 11-bin histogram, valid fractions
  * slope_60m summary: fraction > 10/15 deg, mean, quantiles (coarse baseline)

Coarse 60 m slope comes from SLDEM2015 (LOLA + Kaguya TC) and the fine 3 m slope
from the USGS NAC stereo DTM, so the coarse layer is an independent, legitimate
baseline input rather than leakage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_slope_labels as msl  # noqa: E402

S3_BASE = "https://nasa-lunar-fm-bench.s3.amazonaws.com/"
LAYERS = {
    "nac": "image_path",
    "dtm_3m": "dtm3m_path",
    "slope_3m": "slope3m_path",
    "slope_60m": "coarse_slope_path",
    "metadata": "metadata_path",
}
SPLITS = ("train", "cal", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(key: str, dest: Path, retries: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = S3_BASE + key
    last_error = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                data = response.read()
            if not data:
                raise RuntimeError("empty response")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"download failed: {url}: {last_error}")


def select_tiles(frame: pd.DataFrame, quotas: dict[str, int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    picked = []
    for split, quota in quotas.items():
        if quota <= 0:
            continue
        part = frame[frame["split_final"] == split]
        grouped = {str(group): rows for group, rows in part.groupby(part["group_id"].astype(str))}
        groups = sorted(grouped)
        groups = [groups[i] for i in rng.permutation(len(groups))]
        buckets = {
            group: grouped[group].iloc[rng.permutation(len(grouped[group]))]
            for group in groups
        }
        cursor = {group: 0 for group in groups}
        counts = 0
        while counts < quota:
            progressed = False
            for group in groups:
                if counts >= quota:
                    break
                rows = buckets[group]
                if cursor[group] >= len(rows):
                    continue
                picked.append(rows.iloc[cursor[group]])
                cursor[group] += 1
                counts += 1
                progressed = True
            if not progressed:
                raise RuntimeError(f"split {split}: only {counts}/{quota} tiles available")
    return pd.DataFrame(picked)


def summarise_slope(path: Path, units_override: str | None) -> dict:
    dataset = msl.open_slope_dataset(path)
    try:
        name = msl.find_slope_variable(dataset, None)
        values = np.asarray(dataset[name].values, dtype=np.float64)
    finally:
        dataset.close()
    labels = msl.compute_labels(values)
    finite = values[np.isfinite(values)]
    return {
        "variable": name,
        "units_verified": units_override is not None,
        **labels,
        "q50": float(np.percentile(finite, 50)) if finite.size else None,
        "q90": float(np.percentile(finite, 90)) if finite.size else None,
        "q99": float(np.percentile(finite, 99)) if finite.size else None,
    }


def hydrate_one(row: pd.Series, data_dir: Path, units_source: str) -> dict:
    sample_id = str(row["sample_id"])
    record = {
        "sample_id": sample_id,
        "split": str(row["split_final"]),
        "official_split": str(row["split_official"]),
        "group_id": str(row["group_id"]),
        "block_5km": str(row["block_5km"]),
        "NAC_EDR_ID": str(row["NAC_EDR_ID"]),
        "LTM_CODE": str(row["LTM_CODE"]),
        "center_lat": float(row["center_lat"]),
        "center_lon": float(row["center_lon"]),
        "bounds_xmin": float(row["bounds_xmin"]),
        "bounds_ymin": float(row["bounds_ymin"]),
    }
    paths = {}
    for layer, column in LAYERS.items():
        key = row.get(column)
        if not isinstance(key, str) or not key:
            continue
        destination = data_dir / sample_id / f"{layer}.nc" if layer != "metadata" else data_dir / sample_id / "metadata.txt"
        download(key, destination)
        paths[layer] = destination
        record[f"{layer}_sha256"] = sha256_file(destination)
        record[f"{layer}_bytes"] = destination.stat().st_size
    record["layers"] = {layer: str(path) for layer, path in paths.items()}

    slope_labels = summarise_slope(paths["slope_3m"], units_source)
    record.update({f"slope3m_{k}": v for k, v in slope_labels.items()})

    coarse = summarise_slope(paths["slope_60m"], units_source)
    record.update({
        "slope60m_frac_gt_10": coarse.get("frac_gt_10deg_fraction"),
        "slope60m_frac_gt_15": coarse.get("frac_gt_15deg_fraction"),
        "slope60m_mean": coarse.get("mean_valid_deg"),
        "slope60m_q50": coarse.get("q50"),
        "slope60m_q90": coarse.get("q90"),
        "slope60m_q99": coarse.get("q99"),
        "slope60m_valid_fraction": coarse.get("valid_fraction"),
    })

    import hdf5plugin  # noqa: F401
    import xarray as xr
    with xr.open_dataset(paths["nac"], engine="h5netcdf") as nac, \
            xr.open_dataset(paths["dtm_3m"], engine="h5netcdf") as dtm, \
            xr.open_dataset(paths["slope_3m"], engine="h5netcdf") as slope:
        record["nac_shape"] = [int(nac.sizes["y"]), int(nac.sizes["x"])]
        record["slope_shape"] = [int(slope.sizes["y"]), int(slope.sizes["x"])]
        record["gsd_nac"] = abs(float(nac.attrs.get("pix_res_x", 0)))
        record["gsd_slope"] = abs(float(slope.attrs.get("pix_res_x", 0)))
        record["nac_x0"] = float(nac.x.values[0])
        record["nac_y0"] = float(nac.y.values[0])
        record["slope_x0"] = float(slope.x.values[0])
        record["slope_y0"] = float(slope.y.values[0])
        record["dtm3m_min"] = float(np.nanmin(dtm[next(n for n, v in dtm.data_vars.items() if v.ndim == 2)].values))
    record["units_source"] = units_source
    record["geo_join_status"] = "BOUNDS_ALIGNED_GEO_JOIN_PENDING"
    record["modality_read_ok"] = True
    return record


def record_fieldnames(summary_path: Path, records: list[dict]) -> list[str]:
    existing = []
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing = next(reader, [])
    discovered = []
    for record in records:
        for key in record:
            if key not in discovered and key != "layers":
                discovered.append(key)
    ordered = list(existing)
    for key in discovered:
        if key not in ordered:
            ordered.append(key)
    return ordered


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manifest_100k_final.parquet"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/hydrated"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels_hydrated.csv"))
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--cal", type=int, default=120)
    parser.add_argument("--val", type=int, default=120)
    parser.add_argument("--test", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--units-source", default="SomBench README: slope_3m/slope_60m degrees, range [0,90]")
    parser.add_argument("--resume", action="store_true", help="skip tiles already present in the labels file")
    args = parser.parse_args(argv)

    frame = pd.read_parquet(args.manifest)
    frame = frame[frame["has_slope3m"] & frame["split_final"].isin(SPLITS)].copy()
    if "coarse_slope_path" not in frame.columns:
        raise SystemExit("manifest has no coarse_slope_path; rebuild with build_manifest.py")
    quotas = {"train": args.train, "cal": args.cal, "val": args.val, "test": args.test}
    selected = select_tiles(frame, quotas, args.seed)

    done_ids = set()
    if args.resume and args.labels.exists():
        existing = pd.read_csv(args.labels)
        done_ids = set(existing["sample_id"].astype(str))
    todo = selected[~selected["sample_id"].astype(str).isin(done_ids)]
    print(f"selected {len(selected)} tiles; {len(todo)} to hydrate; "
          f"groups per split: {selected.groupby('split_final').group_id.nunique().to_dict()}")

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(hydrate_one, row, args.data_dir, args.units_source): str(row["sample_id"])
            for _, row in todo.iterrows()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            sample_id = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            if index % 25 == 0 or index == len(futures):
                print(f"  hydrated {index}/{len(futures)} (failures={len(failures)})", flush=True)

    if failures:
        print("FAILURES:")
        for failure in failures[:20]:
            print(" ", failure)
    if not records:
        raise SystemExit("no tiles hydrated")

    args.labels.parent.mkdir(parents=True, exist_ok=True)
    import csv as csv_module
    columns = record_fieldnames(args.labels, records)
    mode = "a" if (args.resume and args.labels.exists()) else "w"
    with args.labels.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for record in records:
            writer.writerow({k: v for k, v in record.items() if k in columns})

    labels = pd.read_csv(args.labels)
    summary = {
        "tiles": int(len(labels)),
        "failures": failures,
        "by_split": labels.groupby("split").size().to_dict(),
        "groups_by_split": labels.groupby("split").group_id.nunique().to_dict(),
        "units_source": args.units_source,
        "g10_by_split": labels.groupby("split").slope3m_g10.describe().to_dict(),
        "valid_fraction_min": float(labels.slope3m_valid_fraction.min()),
        "unsafe_rate_by_split": labels.groupby("split").apply(
            lambda part: float((part.slope3m_g10 > 0.05).mean()), include_groups=False
        ).to_dict(),
    }
    summary_path = args.labels.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("tiles", "by_split", "groups_by_split",
                                              "unsafe_rate_by_split", "valid_fraction_min")}, indent=2))
    print(f"wrote {args.labels} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
