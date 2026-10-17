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
import hashlib
import json
import os
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


def _first_moment_sums(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum of z*x and z*y over every 3x3 window (x, y are the {-1,0,1} offsets)."""
    column_sums = array[:-2, :] + array[1:-1, :] + array[2:, :]   # shape (rows, all columns)
    row_sums = array[:, :-2] + array[:, 1:-1] + array[:, 2:]      # shape (all rows, cols)
    return column_sums[:, 2:] - column_sums[:, :-2], row_sums[2:, :] - row_sums[:-2, :]


def _window_sum(array: np.ndarray) -> np.ndarray:
    """Sum over every 3x3 window."""
    rows, cols = array.shape[0] - 2, array.shape[1] - 2
    out = np.zeros((rows, cols), dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            out += array[di:di + rows, dj:dj + cols]
    return out


def planar_roughness_rms(elevation: np.ndarray) -> float:
    """Mean planar-detrended RMS of a 3x3 window, in metres.

    For offsets x, y in {-1, 0, 1} the least-squares plane over the window has
    a = mean(z), b = Sx/6, c = Sy/6, so the residual sum of squares is
    ``sum(z^2) - sum(z)^2/9 - (Sx^2 + Sy^2)/6``. NaN pixels are filled with the
    tile median before differencing.
    """
    values = np.asarray(elevation, dtype=np.float64)
    if values.shape[0] < 3 or values.shape[1] < 3:
        return float("nan")
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    if not finite.all():
        values = np.where(finite, values, float(np.nanmedian(values)))
    total = _window_sum(values)
    total_sq = _window_sum(values * values)
    sx, sy = _first_moment_sums(values)
    residual = total_sq - total ** 2 / 9.0 - (sx ** 2 + sy ** 2) / 6.0
    rms = np.sqrt(np.clip(residual, 0.0, None) / 9.0)
    return float(rms.mean())


def elevation_relief_p95_p5(elevation: np.ndarray) -> float:
    """p95 - p5 of the tile elevation, in metres."""
    values = np.asarray(elevation, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, 95) - np.percentile(finite, 5))


def hydrate_one(row: pd.Series, data_dir: Path, units_source: str, download_fn=None) -> dict:
    download_fn = download_fn or download
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
        "incidence_angle_deg": float(row.get("incidence_angle_deg", float("nan"))),
        "emission_angle_deg": float(row.get("emission_angle_deg", float("nan"))),
        "phase_angle_deg": float(row.get("phase_angle_deg", float("nan"))),
        "sub_solar_ground_azimuth_deg": float(row.get("sub_solar_ground_azimuth_deg", float("nan"))),
        "bounds_xmin": float(row["bounds_xmin"]),
        "bounds_ymin": float(row["bounds_ymin"]),
    }
    paths = {}
    for layer, column in LAYERS.items():
        key = row.get(column)
        if not isinstance(key, str) or not key:
            continue
        destination = data_dir / sample_id / f"{layer}.nc" if layer != "metadata" else data_dir / sample_id / "metadata.txt"
        download_fn(key, destination)
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
        dtm_var = next(n for n, v in dtm.data_vars.items() if v.ndim == 2)
        dtm_values = np.asarray(dtm[dtm_var].values, dtype=np.float64)
        record["dtm3m_min"] = float(np.nanmin(dtm_values))
        record["dtm3m_relief_p95_p5_m"] = elevation_relief_p95_p5(dtm_values)
        record["dtm3m_roughness_9m_m"] = planar_roughness_rms(dtm_values)
        record["dtm3m_valid_fraction"] = float(np.isfinite(dtm_values).mean())
    record["units_source"] = units_source
    record["geo_join_status"] = "BOUNDS_ALIGNED_GEO_JOIN_PENDING"
    record["modality_read_ok"] = True
    return record


def read_elevation(path: Path) -> np.ndarray:
    import hdf5plugin  # noqa: F401
    import h5netcdf  # noqa: F401
    import xarray as xr
    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        var = next(n for n, v in dataset.data_vars.items() if v.ndim == 2)
        return np.asarray(dataset[var].values, dtype=np.float64)


def enrich_existing(labels_path: Path, manifest_path: Path, data_dir: Path,
                    read_elevation_fn=None) -> int:
    """Backfill geometry and DTM-derived attributes; one-to-one merge, atomic write-back."""
    read_elevation_fn = read_elevation_fn or read_elevation
    labels = pd.read_csv(labels_path)
    if not labels["sample_id"].astype(str).is_unique:
        raise HydrationError("labels file has duplicate sample_id; refusing to enrich")
    geometry_columns = ["incidence_angle_deg", "emission_angle_deg", "phase_angle_deg",
                        "sub_solar_ground_azimuth_deg"]
    manifest = pd.read_parquet(manifest_path, columns=["sample_id", *geometry_columns])
    if not manifest["sample_id"].astype(str).is_unique:
        raise HydrationError("manifest has duplicate sample_id; refusing a non one-to-one merge")
    before_ids = labels["sample_id"].astype(str).reset_index(drop=True)
    labels = labels.drop(columns=[c for c in labels.columns if c in geometry_columns], errors="ignore")
    labels = labels.merge(manifest, on="sample_id", how="left", validate="one_to_one")
    if len(labels) != len(before_ids) or not labels["sample_id"].astype(str).reset_index(drop=True).equals(before_ids):
        raise HydrationError("enrichment merge changed the row set; refusing to write")

    relief, roughness, valid = [], [], []
    for sample_id in labels["sample_id"].astype(str):
        path = data_dir / sample_id / "dtm_3m.nc"
        if not path.exists():
            relief.append(np.nan), roughness.append(np.nan), valid.append(np.nan)
            continue
        elevation = read_elevation_fn(path)
        relief.append(elevation_relief_p95_p5(elevation))
        roughness.append(planar_roughness_rms(elevation))
        valid.append(float(np.isfinite(elevation).mean()))
    labels["dtm3m_relief_p95_p5_m"] = relief
    labels["dtm3m_roughness_9m_m"] = roughness
    labels["dtm3m_valid_fraction"] = valid
    # Atomic replace: an interrupted enrich must not damage the existing labels file.
    atomic_write_csv(labels_path, labels)
    print(f"enriched {labels_path}: "
          f"geometry_rows={int(labels['incidence_angle_deg'].notna().sum())}, "
          f"roughness_rows={int(np.isfinite(labels['dtm3m_roughness_9m_m']).sum())}, "
          f"relief_rows={int(np.isfinite(labels['dtm3m_relief_p95_p5_m']).sum())}")
    return 0


class HydrationError(RuntimeError):
    """Raised when a hydration run would corrupt or silently change the cohort."""


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV through a temp file, fsync it and atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def merge_records(existing: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Append-only merge with column-union schema evolution.

    Existing column order is preserved and newly discovered columns are appended in
    first-seen order; older rows get NaN for them, so late-added fields (geometry,
    DTM attributes) never shift or overwrite previously written values. Duplicate
    ``sample_id`` values are refused because they would silently multiply rows.
    """
    new = pd.DataFrame(records)
    if new.empty:
        return existing
    if "sample_id" not in new.columns:
        raise HydrationError("hydrated records are missing sample_id")
    if existing.empty:
        return new
    columns = list(existing.columns) + [c for c in new.columns if c not in existing.columns]
    merged = pd.concat([existing.reindex(columns=columns), new.reindex(columns=columns)],
                       ignore_index=True)
    duplicated = merged["sample_id"].astype(str)
    duplicated = duplicated[duplicated.duplicated()]
    if not duplicated.empty:
        raise HydrationError(f"duplicate sample_id after merge: {sorted(set(duplicated))[:5]}")
    return merged


# Which labels column must still agree with which manifest column for an existing row.
MANIFEST_FIELD_FOR_LABEL = {
    "split": "split_final",
    "official_split": "split_official",
    "group_id": "group_id",
    "NAC_EDR_ID": "NAC_EDR_ID",
    "LTM_CODE": "LTM_CODE",
    "block_5km": "block_5km",
}
REQUIRED_SOURCE_FILES = ("nac.nc", "slope_3m.nc")


def assert_existing_consistency(existing: pd.DataFrame, selection: pd.DataFrame,
                                data_dir: Path, require_sources: bool = True) -> dict:
    """Fail fast when already-hydrated rows drifted or fall outside the frozen selection.

    The selection is prefix-stable, so growing a quota must keep every existing tile and
    in the same split. Anything else is a cohort change and is refused rather than
    silently rewritten.
    """
    if existing.empty:
        return {"checked": 0, "unknown_ids": [], "missing_sources": [], "missing_columns": []}
    existing_ids_series = existing["sample_id"].astype(str)
    duplicated = existing_ids_series[existing_ids_series.duplicated()]
    if not duplicated.empty:
        raise HydrationError(
            f"labels file has duplicate sample_id rows: {sorted(set(duplicated))[:5]}; "
            "refusing to guess which row is authoritative")
    selection_ids = selection["sample_id"].astype(str)
    unknown = sorted(set(existing["sample_id"].astype(str)) - set(selection_ids))
    if unknown:
        raise HydrationError(
            f"{len(unknown)} already-hydrated sample_id(s) are not part of the current selection "
            f"(the old cohort must be a subset of the new one): {unknown[:5]}. "
            "Keep the previous quotas, or remove those rows explicitly before extending."
        )
    manifest = selection.set_index(selection_ids)
    manifest.index.name = "sample_id"
    drifted = []
    for row in existing.itertuples():
        sample_id = str(row.sample_id)
        target = manifest.loc[sample_id]
        for label_field, manifest_field in MANIFEST_FIELD_FOR_LABEL.items():
            if label_field not in existing.columns or manifest_field not in selection.columns:
                continue
            if str(getattr(row, label_field)) != str(target[manifest_field]):
                drifted.append({"sample_id": sample_id, "field": label_field,
                                "labels": str(getattr(row, label_field)),
                                "manifest": str(target[manifest_field])})
    if drifted:
        raise HydrationError(f"already-hydrated rows disagree with the manifest: {drifted[:5]}")
    missing_sources = []
    if require_sources:
        for sample_id in existing["sample_id"].astype(str):
            directory = data_dir / sample_id
            if any(not (directory / name).exists() for name in REQUIRED_SOURCE_FILES):
                missing_sources.append(sample_id)
    return {
        "checked": int(len(existing)),
        "unknown_ids": [],
        "missing_sources": missing_sources,
        "missing_columns": [c for c in MANIFEST_FIELD_FOR_LABEL if c not in existing.columns],
    }


def build_selection(manifest_path: Path, quotas: dict[str, int], seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(manifest_path)
    frame = frame[frame["has_slope3m"] & frame["split_final"].isin(SPLITS)].copy()
    if "coarse_slope_path" not in frame.columns:
        raise HydrationError("manifest has no coarse_slope_path; rebuild with build_manifest.py")
    return select_tiles(frame, quotas, seed)


def hydrate_records(todo: pd.DataFrame, args, download_fn, hydrate_fn) -> tuple[list[dict], list[dict]]:
    """Hydrate the pending tiles in a thread pool; return (records, failures)."""
    records, failures = [], []
    if todo.empty:
        return records, failures
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(hydrate_fn, row, args.data_dir, args.units_source, download_fn):
                str(row["sample_id"])
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
    return records, failures


def summarise_labels(labels: pd.DataFrame, failures: list[dict], units_source: str) -> dict:
    return {
        "tiles": int(len(labels)),
        "failures": failures,
        "by_split": {k: int(v) for k, v in labels.groupby("split").size().items()},
        "groups_by_split": {k: int(v) for k, v in labels.groupby("split").group_id.nunique().items()},
        "units_source": units_source,
        "g10_by_split": labels.groupby("split").slope3m_g10.describe().to_dict(),
        "valid_fraction_min": float(labels.slope3m_valid_fraction.min()),
        "unsafe_rate_by_split": labels.groupby("split").apply(
            lambda part: float((part.slope3m_g10 > 0.05).mean()), include_groups=False
        ).to_dict(),
    }


def run_hydration(args, *, download_fn=None, hydrate_fn=None) -> tuple[int, dict]:
    """Execute one hydration pass. Returns (exit_code, stats).

    Exit codes: 0 = nothing left to do *or* every pending tile succeeded;
    2 = at least one pending tile failed (failures are always listed, and a run where
    every request failed never reports success).
    """
    download_fn = download_fn or download
    hydrate_fn = hydrate_fn or hydrate_one
    quotas = {"train": args.train, "cal": args.cal, "val": args.val, "test": args.test}
    selected = build_selection(args.manifest, quotas, args.seed)
    existing = pd.read_csv(args.labels) if args.labels.exists() else pd.DataFrame()
    audit = assert_existing_consistency(existing, selected, args.data_dir)
    existing_ids = set(existing["sample_id"].astype(str)) if not existing.empty else set()
    todo = selected[~selected["sample_id"].astype(str).isin(existing_ids)]
    stats = {
        "selected_count": int(len(selected)),
        "existing_count": int(len(existing)),
        "todo_count": int(len(todo)),
        "download_attempted": 0,
        "new_records": 0,
        "failures": [],
        "existing_audit": audit,
        "selected_groups_by_split": {k: int(v) for k, v in
                                     selected.groupby("split_final").group_id.nunique().items()},
        "units_source": args.units_source,
        "labels_written": False,
    }
    if audit["missing_sources"]:
        raise HydrationError(
            f"{len(audit['missing_sources'])} hydrated tile(s) lost their local source files: "
            f"{audit['missing_sources'][:5]}; refusing to continue"
        )
    if todo.empty:
        print(f"nothing to hydrate: all {len(selected)} selected tiles are already in "
              f"{args.labels} ({len(existing)} rows)")
        labels = existing
    else:
        print(f"selected {len(selected)} tiles; {len(todo)} to hydrate; "
              f"groups per split: {stats['selected_groups_by_split']}")
        stats["download_attempted"] = int(len(todo))
        records, failures = hydrate_records(todo, args, download_fn, hydrate_fn)
        stats["new_records"] = len(records)
        stats["failures"] = failures
        requested = set(todo["sample_id"].astype(str))
        unexpected = [str(r.get("sample_id")) for r in records if str(r.get("sample_id")) not in requested]
        if unexpected:
            raise HydrationError(f"hydrated records outside the requested selection: {unexpected[:5]}")
        labels = merge_records(existing, records) if records else existing
        if records:
            atomic_write_csv(args.labels, labels)
            stats["labels_written"] = True
    if labels.empty:
        raise HydrationError("no labels available and nothing was hydrated")
    summary = summarise_labels(labels, stats["failures"], args.units_source)
    summary.update({k: stats[k] for k in ("selected_count", "existing_count", "todo_count",
                                          "download_attempted", "new_records", "labels_written")})
    args.labels.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("tiles", "by_split", "groups_by_split",
                                              "selected_count", "existing_count", "todo_count",
                                              "download_attempted", "new_records")}, indent=2))
    if stats["failures"]:
        print("FAILURES:", file=sys.stderr)
        for failure in stats["failures"][:20]:
            print(" ", failure, file=sys.stderr)
        return 2, stats
    print(f"labels: {args.labels} ({summary['tiles']} rows); summary: {args.labels.with_suffix('.summary.json')}")
    return 0, stats


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
    parser.add_argument("--resume", action="store_true", help="reuse and extend the labels file")
    parser.add_argument("--enrich", action="store_true",
                        help="only backfill geometry/DTM attributes onto the existing labels file")
    args = parser.parse_args(argv)

    if args.enrich:
        if not args.labels.exists():
            print(f"ERROR: --enrich needs an existing labels file: {args.labels}", file=sys.stderr)
            return 2
        try:
            return enrich_existing(args.labels, args.manifest, args.data_dir)
        except HydrationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    try:
        code, _stats = run_hydration(args)
    except HydrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
