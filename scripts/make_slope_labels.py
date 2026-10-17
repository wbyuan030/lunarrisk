#!/usr/bin/env python3
"""Empirical slope labels from native NetCDF slope grids.

For a grid, with the valid pixel set V (finite and 0 <= slope <= 90 degrees):

    g(theta) = count(slope > theta) / |V|

The default engineering rule is g(10 deg) > 0.05. This is a proxy rule, not a
certified landing-safety label. No smoothing, zero replacement, or pseudo-counts
are applied to the raw counts.

The script does NOT georeference the slope grid against NAC imagery. Every row
therefore carries ``geo_join_status=GEO_JOIN_PENDING`` and must not be used as a
supervised NAC -> slope training pair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# 10 thresholds -> 11 left-closed/right-open bins:
#   bin_0 = [0, 1), bin_1 = [1, 2), ..., bin_9 = [40, 60), bin_10 = [60, +inf)
DEFAULT_THRESHOLDS = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)
EXCEEDANCE_THRESHOLDS = (10.0, 15.0)
UNSAFE_PROXY_THRESHOLD = 10.0
UNSAFE_PROXY_FRACTION = 0.05
MIN_SLOPE = 0.0
MAX_SLOPE = 90.0


class SlopeLabelError(RuntimeError):
    pass


def open_slope_dataset(path: Path) -> xr.Dataset:
    """Open NetCDF, tolerating HDF5 filters (e.g. zstd) via h5netcdf/hdf5plugin."""
    try:  # enable extra HDF5 compression filters when available
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    errors = []
    for engine in ("h5netcdf", "netcdf4", None):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:  # noqa: BLE001 - report all engines
            errors.append(f"{engine}: {type(exc).__name__}: {exc}")
    raise SlopeLabelError(f"cannot open {path}: " + " | ".join(errors))


def find_slope_variable(ds: xr.Dataset, requested: str | None = None) -> str:
    if requested:
        if requested not in ds.variables:
            raise SlopeLabelError(
                f"variable {requested!r} not found; available: {list(ds.data_vars)}"
            )
        return requested
    named = [n for n in ds.data_vars if "slope" in n.lower()]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        raise SlopeLabelError(f"multiple slope-like variables: {named}; pass --variable")
    numeric_2d = [
        n for n, v in ds.data_vars.items()
        if v.ndim == 2 and np.issubdtype(v.dtype, np.number)
    ]
    if len(numeric_2d) == 1:
        return numeric_2d[0]
    if not numeric_2d:
        raise SlopeLabelError(f"no 2-D numeric data variable; available: {list(ds.data_vars)}")
    raise SlopeLabelError(
        f"ambiguous 2-D variables {numeric_2d}; no name contains 'slope'; pass --variable"
    )


def angle_units_status(ds: xr.Dataset, variable: str) -> tuple[bool, str, str]:
    var_units = ds[variable].attrs.get("units")
    if var_units is not None and str(var_units).strip():
        units = str(var_units).strip().lower()
        source = f"{variable}.attrs['units']"
    else:
        ds_units = ds.attrs.get("units")
        units = str(ds_units).strip().lower() if ds_units is not None else ""
        source = "dataset.attrs['units']" if units else "absent"
    if not units:
        return False, "", source
    if "deg" in units:
        return True, units, source
    return False, units, source


def _threshold_column(theta: float) -> str:
    return f"frac_gt_{theta:g}deg"


def compute_labels(values: np.ndarray, thresholds=DEFAULT_THRESHOLDS) -> dict:
    """Compute raw counts/fractions. Out-of-range finite values are invalid."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    negative = finite & (values < MIN_SLOPE)
    above_90 = finite & (values > MAX_SLOPE)
    valid = finite & ~negative & ~above_90
    v = values[valid]
    n_total = int(values.size)
    n_valid = int(v.size)
    result = {
        "n_total": n_total,
        "n_finite": int(finite.sum()),
        "n_nan": int((~finite).sum()),
        "n_negative": int(negative.sum()),
        "n_gt90": int(above_90.sum()),
        "n_valid": n_valid,
        "valid_fraction": (n_valid / n_total) if n_total else 0.0,
        "valid_fraction_ge_0_95": bool(n_total and n_valid / n_total >= 0.95),
        "min_valid_deg": float(v.min()) if n_valid else None,
        "max_valid_deg": float(v.max()) if n_valid else None,
        "mean_valid_deg": float(v.mean()) if n_valid else None,
        "median_valid_deg": float(np.median(v)) if n_valid else None,
    }
    for theta in thresholds:
        count = int((v > theta).sum())
        result[_threshold_column(theta)] = count
        result[_threshold_column(theta) + "_fraction"] = (count / n_valid) if n_valid else None
    # 11 bins, left-closed/right-open, last unbounded.
    edges = (MIN_SLOPE, *thresholds, math.inf)
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        counts.append(int(((v >= lo) & (v < hi)).sum()))
    assert len(counts) == len(thresholds) + 1
    assert sum(counts) == n_valid
    for idx, count in enumerate(counts):
        result[f"bin_{idx}"] = count
        result[f"bin_{idx}_fraction"] = (count / n_valid) if n_valid else None
    g10 = result[_threshold_column(UNSAFE_PROXY_THRESHOLD) + "_fraction"]
    g15 = result[_threshold_column(15.0) + "_fraction"]
    result["g10"] = g10
    result["g15"] = g15
    result["unsafe_proxy_g10_gt_0_05"] = (
        None if g10 is None else bool(g10 > UNSAFE_PROXY_FRACTION)
    )
    return result


def source_id_from_stem(stem: str) -> str:
    match = re.match(r"(.+?)_r\d+_c\d+$", stem)
    return match.group(1) if match else stem


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_one_file(
    path: Path,
    variable: str | None,
    allow_unverified_units: bool,
    units_override: str | None = None,
    units_source: str = "auto",
) -> dict:
    try:
        ds = open_slope_dataset(path)
    except SlopeLabelError:
        raise
    try:
        name = find_slope_variable(ds, variable)
        if units_override:
            units_ok, units = True, units_override
            source = units_source
        else:
            units_ok, units, source = angle_units_status(ds, name)
        values = ds[name].values
        if values.ndim != 2:
            raise SlopeLabelError(f"{path}: variable {name!r} is not 2-D ({values.shape})")
        if not units_ok and not allow_unverified_units:
            raise SlopeLabelError(
                f"{path}: slope units are {units!r} ({source}), expected degrees. "
                "Refusing to interpret values as degrees; verify units, pass "
                "--units degrees --units-source '<evidence>', or pass "
                "--allow-unverified-units to emit unverified exploratory rows."
            )
        labels = compute_labels(values)
        crs = ds.attrs.get("crs", ds[name].attrs.get("crs"))
        pix_res = ds.attrs.get("pix_res_x")
        row = {
            "sample_id": path.stem,
            "source_file": str(path),
            "source_id": source_id_from_stem(path.stem),
            "dtm_source_id": "",  # not asserted by the public sample metadata
            "variable": name,
            "units": units,
            "units_source": source,
            "unit_verified": units_ok,
            "dims": "x".join(str(s) for s in values.shape),
            "crs_wkt": crs or "",
            "gsd_m": abs(float(pix_res)) if pix_res is not None else None,
            "geo_join_status": "GEO_JOIN_PENDING",
            "sha256": sha256_file(path),
            **labels,
        }
        return row
    finally:
        ds.close()


def expand_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.nc")))
        elif p.is_file():
            paths.append(p)
        else:
            matches = sorted(Path().glob(item))
            if not matches:
                raise SlopeLabelError(f"input not found: {item}")
            paths.extend(matches)
    unique = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    if not unique:
        raise SlopeLabelError("no *.nc inputs found")
    return unique


def dedupe_sample_ids(rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        sid = row["sample_id"]
        if sid in counts:
            counts[sid] += 1
            suffix = hashlib.sha1(row["source_file"].encode()).hexdigest()[:8]
            row["sample_id"] = f"{sid}__{suffix}"
        else:
            counts[sid] = 1


def flatten_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        elif isinstance(value, np.bool_):
            value = bool(value)
        if isinstance(value, float):
            value = round(value, 6)
        out[key] = "" if value is None else value
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="NetCDF file(s), directory, or glob")
    parser.add_argument("--variable", default=None, help="slope data variable name")
    parser.add_argument(
        "--units", choices=["auto", "degrees"], default="auto",
        help="auto = require a units attribute containing 'deg'; degrees = documented override",
    )
    parser.add_argument(
        "--units-source", default="user-provided documentation",
        help="provenance string recorded when --units degrees is used",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/slope_labels.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/slope_labels_summary.json"))
    parser.add_argument(
        "--allow-unverified-units", action="store_true",
        help="emit rows with unit_verified=false instead of failing on missing/other units",
    )
    args = parser.parse_args(argv)

    paths = expand_inputs(args.inputs)
    override = "degrees" if args.units == "degrees" else None
    rows = [
        label_one_file(p, args.variable, args.allow_unverified_units, override, args.units_source)
        for p in paths
    ]
    dedupe_sample_ids(rows)
    rows.sort(key=lambda r: (r["source_id"], r["sample_id"]))
    rows = [flatten_row(r) for r in rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_files": len(rows),
        "n_valid_pixels_total": int(sum(r["n_valid"] for r in rows)),
        "n_negative_total": int(sum(r["n_negative"] for r in rows)),
        "n_gt90_total": int(sum(r["n_gt90"] for r in rows)),
        "units_verified_all": bool(all(r["unit_verified"] for r in rows)),
        "geo_join_status": "GEO_JOIN_PENDING",
        "thresholds_deg": list(DEFAULT_THRESHOLDS),
        "bins": "bin_0=[0,1), bin_1..bin_9=[t_i-1,t_i), bin_10=[60,inf) degrees",
        "unsafe_proxy_rule": "g(10 deg) > 0.05 (engineering proxy, not certified safety)",
        "independent_parent_groups": sorted({r["source_id"] for r in rows}),
        "rows": rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {args.out} ({len(rows)} files)")
    for row in rows:
        print(
            f"{row['sample_id']}: valid={row['n_valid']}/{row['n_total']} "
            f"frac={row['valid_fraction']:.3f} g10={row['g10']} "
            f"g15={row['g15']} units={row['units'] or 'MISSING'} "
            f"geo={row['geo_join_status']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SlopeLabelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
