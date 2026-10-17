#!/usr/bin/env python3
"""Sample the public S3 SomBench 100k NAC release and verify real triplets.

The HF sample has only 6 tiles. The public S3 corpus
(``s3://nasa-lunar-fm-bench``) exposes the ~100k-tile release with a Parquet
index, NAC/Dtm_3m/Slope_3m NetCDF files and metadata. This script:

  1. downloads the 100k index (if absent),
  2. selects distinct PRODUCT_ID / LTM_CODE tiles across train/val/test,
  3. downloads NAC + DTM_3m + slope_3m + metadata for each,
  4. checks CRS, grid bounds, units metadata and readability,
  5. writes a JSON report and an overlay PNG for human inspection.

It does not train anything and does not claim site-OOD performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

S3_BASE = "https://nasa-lunar-fm-bench.s3.amazonaws.com/"
INDEX_KEY = "pretraining_dataset/100k_release/NAC_HighRes/NAC_HighRes_100k.parquet"
S3_PREFIX = "pretraining_dataset/100k_release/NAC_HighRes/"
MODALITIES = ("NAC_TILE", "DTM_3M_TILE", "SLOPE_3M_TILE", "METADATA_TILE")


def download(key: str, dest: Path, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = S3_BASE + key
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_error}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    candidates = frame.dropna(subset=["NAC_TILE", "DTM_3M_TILE", "SLOPE_3M_TILE", "METADATA_TILE"]).copy()
    rng = np.random.default_rng(seed)
    per_split = max(1, n // 4)
    quotas = {"train": n - 2 * per_split, "val": per_split, "test": per_split}
    used_products: set[str] = set()
    used_zones: set[str] = set()
    selected = []
    for split, quota in quotas.items():
        pool = candidates[candidates["DATASET"] == split]
        order = rng.permutation(len(pool))
        pool = pool.iloc[order]
        picked = 0
        # First pass: distinct product AND distinct LTM zone.
        for _, row in pool.iterrows():
            if picked >= quota:
                break
            if row["PRODUCT_ID"] in used_products or row["LTM_CODE"] in used_zones:
                continue
            selected.append(row)
            used_products.add(row["PRODUCT_ID"])
            used_zones.add(row["LTM_CODE"])
            picked += 1
        # Second pass: distinct product only.
        for _, row in pool.iterrows():
            if picked >= quota:
                break
            if row["PRODUCT_ID"] in used_products:
                continue
            selected.append(row)
            used_products.add(row["PRODUCT_ID"])
            picked += 1
        if picked < quota:
            raise RuntimeError(f"only {picked}/{quota} distinct products available for split {split}")
    return pd.DataFrame(selected).reset_index(drop=True)


def open_dataset(path: Path):
    import h5netcdf  # noqa: F401
    import hdf5plugin  # noqa: F401
    import xarray as xr
    return xr.open_dataset(path, engine="h5netcdf")


def verify_tile(row: pd.Series, root: Path) -> dict:
    paths = {column: root / str(row[column]) for column in MODALITIES}
    result = {
        "product_id": str(row["PRODUCT_ID"]),
        "row": int(row["ROW"]),
        "col": int(row["COL"]),
        "ltm_code": str(row["LTM_CODE"]),
        "split": str(row["DATASET"]),
        "files": {column: {"path": str(path), "bytes": path.stat().st_size,
                           "sha256": sha256_file(path)} for column, path in paths.items()},
    }
    with open_dataset(paths["NAC_TILE"]) as nac, \
            open_dataset(paths["DTM_3M_TILE"]) as dtm, \
            open_dataset(paths["SLOPE_3M_TILE"]) as slope:
        nac_var = next(n for n, v in nac.data_vars.items() if v.ndim == 2)
        slope_var = next(n for n, v in slope.data_vars.items() if v.ndim == 2)
        dtm_var = next(n for n, v in dtm.data_vars.items() if v.ndim == 2)
        nac_values = nac[nac_var].values
        slope_values = slope[slope_var].values
        dtm_values = dtm[dtm_var].values
        gsd_nac = abs(float(nac.attrs.get("pix_res_x", 0)))
        gsd_slope = abs(float(slope.attrs.get("pix_res_x", 0)))
        dx = max(abs(float(nac.x.min()) - float(slope.x.min())),
                 abs(float(nac.x.max()) - float(slope.x.max())))
        dy = max(abs(float(nac.y.min()) - float(slope.y.min())),
                 abs(float(nac.y.max()) - float(slope.y.max())))
        tolerance = max(gsd_nac, gsd_slope, 1.0) * 1.5
        units = slope[slope_var].attrs.get("units", slope.attrs.get("units", ""))
        result.update({
            "shapes": {"nac": list(nac_values.shape), "dtm_3m": list(dtm_values.shape),
                       "slope_3m": list(slope_values.shape)},
            "gsd_m": {"nac": gsd_nac, "slope_3m": gsd_slope},
            "crs_equal_nac_slope": nac.attrs.get("crs") == slope.attrs.get("crs"),
            "crs_equal_nac_dtm": nac.attrs.get("crs") == dtm.attrs.get("crs"),
            "max_abs_dx_m": round(dx, 4),
            "max_abs_dy_m": round(dy, 4),
            "bounds_aligned": bool(max(dx, dy) <= tolerance),
            "bounds_tolerance_m": tolerance,
            "slope_units_attr": str(units),
            "slope_range_deg": [float(np.nanmin(slope_values)), float(np.nanmax(slope_values))],
            "slope_finite_fraction": float(np.isfinite(slope_values).mean()),
            "dtm_range_m": [float(np.nanmin(dtm_values)), float(np.nanmax(dtm_values))],
            "nac_range": [float(np.nanmin(nac_values)), float(np.nanmax(nac_values))],
            "geo_join_status": "BOUNDS_ALIGNED_GEO_JOIN_PENDING",
        })
        # Quantitative alignment sanity check: NAC gradient magnitude vs slope.
        from scipy.ndimage import zoom
        from scipy.stats import spearmanr
        zoom_y = slope_values.shape[0] / nac_values.shape[0]
        zoom_x = slope_values.shape[1] / nac_values.shape[1]
        nac_small = zoom(nac_values.astype(np.float64), (zoom_y, zoom_x), order=1)
        grad_y, grad_x = np.gradient(nac_small)
        grad = np.hypot(grad_x, grad_y)
        finite = np.isfinite(grad) & np.isfinite(slope_values)
        corr = spearmanr(grad[finite], slope_values[finite]).correlation
        result["nac_gradient_vs_slope_spearman"] = float(corr)
        result["nac_small_shape"] = list(nac_small.shape)

        # Empirical unit/co-registration check: slope from the DTM with gsd in metres.
        gy, gx = np.gradient(dtm_values.astype(np.float64), gsd_slope, gsd_slope)
        slope_from_dtm = np.degrees(np.arctan(np.hypot(gx, gy)))
        both = np.isfinite(slope_from_dtm) & np.isfinite(slope_values)
        dtm_corr = float(np.corrcoef(slope_from_dtm[both], slope_values[both])[0, 1])
        ratio = float(slope_values[both].mean() / slope_from_dtm[both].mean())
        median_abs_diff = float(np.median(np.abs(slope_from_dtm[both] - slope_values[both])))
        result.update({
            "slope_from_dtm_corr": dtm_corr,
            "slope_from_dtm_median_abs_diff_deg": median_abs_diff,
            "slope_from_dtm_mean_ratio": ratio,
            "units_empirically_consistent_with_degrees": bool(
                dtm_corr > 0.9 and 0.8 < ratio < 1.2
            ),
        })
    return result


def make_overlay(rows: list[dict], root: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    n = len(rows)
    cols = 4
    fig, axes = plt.subplots(n, cols, figsize=(16, 3.2 * n))
    if n == 1:
        axes = np.array([axes])
    for i, row in enumerate(rows):
        paths = {column: Path(info["path"]) for column, info in row["files"].items()}
        with open_dataset(paths["NAC_TILE"]) as nac, open_dataset(paths["SLOPE_3M_TILE"]) as slope:
            nac_var = next(n for n, v in nac.data_vars.items() if v.ndim == 2)
            slope_var = next(n for n, v in slope.data_vars.items() if v.ndim == 2)
            nac_values = nac[nac_var].values
            slope_values = slope[slope_var].values
        ax_nac, ax_slope, ax_overlay, ax_hist = axes[i]
        ax_nac.imshow(nac_values, cmap="gray", origin="upper")
        ax_nac.set_title(f"{row['product_id']} {row['ltm_code']} {row['split']}\nNAC 512x512, 1 m/px", fontsize=8)
        ax_slope.imshow(slope_values, cmap="magma", origin="upper", vmin=0, vmax=45)
        ax_slope.set_title(f"slope_3m {slope_values.shape}, {row['gsd_m']['slope_3m']:.2f} m/px", fontsize=8)
        ax_overlay.imshow(nac_values, cmap="gray", origin="upper")
        small = zoom(slope_values, (nac_values.shape[0] / slope_values.shape[0],
                                    nac_values.shape[1] / slope_values.shape[1]), order=1)
        masked = np.ma.masked_where(small < 10, small)
        ax_overlay.imshow(masked, cmap="turbo", alpha=0.45, origin="upper", vmin=0, vmax=45)
        ax_overlay.set_title(f"slope>10 deg overlay\nspearman={row['nac_gradient_vs_slope_spearman']:.3f}", fontsize=8)
        ax_hist.hist(slope_values.ravel(), bins=40, range=(0, 45), color="tab:purple")
        ax_hist.set_title(f"slope histogram\nmax={row['slope_range_deg'][1]:.1f} deg", fontsize=8)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", type=Path, default=Path("data/s3/NAC_HighRes_100k.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/s3/sample"))
    parser.add_argument("--report", type=Path, default=Path("outputs/s3_sample_audit.json"))
    parser.add_argument("--overlay", type=Path, default=Path("outputs/s3_sample_overlays.png"))
    args = parser.parse_args(argv)

    if args.index.exists():
        print(f"using existing index {args.index}")
    else:
        print(f"downloading {INDEX_KEY}")
        download(INDEX_KEY, args.index)
    frame = pd.read_parquet(args.index)
    print(f"index rows: {len(frame)}; products: {frame.PRODUCT_ID.nunique()}; "
          f"zones: {frame.LTM_CODE.nunique()}; splits: {frame.DATASET.value_counts().to_dict()}")
    selected = select_rows(frame, args.n, args.seed)
    print(selected[["PRODUCT_ID", "LTM_CODE", "DATASET", "NAC_TILE"]].to_string(index=False))

    rows = []
    for _, row in selected.iterrows():
        print(f"downloading {row['PRODUCT_ID']} {row['LTM_CODE']} ({row['DATASET']})", flush=True)
        for column in MODALITIES:
            download(S3_PREFIX + str(row[column]), args.out_dir / str(row[column]))
        rows.append(verify_tile(row, args.out_dir))

    aligned = sum(row["bounds_aligned"] and row["crs_equal_nac_slope"] for row in rows)
    report = {
        "source": S3_BASE + INDEX_KEY,
        "index_rows": int(len(frame)),
        "index_products": int(frame.PRODUCT_ID.nunique()),
        "index_zones": int(frame.LTM_CODE.nunique()),
        "index_split_counts": {k: int(v) for k, v in frame.DATASET.value_counts().items()},
        "index_slope3m_present": int(frame.SLOPE_3M_TILE.notna().sum()),
        "sampled": len(rows),
        "bounds_aligned": aligned,
        "all_bounds_aligned": aligned == len(rows),
        "unique_products": len({row["product_id"] for row in rows}),
        "unique_zones": len({row["ltm_code"] for row in rows}),
        "slope_units_attr_present": sum(bool(row["slope_units_attr"]) for row in rows),
        "units_empirically_consistent_with_degrees": sum(
            row["units_empirically_consistent_with_degrees"] for row in rows
        ),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_overlay(rows, args.out_dir, args.overlay)
    print(json.dumps({k: report[k] for k in (
        "sampled", "bounds_aligned", "unique_products", "unique_zones",
        "slope_units_attr_present")}, indent=2))
    print(f"wrote {args.report} and {args.overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
