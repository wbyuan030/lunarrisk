#!/usr/bin/env python3
"""P1-03/P1-04: first real physical baseline on the hydrated tiles.

Inputs
------
* ``outputs/labels_hydrated.csv`` — real labels from ``scripts/hydrate_tiles.py``
  (raw counts over the native 3 m slope grid).
* Hydrated NetCDF tiles under ``data/hydrated/<sample_id>/``.

Alignment
---------
The NAC tile is resampled with ``map_coordinates`` onto the *slope grid
coordinates* using the per-file ``x``/``y`` vectors. This is a coordinate-driven
footprint mapping, not an assumption that pixel (i, j) of one grid equals pixel
(i, j) of another; the grids differ in size (512 vs 171) and origin (~0.997 m).

Targets and rules
-----------------
``y1 = g(10 deg)``, ``y2 = g(15 deg)`` (area fractions of valid native pixels).
Pre-registered engineering rule: ``unsafe = 1[y1 > 0.05]``.
A second, independently reported rule uses ``y2 > 0.02``.

Models
------
* ``constant``     — train mean / train prevalence.
* ``coarse_ridge`` — ridge + logistic regression on the SLDEM2015 60 m slope
  summary features of the tile.
* ``cnn_<variant>`` — 4-layer CNN over the same architecture and the same tiles:
    * ``nac``         : channel 0, geo-aligned monocular NAC,
    * ``coarse_grid`` : channel 1, the 9x9 60 m slope grid upsampled,
    * ``fusion``      : both channels.
  ``coarse_grid`` vs ``nac`` isolates what the optical image adds at fixed
  architecture, tiles, seed and early-stopping protocol.

The ``cal`` split is used only to Platt-calibrate the CNN event probability.
Metrics carry group-level bootstrap CIs; the group is the leakage unit.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import map_coordinates, zoom
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

COARSE_FEATURES = [
    "slope60m_frac_gt_10", "slope60m_frac_gt_15", "slope60m_mean",
    "slope60m_q50", "slope60m_q90", "slope60m_q99", "slope60m_valid_fraction",
]
RULES = {
    "g10_gt_0.05": ("slope3m_g10", 0.05),
    "g15_gt_0.02": ("slope3m_g15", 0.02),
}
CNN_VARIANTS = {"nac": (0,), "coarse_grid": (1,), "fusion": (0, 1)}
CHANNELS = ("nac", "coarse_slope")
COVERAGES = (0.2, 0.4, 0.6, 0.8)
PRIMARY_RULE = "g10_gt_0.05"
SLOPE_MAX_DEG = 90.0


def open_dataset(path: Path):
    import hdf5plugin  # noqa: F401
    import h5netcdf  # noqa: F401
    import xarray as xr
    return xr.open_dataset(path, engine="h5netcdf")


def align_nac_to_slope_grid(nac_path: Path, slope_path: Path) -> tuple[np.ndarray, dict]:
    """Resample NAC onto the slope grid coordinates (bilinear)."""
    with open_dataset(nac_path) as nac, open_dataset(slope_path) as slope:
        nac_var = next(n for n, v in nac.data_vars.items() if v.ndim == 2)
        nac_values = np.asarray(nac[nac_var].values, dtype=np.float32)
        nac_x = np.asarray(nac.x.values, dtype=np.float64)
        nac_y = np.asarray(nac.y.values, dtype=np.float64)
        slope_x = np.asarray(slope.x.values, dtype=np.float64)
        slope_y = np.asarray(slope.y.values, dtype=np.float64)
        gsd_nac = abs(float(nac.attrs.get("pix_res_x", 0.0)))
        gsd_slope = abs(float(slope.attrs.get("pix_res_x", 0.0)))
    dx_nac = (nac_x[-1] - nac_x[0]) / (len(nac_x) - 1)
    dy_nac = (nac_y[-1] - nac_y[0]) / (len(nac_y) - 1)
    cols = (slope_x - nac_x[0]) / dx_nac
    rows = (slope_y - nac_y[0]) / dy_nac
    fine, coarse = np.meshgrid(rows, cols, indexing="ij")
    aligned = map_coordinates(nac_values, [fine, coarse], order=1, mode="nearest")
    shift_m = float(abs(slope_x[0] - nac_x[0]))
    return aligned, {
        "gsd_nac": gsd_nac,
        "gsd_slope": gsd_slope,
        "origin_shift_m": shift_m,
        "origin_shift_in_nac_pixels": shift_m / gsd_nac if gsd_nac else None,
        "aligned_shape": list(aligned.shape),
    }


def robust_normalize(array: np.ndarray) -> np.ndarray:
    """Percentile normalization that tolerates NaN; NaN pixels become 0."""
    valid = np.isfinite(array)
    if not valid.any():
        return np.zeros_like(array, dtype=np.float32)
    p1, p99 = np.nanpercentile(np.where(valid, array, np.nan), [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 - p1 < 1e-9:
        return np.zeros_like(array, dtype=np.float32)
    normalized = (array - p1) / (p99 - p1)
    return np.nan_to_num(np.clip(normalized, 0.0, 1.5), nan=0.0, posinf=1.5, neginf=0.0).astype(np.float32)


def load_coarse_channel(path: Path, size: int) -> np.ndarray:
    """60 m slope grid (9x9) as a physically scaled channel in [0, 1]."""
    with open_dataset(path) as dataset:
        var = next(n for n, v in dataset.data_vars.items() if v.ndim == 2)
        values = np.asarray(dataset[var].values, dtype=np.float64)
    values = np.nan_to_num(values, nan=float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0)
    values = np.clip(values, 0.0, SLOPE_MAX_DEG) / SLOPE_MAX_DEG
    if values.shape != (size, size):
        values = zoom(values, (size / values.shape[0], size / values.shape[1]), order=1)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def build_cache(labels: pd.DataFrame, cache_path: Path, size: int, force: bool = False) -> tuple[dict, dict]:
    """Cache aligned inputs as (2, size, size): channel 0 NAC, channel 1 coarse slope."""
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        meta = json.loads(str(data["__meta__"])) if "__meta__" in data.files else {}
        arrays = {str(k): data[k] for k in data.files if k != "__meta__"}
        sample = next(iter(arrays.values()), None)
        if sample is not None and sample.ndim == 3 and sample.shape[0] == len(CHANNELS):
            return arrays, meta
        print(f"cache {cache_path} has an incompatible layout; rebuilding")
    cache, meta = {}, {}
    for index, row in labels.iterrows():
        sample_id = str(row["sample_id"])
        aligned, info = align_nac_to_slope_grid(Path(row["layers_nac"]), Path(row["layers_slope_3m"]))
        valid_fraction = float(np.isfinite(aligned).mean())
        small = zoom(aligned, (size / aligned.shape[0], size / aligned.shape[1]), order=1)
        nac_channel = robust_normalize(small)
        coarse_channel = load_coarse_channel(Path(row["layers_coarse_slope"]), size)
        cache[sample_id] = np.stack([nac_channel, coarse_channel]).astype(np.float16)
        info.update({
            "nac_valid_fraction": valid_fraction,
            "nac_all_nan": valid_fraction == 0.0,
            "cache_shape": list(cache[sample_id].shape),
        })
        meta[sample_id] = info
        if (index + 1) % 100 == 0:
            print(f"  aligned {index + 1}/{len(labels)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, __meta__=json.dumps(meta), **cache)
    return cache, meta


class TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def batch_tensor(ids: list[str], cache: dict, device: torch.device, channels: tuple[int, ...]) -> torch.Tensor:
    stacked = np.stack([cache[i][list(channels)] for i in ids]).astype(np.float32)
    return torch.from_numpy(stacked).to(device)


def train_variant(cache, data: pd.DataFrame, args, device, channels: tuple[int, ...]) -> dict:
    train_rows = data[data["split"] == "train"].reset_index(drop=True)
    val_rows = data[data["split"] == "val"].reset_index(drop=True)
    y_train = train_rows[["slope3m_g10", "slope3m_g15"]].to_numpy(dtype=np.float32)
    # Each rule receives its own event logit, not a shared g10 classification score.
    e_train = np.column_stack([
        (train_rows[target] > threshold).to_numpy(dtype=np.float32)
        for target, threshold in RULES.values()
    ])
    y_val = val_rows[["slope3m_g10", "slope3m_g15"]].to_numpy(dtype=np.float32)

    seed = args.seed  # identical across variants for paired comparisons
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyCNN(in_channels=len(channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    def evaluate(rows):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(rows), args.batch):
                ids = rows["sample_id"].iloc[start:start + args.batch].tolist()
                outputs.append(model(batch_tensor(ids, cache, device, channels)).cpu().numpy())
        out = np.concatenate(outputs)
        return 1 / (1 + np.exp(-out[:, :2])), out[:, 2]

    best = {"val_mae": float("inf"), "state": None, "epoch": 0}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            ids = train_rows["sample_id"].iloc[index].tolist()
            x = batch_tensor(ids, cache, device, channels)
            if args.augment:
                if rng.random() < 0.5:
                    x = torch.flip(x, dims=[3])
                if rng.random() < 0.5:
                    x = torch.flip(x, dims=[2])
                k = int(rng.integers(0, 4))
                if k:
                    x = torch.rot90(x, k, dims=[2, 3])
            target = torch.from_numpy(y_train[index]).to(device)
            event = torch.from_numpy(e_train[index]).to(device)
            out = model(x)
            loss = mse(torch.sigmoid(out[:, :2]), target) + 0.5 * bce(out[:, 2:4], event)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_fractions, _ = evaluate(val_rows)
        val_mae = float(np.abs(val_fractions - y_val).mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best["val_mae"]:
            best = {"val_mae": val_mae,
                    "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": epoch}
        if args.verbose:
            print(f"  [{'+'.join(CHANNELS[c] for c in channels)}] epoch {epoch:3d} "
                  f"loss={history[-1]['train_loss']:.5f} val_mae={val_mae:.5f}", flush=True)
        if epoch - best["epoch"] >= args.patience:
            break
    model.load_state_dict(best["state"])
    return {"model": model, "history": history, "best_epoch": best["epoch"],
            "best_val_mae": best["val_mae"], "seed": seed}


def predict_variant(result, rows, cache, channels, batch, device) -> tuple[np.ndarray, np.ndarray]:
    model = result["model"]
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            ids = rows["sample_id"].iloc[start:start + batch].tolist()
            outputs.append(model(batch_tensor(ids, cache, device, channels)).cpu().numpy())
    out = np.concatenate(outputs)
    return 1 / (1 + np.exp(-out[:, :2])), out[:, 2]


def platt_scale(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float, bool]:
    """Fit Platt scaling; return (slope, intercept, degenerate)."""
    targets = np.asarray(targets)
    if len(np.unique(targets)) < 2:
        return 1.0, 0.0, True
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(scores.reshape(-1, 1), targets)
    return float(model.coef_[0][0]), float(model.intercept_[0]), False


def fit_logistic(features: np.ndarray, targets: np.ndarray) -> tuple[object | None, bool]:
    targets = np.asarray(targets)
    if len(np.unique(targets)) < 2:
        return None, True
    return LogisticRegression(max_iter=5000).fit(features, targets), False


def logistic_probability(model, features: np.ndarray, fallback: float) -> np.ndarray:
    if model is None:
        return np.full(len(features), fallback)
    return model.predict_proba(features)[:, 1]


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def metrics(y: np.ndarray, p: np.ndarray, rule: tuple[str, float] | None = None,
            event_probability: np.ndarray | None = None) -> dict:
    result = {
        "mae": float(np.abs(p - y).mean()),
        "rmse": float(np.sqrt(((p - y) ** 2).mean())),
    }
    if rule is not None and event_probability is not None:
        name, threshold = rule
        truth = (y > threshold).astype(np.float64)
        probability = np.clip(event_probability, 1e-7, 1 - 1e-7)
        result.update({
            "rule": name,
            "threshold": threshold,
            "unsafe_rate": float(truth.mean()),
            "brier": float(((probability - truth) ** 2).mean()),
            "nll": float(-(truth * np.log(probability) + (1 - truth) * np.log(1 - probability)).mean()),
            "auc": float(roc_auc_score(truth, probability)) if 0 < truth.sum() < len(truth) else None,
            "brier_constant": float(((truth.mean() - truth) ** 2).mean()),
        })
    return result


def risk_coverage(y: np.ndarray, unsafe_probability: np.ndarray, rule_name: str,
                  coverages=COVERAGES) -> list[dict]:
    """Accept lowest P(unsafe) first; do not pass P(safe)."""
    _, threshold = RULES[rule_name]
    truth = (y > threshold).astype(int)
    unsafe_probability = np.asarray(unsafe_probability, dtype=np.float64)
    if unsafe_probability.shape != truth.shape or not np.isfinite(unsafe_probability).all():
        raise ValueError("unsafe_probability must be a finite vector aligned with y")
    if np.any((unsafe_probability < 0) | (unsafe_probability > 1)):
        raise ValueError("unsafe_probability must lie in [0, 1]")
    order = np.argsort(unsafe_probability, kind="mergesort")
    rows = []
    for coverage in coverages:
        k = max(1, int(round(coverage * len(y))))
        accepted = order[:k]
        unsafe_accepted = int(truth[accepted].sum())
        rows.append({
            "rule": rule_name,
            "threshold": threshold,
            "coverage_requested": coverage,
            "coverage_actual": round(k / len(y), 4),
            "accepted_n": k,
            "test_n": len(y),
            "unsafe_accepted_n": unsafe_accepted,
            "unsafe_total": int(truth.sum()),
            "UAR": unsafe_accepted / k,
            "HMR": unsafe_accepted / int(truth.sum()) if truth.sum() else None,
        })
    return rows


def grouped_bootstrap(values: np.ndarray, groups: np.ndarray, statistic, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique}
    samples = []
    for _ in range(n):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in chosen])
        samples.append(statistic(values[idx], idx))
    samples = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(np.mean(samples)),
        "lo95": float(np.percentile(samples, 2.5)),
        "hi95": float(np.percentile(samples, 97.5)),
        "n_groups": int(len(unique)),
        "n_resamples": n,
    }


def rule_column_index(rule_name: str) -> int:
    """Column index of the regression head that matches this rule's target."""
    target = RULES[rule_name][0]
    return 0 if target.endswith("g10") else 1


def evaluate_model(y_true: np.ndarray, fractions: np.ndarray, raw_scores: np.ndarray,
                   rule_name: str, cal_scores=None, cal_truth=None) -> dict:
    """Uniform payload: primary (calibrated when possible) plus raw metrics."""
    rule = RULES[rule_name]
    index = rule_column_index(rule_name)
    raw_probability = 1 / (1 + np.exp(-raw_scores))
    payload = {
        "metrics_raw": metrics(y_true, fractions[:, index], rule, raw_probability),
        "risk_coverage_raw": risk_coverage(y_true, raw_probability, rule_name),
    }
    if cal_scores is not None and cal_truth is not None:
        slope, intercept, degenerate = platt_scale(cal_scores, cal_truth)
        calibrated = 1 / (1 + np.exp(-(slope * raw_scores + intercept)))
        payload["platt"] = {"slope": slope, "intercept": intercept, "degenerate": degenerate,
                            "cal_unsafe_rate": float(cal_truth.mean())}
        payload["metrics_calibrated"] = metrics(y_true, fractions[:, index], rule, calibrated)
        payload["risk_coverage_calibrated"] = risk_coverage(y_true, calibrated, rule_name)
        payload["primary"] = "calibrated" if not degenerate else "raw"
        payload["probability"] = calibrated
        payload["metrics"] = payload["metrics_calibrated"] if not degenerate else payload["metrics_raw"]
        payload["risk_coverage"] = (payload["risk_coverage_calibrated"] if not degenerate
                                    else payload["risk_coverage_raw"])
    else:
        payload["platt"] = None
        payload["metrics_calibrated"] = None
        payload["risk_coverage_calibrated"] = None
        payload["primary"] = "raw"
        payload["probability"] = raw_probability
        payload["metrics"] = payload["metrics_raw"]
        payload["risk_coverage"] = payload["risk_coverage_raw"]
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels_hydrated.csv"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache_v2.npz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--min-nac-valid", type=float, default=0.99,
                        help="minimum finite fraction of the aligned NAC layer")
    parser.add_argument("--variants", default="nac,coarse_grid,fusion")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    labels = pd.read_csv(args.labels)
    labels["layers_nac"] = labels["sample_id"].map(lambda s: f"data/hydrated/{s}/nac.nc")
    labels["layers_slope_3m"] = labels["sample_id"].map(lambda s: f"data/hydrated/{s}/slope_3m.nc")
    labels["layers_coarse_slope"] = labels["sample_id"].map(lambda s: f"data/hydrated/{s}/slope_60m.nc")
    missing = [path for path in labels["layers_nac"].head(1) if not Path(path).exists()]
    if missing:
        raise SystemExit(f"hydrated tiles not found: {missing[0]}")
    labels = labels[labels["split"].isin(["train", "cal", "val", "test"])].reset_index(drop=True)
    print("tiles:", labels.groupby("split").size().to_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache, cache_meta = build_cache(labels, args.cache, args.size, force=args.force_cache)
    if set(labels["sample_id"]) - set(cache):
        raise SystemExit("cache is missing tiles; rerun with --force-cache")

    labels["nac_valid_fraction"] = labels["sample_id"].map(
        lambda s: float(cache_meta.get(s, {}).get("nac_valid_fraction", 1.0)))
    labels["nac_all_nan"] = labels["sample_id"].map(
        lambda s: bool(cache_meta.get(s, {}).get("nac_all_nan", False)))
    usable = labels["nac_valid_fraction"] >= args.min_nac_valid
    excluded = labels[~usable]
    print(f"NAC availability: {int(usable.sum())}/{len(labels)} usable "
          f"(>= {args.min_nac_valid}); excluded {int((~usable).sum())}")
    if len(excluded):
        print("  excluded by split:", excluded.groupby("split").size().to_dict())
        print("  excluded label means:", {
            "g10": round(float(excluded["slope3m_g10"].mean()), 4),
            "g15": round(float(excluded["slope3m_g15"].mean()), 4),
            "unsafe_rate": round(float((excluded["slope3m_g10"] > 0.05).mean()), 4),
        })
    data = labels[usable].reset_index(drop=True)
    for split in ("train", "cal", "val", "test"):
        if data[data["split"] == split].empty:
            raise SystemExit(f"split {split} has no usable tiles after NAC filtering")

    train_rows = data[data["split"] == "train"].reset_index(drop=True)
    cal_rows = data[data["split"] == "cal"].reset_index(drop=True)
    val_rows = data[data["split"] == "val"].reset_index(drop=True)
    test_rows = data[data["split"] == "test"].reset_index(drop=True)
    y_train = train_rows[["slope3m_g10", "slope3m_g15"]].to_numpy(dtype=np.float64)
    test_groups = test_rows["group_id"].to_numpy()

    results = {
        "protocol": {
            "models": ["constant", "coarse_ridge"] + [f"cnn_{v}" for v in CNN_VARIANTS],
            "shared_train_tiles": int(len(train_rows)),
            "shared_test_tiles": int(len(test_rows)),
            "test_groups": int(test_rows["group_id"].nunique()),
            "seed": args.seed,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "input_size": args.size,
            "augment": args.augment,
            "calibration": "Platt scaling fitted on the cal split, applied to coarse_ridge and cnn_* probabilities",
        },
        "tiles_all_splits": {s: int((labels["split"] == s).sum()) for s in ("train", "cal", "val", "test")},
        "tiles_used_by_split": {s: int((data["split"] == s).sum()) for s in ("train", "cal", "val", "test")},
        "groups_used_by_split": {s: int(data[data["split"] == s]["group_id"].nunique())
                                 for s in ("train", "cal", "val", "test")},
        "nac_availability": {
            "threshold": args.min_nac_valid,
            "usable": int(usable.sum()),
            "excluded": int((~usable).sum()),
            "excluded_by_split": {k: int(v) for k, v in excluded.groupby("split").size().items()},
            "excluded_all_nan_tiles": int(excluded["nac_all_nan"].sum()),
            "excluded_mean_g10": float(excluded["slope3m_g10"].mean()) if len(excluded) else None,
            "excluded_mean_g15": float(excluded["slope3m_g15"].mean()) if len(excluded) else None,
            "excluded_unsafe_rate": float((excluded["slope3m_g10"] > 0.05).mean()) if len(excluded) else None,
            "note": "tiles with an unusable NAC layer are dropped from every model so all "
                    "comparisons share one test set; the resulting selection bias is reported",
        },
        "alignment_check": {row["sample_id"]: cache_meta.get(row["sample_id"], {})
                            for _, row in data.head(5).iterrows()},
        "rules": {k: {"target": v[0], "threshold": v[1]} for k, v in RULES.items()},
        "models": {},
    }

    # ---- Baseline A: constant ----
    constant = {
        "y1": float(y_train[:, 0].mean()),
        "y2": float(y_train[:, 1].mean()),
        "prevalence_g10_gt_0.05": float((train_rows["slope3m_g10"] > 0.05).mean()),
        "prevalence_g15_gt_0.02": float((train_rows["slope3m_g15"] > 0.02).mean()),
    }
    payload = {}
    for name, (target, threshold) in RULES.items():
        y_true = test_rows[target].to_numpy(dtype=np.float64)
        prevalence = constant["prevalence_g10_gt_0.05" if target.endswith("g10") else "prevalence_g15_gt_0.02"]
        mean = constant["y1" if target.endswith("g10") else "y2"]
        payload[name] = evaluate_model(
            y_true,
            np.full((len(y_true), 2), mean),
            logit(np.full(len(y_true), prevalence)),
            name,
        )
    results["models"]["constant"] = {"params": constant, "test": payload}

    # ---- Baseline B: coarse 60 m slope summary features ----
    features = data[COARSE_FEATURES].to_numpy(dtype=np.float64)
    nonfinite = ~np.isfinite(features).all(axis=1)
    if nonfinite.any():
        print(f"WARNING: {int(nonfinite.sum())} tiles have non-finite coarse features; imputing medians")
    medians = np.nanmedian(features, axis=0)
    features = np.where(np.isfinite(features), features, medians)
    scaler = StandardScaler().fit(features[data["split"] == "train"])
    features_scaled = scaler.transform(features)
    split_index = {s: np.where(data["split"].to_numpy() == s)[0] for s in ("train", "cal", "val", "test")}
    ridge = Ridge(alpha=1.0).fit(features_scaled[split_index["train"]], y_train)
    ridge_predictions = ridge.predict(features_scaled[split_index["test"]])
    payload, degenerate_flags = {}, {}
    for name, (target, threshold) in RULES.items():
        truth = (train_rows[target].to_numpy() > threshold).astype(int)
        model, degenerate = fit_logistic(features_scaled[split_index["train"]], truth)
        degenerate_flags[name] = degenerate
        probability = logistic_probability(model, features_scaled[split_index["test"]], float(truth.mean()))
        cal_probability = logistic_probability(
            model, features_scaled[split_index["cal"]], float(truth.mean())
        )
        payload[name] = evaluate_model(
            test_rows[target].to_numpy(dtype=np.float64), ridge_predictions,
            logit(probability), name,
            cal_scores=logit(cal_probability),
            cal_truth=(cal_rows[target].to_numpy() > threshold).astype(int),
        )
    results["models"]["coarse_ridge"] = {
        "features": COARSE_FEATURES,
        "ridge_coef": ridge.coef_.tolist(),
        "ridge_intercept": ridge.intercept_.tolist(),
        "logistic_degenerate": degenerate_flags,
        "test": payload,
    }

    # ---- Model C: CNN variants over aligned inputs ----
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variant_names if v not in CNN_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; known: {list(CNN_VARIANTS)}")
    for variant in variant_names:
        channels = CNN_VARIANTS[variant]
        print(f"training cnn_{variant} with channels {[CHANNELS[c] for c in channels]}")
        trained = train_variant(cache, data, args, device, channels)
        test_fractions, test_scores = predict_variant(trained, test_rows, cache, channels, args.batch, device)
        val_fractions, _ = predict_variant(trained, val_rows, cache, channels, args.batch, device)
        cal_fractions, cal_scores = predict_variant(trained, cal_rows, cache, channels, args.batch, device)
        payload = {}
        for name, (target, threshold) in RULES.items():
            payload[name] = evaluate_model(
                test_rows[target].to_numpy(dtype=np.float64), test_fractions,
                test_scores[:, rule_column_index(name)], name,
                cal_scores=cal_scores[:, rule_column_index(name)],
                cal_truth=(cal_rows[target].to_numpy() > threshold).astype(int),
            )
        probabilities = {name: payload[name]["probability"] for name in RULES}
        results["models"][f"cnn_{variant}"] = {
            "channels": [CHANNELS[c] for c in channels],
            "seed": trained["seed"],
            "best_epoch": trained["best_epoch"],
            "best_val_mae": trained["best_val_mae"],
            "history": trained["history"],
            "test": payload,
            "_predictions": {"fractions": test_fractions, "probabilities": probabilities,
                             "scores": test_scores},
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }

    # ---- Grouped bootstrap on the primary rule ----
    target, threshold = RULES[PRIMARY_RULE]
    y_true = test_rows[target].to_numpy(dtype=np.float64)
    truth = (y_true > threshold).astype(np.float64)
    predictions, probabilities, primary_scores = {}, {}, {}
    for name, model in results["models"].items():
        payload = model["test"][PRIMARY_RULE]
        probabilities[name] = payload["probability"]
        if name == "constant":
            predictions[name] = np.full(len(y_true), model["params"]["y1"])
        elif name == "coarse_ridge":
            predictions[name] = ridge_predictions[:, 0]
        else:
            predictions[name] = model["_predictions"]["fractions"][:, 0]
            primary_scores[name] = model["_predictions"]["scores"]
    bootstrap = {}
    for name, prediction in predictions.items():
        probability = np.clip(probabilities[name], 1e-7, 1 - 1e-7)
        bootstrap[name] = {
            "mae_g10": grouped_bootstrap(
                y_true, test_groups, lambda v, i: float(np.abs(v - prediction[i]).mean()),
                args.bootstrap, args.seed),
            "brier_g10_gt_0.05": grouped_bootstrap(
                truth, test_groups, lambda v, i: float(((probability[i] - v) ** 2).mean()),
                args.bootstrap, args.seed),
        }
    results["grouped_bootstrap"] = bootstrap

    # ---- Label distributions ----
    distribution = {}
    for split in ("train", "cal", "val", "test"):
        part = data[data["split"] == split]
        distribution[split] = {
            "tiles": int(len(part)),
            "g10_quantiles": {str(q): round(float(part["slope3m_g10"].quantile(q)), 5)
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "g15_quantiles": {str(q): round(float(part["slope3m_g15"].quantile(q)), 5)
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "g10_mean": round(float(part["slope3m_g10"].mean()), 5),
            "unsafe_rate_g10_gt_0.05": round(float((part["slope3m_g10"] > 0.05).mean()), 5),
            "unsafe_rate_g15_gt_0.02": round(float((part["slope3m_g15"] > 0.02).mean()), 5),
            "ltm_zones": int(part["LTM_CODE"].nunique()),
        }
    results["label_distribution"] = distribution

    # ---- Write artifacts ----
    args.out.mkdir(parents=True, exist_ok=True)
    export = {
        "sample_id": test_rows["sample_id"], "group_id": test_rows["group_id"],
        "LTM_CODE": test_rows["LTM_CODE"], "y_g10": test_rows["slope3m_g10"],
        "y_g15": test_rows["slope3m_g15"],
        "slope60m_frac_gt_10": test_rows["slope60m_frac_gt_10"],
        "constant_g10": predictions["constant"], "coarse_g10": predictions["coarse_ridge"],
    }
    for variant in variant_names:
        export[f"cnn_{variant}_g10"] = results["models"][f"cnn_{variant}"]["_predictions"]["fractions"][:, 0]
        export[f"cnn_{variant}_g15"] = results["models"][f"cnn_{variant}"]["_predictions"]["fractions"][:, 1]
        export[f"cnn_{variant}_event_prob"] = results["models"][f"cnn_{variant}"]["test"][PRIMARY_RULE]["probability"]
    pd.DataFrame(export).to_csv(args.out / "test_predictions.csv", index=False)

    rc_rows = []
    for name, model in results["models"].items():
        for rule_name, payload in model["test"].items():
            for label, key in (("raw", "risk_coverage_raw"), ("calibrated", "risk_coverage_calibrated")):
                for entry in (payload[key] or []):
                    rc_rows.append({"model": name, "probability_variant": label, **entry})
    with (args.out / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rc_rows[0]))
        writer.writeheader()
        writer.writerows(rc_rows)

    for model in results["models"].values():
        test_payload = model["test"]
        for rule_name, payload in test_payload.items():
            payload.pop("probability", None)
    for name, model in results["models"].items():
        model.pop("_predictions", None)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = {
        "test_tiles": len(test_rows),
        "test_groups": int(test_rows["group_id"].nunique()),
        "excluded_tiles": int((~usable).sum()),
        "mae_g10": {name: round(model["test"][PRIMARY_RULE]["metrics"]["mae"], 5)
                    for name, model in results["models"].items()},
        "brier_g10": {name: round(model["test"][PRIMARY_RULE]["metrics"]["brier"], 5)
                      for name, model in results["models"].items()},
        "brier_g10_raw": {name: round(model["test"][PRIMARY_RULE]["metrics_raw"]["brier"], 5)
                          for name, model in results["models"].items()},
        "best_epochs": {name: model.get("best_epoch") for name, model in results["models"].items()
                        if "best_epoch" in model},
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}/metrics.json, test_predictions.csv, risk_coverage.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
