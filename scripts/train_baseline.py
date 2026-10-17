#!/usr/bin/env python3
"""P1-03/P1-04: first real physical baseline on the hydrated tiles.

Inputs
------
* ``outputs/labels_hydrated.csv`` — real labels from ``scripts/hydrate_tiles.py``
  (raw counts of the native 3 m slope grid).
* Hydrated NetCDF tiles under ``data/hydrated/<sample_id>/``.

Alignment
---------
The NAC tile is resampled with ``map_coordinates`` onto the *slope grid
coordinates* using the per-file ``x``/``y`` vectors. This is a coordinate-driven
footprint mapping, not an assumption that pixel (i, j) of one grid equals pixel
(i, j) of the other; the two grids differ in size (512 vs 171) and origin
(~0.997 m).

Targets and rules
-----------------
``y1 = g(10 deg)``, ``y2 = g(15 deg)`` (area fractions of valid native pixels).
Pre-registered engineering rule: ``unsafe = 1[y1 > 0.05]``.
A second, independently reported rule uses ``y2 > 0.02``.

Models
------
* ``constant``     — train mean / train prevalence.
* ``coarse_ridge`` — ridge + logistic regression on SLDEM2015 60 m slope summary.
* ``cnn``          — 4-layer CNN on geo-aligned monocular NAC.

Metrics are reported with group-level bootstrap CIs; the ``cal`` split is used
only to calibrate the CNN event probability (Platt scaling).
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
COVERAGES = (0.2, 0.4, 0.6, 0.8)


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


def build_cache(labels: pd.DataFrame, cache_path: Path, size: int, force: bool = False) -> tuple[dict, dict]:
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        meta = json.loads(str(data["__meta__"])) if "__meta__" in data.files else {}
        return {str(k): data[k] for k in data.files if k != "__meta__"}, meta
    cache, meta = {}, {}
    for index, row in labels.iterrows():
        aligned, info = align_nac_to_slope_grid(Path(row["layers_nac"]), Path(row["layers_slope_3m"]))
        valid_fraction = float(np.isfinite(aligned).mean())
        info["nac_valid_fraction"] = valid_fraction
        info["nac_all_nan"] = valid_fraction == 0.0
        meta[row["sample_id"]] = info
        small = zoom(aligned, (size / aligned.shape[0], size / aligned.shape[1]), order=1)
        cache[row["sample_id"]] = robust_normalize(small).astype(np.float16)
        if (index + 1) % 100 == 0:
            print(f"  aligned {index + 1}/{len(labels)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, __meta__=json.dumps(meta), **cache)
    return cache, meta


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def batch_tensor(ids: list[str], cache: dict, device: torch.device) -> torch.Tensor:
    stacked = np.stack([cache[i] for i in ids]).astype(np.float32)[:, None, :, :]
    return torch.from_numpy(stacked).to(device)


def train_cnn(cache, labels, args, device) -> dict:
    train_rows = labels[labels["split"] == "train"].reset_index(drop=True)
    val_rows = labels[labels["split"] == "val"].reset_index(drop=True)
    y_train = train_rows[["slope3m_g10", "slope3m_g15"]].to_numpy(dtype=np.float32)
    e_train = (train_rows["slope3m_g10"] > 0.05).to_numpy(dtype=np.float32)
    y_val = val_rows[["slope3m_g10", "slope3m_g15"]].to_numpy(dtype=np.float32)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    def evaluate(rows):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(rows), args.batch):
                ids = rows["sample_id"].iloc[start:start + args.batch].tolist()
                outputs.append(model(batch_tensor(ids, cache, device)).cpu().numpy())
        out = np.concatenate(outputs)
        fractions = 1 / (1 + np.exp(-out[:, :2]))
        return fractions, out[:, 2]

    best = {"val_mae": float("inf"), "state": None, "epoch": 0}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            ids = train_rows["sample_id"].iloc[index].tolist()
            x = batch_tensor(ids, cache, device)
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
            loss = mse(torch.sigmoid(out[:, :2]), target) + 0.5 * bce(out[:, 2], event)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_fractions, _ = evaluate(val_rows)
        val_mae = float(np.abs(val_fractions - y_val).mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best["val_mae"]:
            best = {"val_mae": val_mae, "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": epoch}
        if args.verbose:
            print(f"  epoch {epoch:3d} loss={history[-1]['train_loss']:.5f} val_mae={val_mae:.5f}", flush=True)
        if epoch - best["epoch"] >= args.patience:
            break
    model.load_state_dict(best["state"])
    return {"model": model, "history": history, "best_epoch": best["epoch"], "best_val_mae": best["val_mae"]}


def predict_cnn(model, rows, cache, batch, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            ids = rows["sample_id"].iloc[start:start + batch].tolist()
            outputs.append(model(batch_tensor(ids, cache, device)).cpu().numpy())
    out = np.concatenate(outputs)
    return 1 / (1 + np.exp(-out[:, :2])), out[:, 2]


def platt_scale(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float, bool]:
    """Fit Platt scaling; return (slope, intercept, degenerate)."""
    targets = np.asarray(targets)
    if len(np.unique(targets)) < 2:
        # Only one class present: calibration is not identifiable, keep identity.
        return 1.0, 0.0, True
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(scores.reshape(-1, 1), targets)
    return float(model.coef_[0][0]), float(model.intercept_[0]), False


def fit_logistic(features: np.ndarray, targets: np.ndarray) -> tuple[object | None, bool]:
    """Fit a logistic regression; return (model, degenerate)."""
    targets = np.asarray(targets)
    if len(np.unique(targets)) < 2:
        return None, True
    return LogisticRegression(max_iter=5000).fit(features, targets), False


def logistic_probability(model, features: np.ndarray, fallback: float) -> np.ndarray:
    if model is None:
        return np.full(len(features), fallback)
    return model.predict_proba(features)[:, 1]


def metrics(y: np.ndarray, p: np.ndarray, rule: tuple[str, float] | None = None,
            event_scores: np.ndarray | None = None) -> dict:
    mae = float(np.abs(p - y).mean())
    rmse = float(np.sqrt(((p - y) ** 2).mean()))
    result = {"mae": mae, "rmse": rmse}
    if rule is not None and event_scores is not None:
        name, threshold = rule
        truth = (y > threshold).astype(np.float64)
        probability = 1 / (1 + np.exp(-event_scores))
        eps = 1e-7
        probability = np.clip(probability, eps, 1 - eps)
        result.update({
            "rule": name,
            "threshold": threshold,
            "unsafe_rate": float(truth.mean()),
            "brier": float(((probability - truth) ** 2).mean()),
            "nll": float(-(truth * np.log(probability) + (1 - truth) * np.log(1 - probability)).mean()),
            "auc": float(roc_auc_score(truth, probability)) if 0 < truth.sum() < len(truth) else None,
            "brier_constant": float(((truth.mean() - truth) ** 2).mean()) if len(truth) else None,
        })
    return result


def risk_coverage(y: np.ndarray, scores: np.ndarray, rule: tuple[str, float],
                  coverages=COVERAGES) -> list[dict]:
    """Accept the lowest-predicted-risk tiles first."""
    name, threshold = rule
    truth = (y > threshold).astype(int)
    order = np.argsort(scores, kind="mergesort")
    rows = []
    for coverage in coverages:
        k = max(1, int(round(coverage * len(y))))
        accepted = order[:k]
        unsafe_accepted = int(truth[accepted].sum())
        rows.append({
            "rule": name,
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels_hydrated.csv"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/nac_aligned_cache.npz"))
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    labels = pd.read_csv(args.labels)
    labels["layers_nac"] = labels["sample_id"].map(lambda s: f"data/hydrated/{s}/nac.nc")
    labels["layers_slope_3m"] = labels["sample_id"].map(lambda s: f"data/hydrated/{s}/slope_3m.nc")
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
        lambda s: float(cache_meta.get(s, {}).get("nac_valid_fraction", 1.0))
    )
    labels["nac_all_nan"] = labels["sample_id"].map(
        lambda s: bool(cache_meta.get(s, {}).get("nac_all_nan", False))
    )
    usable = labels["nac_valid_fraction"] >= args.min_nac_valid
    excluded = labels[~usable]
    print(f"NAC availability: {int(usable.sum())}/{len(labels)} usable at threshold {args.min_nac_valid}"
          f" ({int((~usable).sum())} excluded)")
    if len(excluded):
        print("  excluded by split:", excluded.groupby("split").size().to_dict())
        print("  excluded label means:", {
            "g10": round(float(excluded["slope3m_g10"].mean()), 4),
            "g15": round(float(excluded["slope3m_g15"].mean()), 4),
            "unsafe_rate": round(float((excluded["slope3m_g10"] > 0.05).mean()), 4),
            "all_nan_tiles": int(excluded["nac_all_nan"].sum()),
        })
    data = labels[usable].reset_index(drop=True)
    if data.empty:
        raise SystemExit("no tiles have usable NAC input")

    train_rows = data[data["split"] == "train"].reset_index(drop=True)
    cal_rows = data[data["split"] == "cal"].reset_index(drop=True)
    val_rows = data[data["split"] == "val"].reset_index(drop=True)
    test_rows = data[data["split"] == "test"].reset_index(drop=True)
    for name, rows in (("train", train_rows), ("cal", cal_rows), ("val", val_rows), ("test", test_rows)):
        if rows.empty:
            raise SystemExit(f"split {name} has no usable tiles after NAC filtering")

    y_train = train_rows[["slope3m_g10", "slope3m_g15"]].to_numpy(dtype=np.float64)
    y_test = test_rows[["slope3m_g10", "slope3m_g15"]].to_numpy(dtype=np.float64)
    test_groups = test_rows["group_id"].to_numpy()

    results = {
        "tiles_all_splits": {split: int((labels["split"] == split).sum()) for split in ("train", "cal", "val", "test")},
        "tiles_used_by_split": {split: int((data["split"] == split).sum()) for split in ("train", "cal", "val", "test")},
        "groups_used_by_split": {split: int(data[data["split"] == split]["group_id"].nunique())
                                 for split in ("train", "cal", "val", "test")},
        "nac_availability": {
            "threshold": args.min_nac_valid,
            "usable": int(usable.sum()),
            "excluded": int((~usable).sum()),
            "excluded_by_split": excluded.groupby("split").size().to_dict(),
            "excluded_all_nan_tiles": int(excluded["nac_all_nan"].sum()),
            "excluded_mean_g10": float(excluded["slope3m_g10"].mean()) if len(excluded) else None,
            "excluded_mean_g15": float(excluded["slope3m_g15"].mean()) if len(excluded) else None,
            "excluded_unsafe_rate": float((excluded["slope3m_g10"] > 0.05).mean()) if len(excluded) else None,
            "note": "tiles whose NAC layer is unusable are excluded from every model so all "
                    "comparisons share one test set; see reports for the resulting selection bias",
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
    model_metrics = {}
    for name, (target, threshold) in RULES.items():
        y_true = test_rows[target].to_numpy(dtype=np.float64)
        probabilities = np.full(len(y_true), constant["prevalence_g10_gt_0.05" if name == "g10_gt_0.05"
                                                   else "prevalence_g15_gt_0.02"])
        m = metrics(y_true, np.full(len(y_true), constant["y1" if target.endswith("g10") else "y2"]),
                    (name, threshold), np.log(probabilities / (1 - probabilities)))
        risk = 1 - probabilities
        model_metrics[name] = {"metrics": m, "risk_coverage": risk_coverage(y_true, risk, (name, threshold))}
    results["models"]["constant"] = {"params": constant, "test": model_metrics}

    # ---- Baseline B: coarse 60 m slope ----
    features = data[COARSE_FEATURES].to_numpy(dtype=np.float64)
    finite_rows = np.isfinite(features).all(axis=1)
    if not finite_rows.all():
        print(f"WARNING: {int((~finite_rows).sum())} tiles have non-finite coarse features; "
              "imputing with column medians")
    medians = np.nanmedian(features, axis=0)
    features = np.where(np.isfinite(features), features, medians)
    scaler = StandardScaler().fit(features[data["split"] == "train"])
    features_scaled = scaler.transform(features)
    split_index = {split: np.where(data["split"].to_numpy() == split)[0] for split in ("train", "cal", "val", "test")}
    ridge = Ridge(alpha=1.0).fit(features_scaled[split_index["train"]], y_train)
    coarse_predictions = ridge.predict(features_scaled[split_index["test"]])
    coarse_event, coarse_degenerate = {}, {}
    for name, (target, threshold) in RULES.items():
        truth = (train_rows[target].to_numpy() > threshold).astype(int)
        model, degenerate = fit_logistic(features_scaled[split_index["train"]], truth)
        coarse_degenerate[name] = degenerate
        coarse_event[name] = logistic_probability(
            model, features_scaled[split_index["test"]], float(truth.mean())
        )
    model_metrics = {}
    for index, (name, (target, threshold)) in enumerate(RULES.items()):
        y_true = test_rows[target].to_numpy(dtype=np.float64)
        probability = coarse_event[name]
        m = metrics(y_true, coarse_predictions[:, index], (name, threshold),
                    np.log(np.clip(probability, 1e-7, 1 - 1e-7) / (1 - np.clip(probability, 1e-7, 1 - 1e-7))))
        model_metrics[name] = {"metrics": m, "risk_coverage": risk_coverage(y_true, 1 - probability, (name, threshold))}
    results["models"]["coarse_ridge"] = {
        "ridge_coef": ridge.coef_.tolist(),
        "ridge_intercept": ridge.intercept_.tolist(),
        "logistic_degenerate": coarse_degenerate,
        "test": model_metrics,
    }

    # ---- Model C: CNN on aligned NAC ----
    cnn = train_cnn(cache, data, args, device)
    results["models"]["cnn"] = {
        "best_epoch": cnn["best_epoch"],
        "best_val_mae": cnn["best_val_mae"],
        "history": cnn["history"],
        "input_size": args.size,
        "augment": args.augment,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    cal_fractions, cal_scores = predict_cnn(cnn["model"], cal_rows, cache, args.batch, device)
    test_fractions, test_scores = predict_cnn(cnn["model"], test_rows, cache, args.batch, device)
    val_fractions, val_scores = predict_cnn(cnn["model"], val_rows, cache, args.batch, device)
    model_metrics = {}
    for index, (name, (target, threshold)) in enumerate(RULES.items()):
        y_true = test_rows[target].to_numpy(dtype=np.float64)
        cal_truth = (cal_rows[target].to_numpy() > threshold).astype(int)
        slope, intercept, platt_degenerate = platt_scale(cal_scores, cal_truth)
        calibrated = 1 / (1 + np.exp(-(slope * test_scores + intercept)))
        raw_probability = 1 / (1 + np.exp(-test_scores))
        raw_metrics = metrics(y_true, test_fractions[:, index], (name, threshold),
                              np.log(np.clip(raw_probability, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_probability, 1e-7, 1 - 1e-7))))
        calibrated_metrics = metrics(y_true, test_fractions[:, index], (name, threshold),
                                     np.log(np.clip(calibrated, 1e-7, 1 - 1e-7) / (1 - np.clip(calibrated, 1e-7, 1 - 1e-7))))
        model_metrics[name] = {
            "metrics_raw": raw_metrics,
            "metrics_calibrated": calibrated_metrics,
            "platt": {"slope": slope, "intercept": intercept, "degenerate": platt_degenerate,
                      "cal_unsafe_rate": float(cal_truth.mean())},
            "risk_coverage_raw": risk_coverage(y_true, 1 - raw_probability, (name, threshold)),
            "risk_coverage_calibrated": risk_coverage(y_true, 1 - calibrated, (name, threshold)),
        }
    results["models"]["cnn"]["test"] = model_metrics

    # ---- Grouped bootstrap on the primary rule ----
    target, threshold = RULES["g10_gt_0.05"]
    y_true = test_rows[target].to_numpy(dtype=np.float64)
    truth = (y_true > threshold).astype(np.float64)
    predictions = {
        "constant": np.full(len(y_true), constant["y1"]),
        "coarse_ridge": coarse_predictions[:, 0],
        "cnn": test_fractions[:, 0],
    }
    probabilities = {"constant": np.full(len(y_true), constant["prevalence_g10_gt_0.05"]),
                     "coarse_ridge": coarse_event["g10_gt_0.05"],
                     "cnn": 1 / (1 + np.exp(-test_scores))}
    bootstrap = {}
    for name, prediction in predictions.items():
        probability = np.clip(probabilities[name], 1e-7, 1 - 1e-7)
        bootstrap[name] = {
            "mae_g10": grouped_bootstrap(y_true, test_groups,
                                         lambda v, i: float(np.abs(v - prediction[i]).mean()), args.bootstrap, args.seed),
            "brier_g10_gt_0.05": grouped_bootstrap(truth, test_groups,
                                                   lambda v, i: float(((probability[i] - v) ** 2).mean()),
                                                   args.bootstrap, args.seed),
        }
    results["grouped_bootstrap"] = bootstrap
    results["test_targets"] = {
        "unsafe_rate_g10_gt_0.05": float(truth.mean()),
        "unsafe_rate_g15_gt_0.02": float((test_rows['slope3m_g15'] > 0.02).mean()),
        "g10_mean": float(y_true.mean()),
        "g10_std": float(y_true.std()),
    }
    distribution = {}
    for split in ("train", "cal", "val", "test"):
        part = data[data["split"] == split]
        distribution[split] = {
            "tiles": int(len(part)),
            "g10_quantiles": {q: round(float(part["slope3m_g10"].quantile(q)), 5)
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "g15_quantiles": {q: round(float(part["slope3m_g15"].quantile(q)), 5)
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "g10_mean": round(float(part["slope3m_g10"].mean()), 5),
            "unsafe_rate_g10_gt_0.05": round(float((part["slope3m_g10"] > 0.05).mean()), 5),
            "unsafe_rate_g15_gt_0.02": round(float((part["slope3m_g15"] > 0.02).mean()), 5),
            "ltm_zones": int(part["LTM_CODE"].nunique()),
        }
    results["label_distribution"] = distribution

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    prediction_export = {
        "sample_id": test_rows["sample_id"], "group_id": test_rows["group_id"],
        "split": test_rows["split"], "y_g10": test_rows["slope3m_g10"], "y_g15": test_rows["slope3m_g15"],
        "constant_g10": predictions["constant"], "coarse_g10": predictions["coarse_ridge"],
        "cnn_g10": predictions["cnn"], "cnn_g15": test_fractions[:, 1],
        "cnn_event_prob": probabilities["cnn"], "coarse_event_prob": probabilities["coarse_ridge"],
    }
    pd.DataFrame(prediction_export).to_csv(args.out / "test_predictions.csv", index=False)

    rc_rows = []
    for model_name, model_result in results["models"].items():
        for rule_name, payload in model_result["test"].items():
            for entry in payload["risk_coverage" if model_name != "cnn" else "risk_coverage_raw"]:
                rc_rows.append({"model": model_name, **entry})
    with (args.out / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rc_rows[0]))
        writer.writeheader()
        writer.writerows(rc_rows)

    for name in ("coarse_ridge", "cnn"):
        payload = results["models"][name]["test"]["g10_gt_0.05"]
        metric_dict = payload["metrics_raw"] if "metrics_raw" in payload else payload["metrics"]
        results["summary_mae_g10_" + name] = round(metric_dict["mae"], 5)
    print(json.dumps({
        "test_tiles": len(test_rows),
        "test_groups": int(test_rows["group_id"].nunique()),
        "excluded_tiles": int((~usable).sum()),
        "mae_g10": {name: round(payload["metrics"]["mae"], 5)
                    for name, payload in results["models"]["constant"]["test"].items()}
        | {name: results["summary_mae_g10_" + name] for name in ("coarse_ridge", "cnn")},
        "brier_g10": {
            "constant": round(results["models"]["constant"]["test"]["g10_gt_0.05"]["metrics"]["brier"], 5),
            "coarse_ridge": round(results["models"]["coarse_ridge"]["test"]["g10_gt_0.05"]["metrics"]["brier"], 5),
            "cnn_raw": round(results["models"]["cnn"]["test"]["g10_gt_0.05"]["metrics_raw"]["brier"], 5),
            "cnn_calibrated": round(results["models"]["cnn"]["test"]["g10_gt_0.05"]["metrics_calibrated"]["brier"], 5),
        },
        "cnn_best_epoch": cnn["best_epoch"],
    }, indent=2))
    print(f"wrote {args.out}/metrics.json, test_predictions.csv, risk_coverage.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
