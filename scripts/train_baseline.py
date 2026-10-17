#!/usr/bin/env python3
"""P1-03/P1-04: first real physical baseline on the hydrated tiles.

Inputs
------
* ``outputs/labels_hydrated.csv`` — real labels from ``scripts/hydrate_tiles.py``
  (raw counts over the native 3 m slope grid, plus viewing geometry and DTM-derived
  roughness/relief attributes).
* Hydrated NetCDF tiles under ``data/hydrated/<sample_id>/``.

Alignment
---------
Every grid is resampled with ``map_coordinates`` using the per-file ``x``/``y``
coordinate vectors: the NAC and the 60 m slope grid are both sampled onto the
native 3 m slope grid. No channel assumes "pixel (i, j) equals pixel (i, j)".

Photometry
----------
Two NAC channels are kept deliberately:

* ``nac_contrast`` — ``(I - median(I)) / median(I)``: preserves the *absolute*
  shading contrast that carries the slope signal, so a flat mare is not stretched
  into looking like a crater wall.
* ``nac_texture``  — per-tile percentile stretch, which exposes local structure.

Per-tile scalar features (NAC contrast statistics and viewing geometry) are
concatenated after global pooling. Geometry is given to every CNN variant
(it is metadata, not imagery); the NAC-derived contrast scalars are only given to
variants that actually consume the NAC channels, so the fixed-architecture
comparison still isolates "does the optical image add information".

Targets and rules
-----------------
``y1 = g(10 deg)``, ``y2 = g(15 deg)`` (area fractions of valid native pixels).
Pre-registered engineering rules: ``unsafe = 1[y1 > 0.05]`` (primary) and
``1[y2 > 0.02]``. Extra balanced-prevalence rules are pre-registered from the
*training* distribution only and evaluated on test; see ``RULES``.

Models
------
* ``constant``     — train mean / train prevalence.
* ``coarse_ridge`` — ridge + logistic regression on the 60 m slope summary features.
* ``cnn_<variant>`` — same architecture, tiles, seed and early-stopping protocol:
    * ``nac``         : NAC contrast + texture,
    * ``coarse_grid`` : the 60 m slope grid resampled onto the 3 m grid,
    * ``fusion``      : all channels.

The ``cal`` split is used only for Platt calibration of event probabilities.
Metrics carry group-level bootstrap CIs; the group is the leakage unit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
# Rules are fixed before evaluation. ``g10_gt_0.05`` / ``g15_gt_0.02`` are the
# original proxies; the ``_bal_`` entries are balanced-prevalence rules whose
# thresholds were read off the *training* quantiles (never the test set).
RULES = {
    "g10_gt_0.05": ("slope3m_g10", 0.05),
    "g15_gt_0.02": ("slope3m_g15", 0.02),
    "g10_gt_0.15": ("slope3m_g10", 0.15),
    "g10_gt0.10_or_g15_gt0.03": ("composite", 0.0),
}
COMPOSITE_RULE = "g10_gt0.10_or_g15_gt0.03"
COMPOSITE_G10 = 0.10
COMPOSITE_G15 = 0.03
PRIMARY_RULE = "g10_gt_0.05"
CHANNELS = ("nac_contrast", "nac_texture", "coarse_slope")
CNN_VARIANTS = {"nac": (0, 1), "coarse_grid": (2,), "fusion": (0, 1, 2)}
GEOMETRY_SCALARS = ("incidence", "emission", "phase", "az_sin", "az_cos")
NAC_SCALARS = ("log_contrast", "log_median")
VARIANT_SCALARS = {
    "nac": NAC_SCALARS + GEOMETRY_SCALARS,
    "coarse_grid": GEOMETRY_SCALARS,
    "fusion": NAC_SCALARS + GEOMETRY_SCALARS,
}


def ablation_features(variant: str, *, use_geometry: bool = True,
                      use_nac_photometry: bool = True) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Select a paired ablation without changing the cached input or global model specs.

    --no-nac-photometry preserves the NAC texture channel, dropping only the
    contrast-preserving NAC image channel and NAC brightness/contrast scalars.
    """
    channels = CNN_VARIANTS[variant]
    scalars = VARIANT_SCALARS[variant]
    if not use_geometry:
        scalars = tuple(name for name in scalars if name not in GEOMETRY_SCALARS)
    if not use_nac_photometry:
        channels = tuple(c for c in channels if CHANNELS[c] != "nac_contrast")
        scalars = tuple(name for name in scalars if name not in NAC_SCALARS)
    return channels, scalars
COVERAGES = (0.2, 0.4, 0.6, 0.8)
SLOPE_MAX_DEG = 90.0
CONTRAST_EPS = 1e-6


def open_dataset(path: Path):
    import hdf5plugin  # noqa: F401
    import h5netcdf  # noqa: F401
    import xarray as xr
    return xr.open_dataset(path, engine="h5netcdf")


def sample_grid_on_coordinates(values: np.ndarray, source_x: np.ndarray, source_y: np.ndarray,
                               target_x: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    """Bilinear sample of a source grid onto target coordinate vectors."""
    if len(source_x) > 1 and source_x[-1] != source_x[0]:
        step_x = (source_x[-1] - source_x[0]) / (len(source_x) - 1)
        cols = (np.asarray(target_x, dtype=np.float64) - source_x[0]) / step_x
    else:
        cols = np.zeros(len(target_x), dtype=np.float64)  # single column: no axis to map
    if len(source_y) > 1 and source_y[-1] != source_y[0]:
        step_y = (source_y[-1] - source_y[0]) / (len(source_y) - 1)
        rows = (np.asarray(target_y, dtype=np.float64) - source_y[0]) / step_y
    else:
        rows = np.zeros(len(target_y), dtype=np.float64)
    fine, coarse = np.meshgrid(rows, cols, indexing="ij")
    return map_coordinates(values, [fine, coarse], order=1, mode="nearest")


def grid_offsets_m(source_x, source_y, target_x, target_y) -> dict:
    """Extent disagreement between two grids in metres (0 means identical bounds)."""
    return {
        "dx_min_m": float(abs(source_x[0] - target_x[0])),
        "dx_max_m": float(abs(source_x[-1] - target_x[-1])),
        "dy_min_m": float(abs(source_y[0] - target_y[0])),
        "dy_max_m": float(abs(source_y[-1] - target_y[-1])),
    }


def load_aligned_tile(nac_path: Path, slope3m_path: Path, slope60m_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (nac aligned to the 3 m grid, 60 m slope aligned to the 3 m grid, info)."""
    with open_dataset(slope3m_path) as slope3m:
        target_x = np.asarray(slope3m.x.values, dtype=np.float64)
        target_y = np.asarray(slope3m.y.values, dtype=np.float64)
        gsd_slope = abs(float(slope3m.attrs.get("pix_res_x", 0.0)))
    with open_dataset(nac_path) as nac:
        nac_var = next(n for n, v in nac.data_vars.items() if v.ndim == 2)
        nac_values = np.asarray(nac[nac_var].values, dtype=np.float32)
        nac_x = np.asarray(nac.x.values, dtype=np.float64)
        nac_y = np.asarray(nac.y.values, dtype=np.float64)
        gsd_nac = abs(float(nac.attrs.get("pix_res_x", 0.0)))
    aligned = sample_grid_on_coordinates(nac_values, nac_x, nac_y, target_x, target_y)
    nac_offsets = grid_offsets_m(nac_x, nac_y, target_x, target_y)

    with open_dataset(slope60m_path) as coarse:
        coarse_var = next(n for n, v in coarse.data_vars.items() if v.ndim == 2)
        coarse_values = np.asarray(coarse[coarse_var].values, dtype=np.float64)
        coarse_x = np.asarray(coarse.x.values, dtype=np.float64)
        coarse_y = np.asarray(coarse.y.values, dtype=np.float64)
        gsd_coarse = abs(float(coarse.attrs.get("pix_res_x", 0.0)))
    coarse_offsets = grid_offsets_m(coarse_x, coarse_y, target_x, target_y)
    coarse_finite = np.isfinite(coarse_values)
    if not coarse_finite.all():
        fill = float(np.nanmedian(coarse_values)) if coarse_finite.any() else 0.0
        coarse_values = np.where(coarse_finite, coarse_values, fill)
    coarse_aligned = sample_grid_on_coordinates(coarse_values, coarse_x, coarse_y, target_x, target_y)
    coarse_aligned = np.clip(coarse_aligned, 0.0, SLOPE_MAX_DEG) / SLOPE_MAX_DEG

    valid = np.isfinite(aligned)
    if valid.any():
        median = float(np.nanmedian(aligned))
        p1, p99 = (float(v) for v in np.nanpercentile(aligned, [1, 99]))
    else:
        median, p1, p99 = 0.0, 0.0, 1.0
    info = {
        "gsd_nac": gsd_nac,
        "gsd_slope": gsd_slope,
        "gsd_coarse": gsd_coarse,
        "origin_shift_m": float(abs(nac_x[0] - target_x[0])),
        "origin_shift_in_nac_pixels": float(abs(nac_x[0] - target_x[0]) / gsd_nac) if gsd_nac else None,
        "aligned_shape": list(aligned.shape),
        "nac_p1": p1,
        "nac_p99": p99,
        "nac_median": median,
        "nac_p99_p1": p99 - p1,
        "nac_valid_fraction": float(valid.mean()),
        "nac_all_nan": bool(not valid.any()),
        "coarse_offsets_m": coarse_offsets,
        "coarse_valid_fraction_prefill": float(coarse_finite.mean()),
        "nac_offsets_m": nac_offsets,
        "coarse_aligned_shape": list(coarse_aligned.shape),
    }
    return aligned, coarse_aligned, info


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


def contrast_normalize(array: np.ndarray) -> np.ndarray:
    """Absolute-contrast channel: (I - median) / median, clipped to a fixed range.

    Unlike a per-tile percentile stretch this keeps the *magnitude* of the shading
    contrast, so flat terrain stays flat instead of being amplified to look rough.
    """
    valid = np.isfinite(array)
    if not valid.any():
        return np.zeros_like(array, dtype=np.float32)
    median = float(np.nanmedian(array))
    scale = abs(median) + CONTRAST_EPS
    relative = (np.where(valid, array, median) - median) / scale
    return np.nan_to_num(np.clip(relative, -1.0, 3.0), nan=0.0, posinf=3.0, neginf=-1.0).astype(np.float32)


def build_cache(labels: pd.DataFrame, cache_path: Path, size: int, force: bool = False) -> tuple[dict, dict]:
    """Cache aligned inputs as (3, size, size): contrast, texture, coarse slope."""
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
        aligned, coarse_aligned, info = load_aligned_tile(
            Path(row["layers_nac"]), Path(row["layers_slope_3m"]), Path(row["layers_coarse_slope"]))
        small = zoom(aligned, (size / aligned.shape[0], size / aligned.shape[1]), order=1)
        channels = np.stack([
            contrast_normalize(small),
            robust_normalize(small),
            zoom(coarse_aligned, (size / coarse_aligned.shape[0], size / coarse_aligned.shape[1]), order=1),
        ]).astype(np.float16)
        cache[sample_id] = channels
        info["cache_shape"] = list(channels.shape)
        meta[sample_id] = info
        if (index + 1) % 100 == 0:
            print(f"  aligned {index + 1}/{len(labels)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, __meta__=json.dumps(meta), **cache)
    return cache, meta


def build_scalar_table(labels: pd.DataFrame, cache_meta: dict) -> pd.DataFrame:
    """Unscaled per-tile scalar features; scaling is fitted on the train split only."""
    table = pd.DataFrame({"sample_id": labels["sample_id"].astype(str).to_numpy()}).set_index("sample_id")
    table["log_contrast"] = [math.log(float(cache_meta.get(s, {}).get("nac_p99_p1", 0.0)) + CONTRAST_EPS)
                             for s in table.index]
    table["log_median"] = [math.log(abs(float(cache_meta.get(s, {}).get("nac_median", 0.0))) + CONTRAST_EPS)
                           for s in table.index]
    geometry = labels.set_index(labels["sample_id"].astype(str))
    azimuth = np.radians(geometry["sub_solar_ground_azimuth_deg"].astype(float).to_numpy())
    table["incidence"] = geometry["incidence_angle_deg"].astype(float).to_numpy() / 90.0
    table["emission"] = geometry["emission_angle_deg"].astype(float).to_numpy() / 90.0
    table["phase"] = geometry["phase_angle_deg"].astype(float).to_numpy() / 120.0
    table["az_sin"] = np.sin(azimuth)
    table["az_cos"] = np.cos(azimuth)
    return table


class TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 1, n_scalars: int = 0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 + n_scalars, 4))

    def forward(self, x: torch.Tensor, scalars: torch.Tensor | None = None) -> torch.Tensor:
        features = self.pool(self.features(x)).flatten(1)
        if scalars is not None and scalars.shape[1]:
            features = torch.cat([features, scalars], dim=1)
        return self.head(features)


def batch_tensor(ids: list[str], cache: dict, device: torch.device, channels: tuple[int, ...]) -> torch.Tensor:
    stacked = np.stack([cache[i][list(channels)] for i in ids]).astype(np.float32)
    return torch.from_numpy(stacked).to(device)


def scalar_tensor(ids: list[str], table: pd.DataFrame | None, columns: tuple[str, ...],
                  device: torch.device) -> torch.Tensor | None:
    if table is None or not columns:
        return None
    values = table.loc[ids, list(columns)].to_numpy(dtype=np.float32)
    return torch.from_numpy(values).to(device)


def train_variant(cache, data: pd.DataFrame, args, device, channels: tuple[int, ...],
                  scalar_table: pd.DataFrame | None = None,
                  scalar_columns: tuple[str, ...] = ()) -> dict:
    train_rows = data[data["split"] == "train"].reset_index(drop=True)
    val_rows = data[data["split"] == "val"].reset_index(drop=True)
    y_train = train_rows[["slope3m_g10", "slope3m_g15"]].to_numpy(dtype=np.float32)
    # Each rule receives its own event logit, not a shared g10 classification score.
    e_train = np.column_stack([
        rule_truth(train_rows[target].to_numpy(), target, threshold).astype(np.float32)
        for target, threshold in (RULES["g10_gt_0.05"], RULES["g15_gt_0.02"])
    ])
    y_val = val_rows[["slope3m_g10", "slope3m_g15"]].to_numpy(dtype=np.float32)

    seed = args.seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyCNN(in_channels=len(channels), n_scalars=len(scalar_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    def evaluate(rows):
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(rows), args.batch):
                ids = rows["sample_id"].iloc[start:start + args.batch].tolist()
                outputs.append(model(batch_tensor(ids, cache, device, channels),
                                     scalar_tensor(ids, scalar_table, scalar_columns, device)).cpu().numpy())
        out = np.concatenate(outputs)
        return 1 / (1 + np.exp(-out[:, :2])), out[:, 2:4]

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
            scalars = scalar_tensor(ids, scalar_table, scalar_columns, device)
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
            out = model(x, scalars)
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
            "best_val_mae": best["val_mae"], "seed": seed, "channels": channels,
            "scalar_table": scalar_table, "scalar_columns": scalar_columns}


def predict_variant(result, rows, cache, channels, batch, device) -> tuple[np.ndarray, np.ndarray]:
    model = result["model"]
    scalar_table = result.get("scalar_table")
    scalar_columns = result.get("scalar_columns", ())
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            ids = rows["sample_id"].iloc[start:start + batch].tolist()
            outputs.append(model(batch_tensor(ids, cache, device, channels),
                                 scalar_tensor(ids, scalar_table, scalar_columns, device)).cpu().numpy())
    out = np.concatenate(outputs)
    return 1 / (1 + np.exp(-out[:, :2])), out[:, 2:4]


def rule_truth(values: np.ndarray, target: str, threshold: float) -> np.ndarray:
    """Boolean event for a rule; 'composite' is the OR of two pre-registered thresholds."""
    values = np.asarray(values, dtype=np.float64)
    if target == "composite":
        return values > threshold
    return values > threshold


def rule_event(row_like) -> np.ndarray:
    """Evaluate every rule for a frame (vectorised helper for the composite rule)."""
    g10 = np.asarray(row_like["slope3m_g10"], dtype=np.float64)
    g15 = np.asarray(row_like["slope3m_g15"], dtype=np.float64)
    events = {}
    for name, (target, threshold) in RULES.items():
        if name == COMPOSITE_RULE:
            events[name] = (g10 > 0.10) | (g15 > 0.03)
        else:
            events[name] = rule_truth(g10 if target.endswith("g10") else g15, target, threshold)
    return events


def platt_scale(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float, bool]:
    """Fit Platt scaling; return (slope, intercept, degenerate)."""
    targets = np.asarray(targets, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(targets)) < 2 or float(np.std(scores)) < 1e-12:
        # Single class or a constant score: the calibration is not identifiable.
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


def file_fingerprint(path: Path) -> dict:
    """Size + SHA256 for provenance; UNKNOWN rather than guessing when unavailable."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "bytes": None, "sha256": "UNKNOWN"}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def rule_score(rule_name: str, fractions: np.ndarray) -> np.ndarray:
    """Fixed, model-uniform ranking score for the balanced rules.

    The two original rules have dedicated event heads; the balanced rules are
    ranked by a normalised regression margin (documented as a plug-in, the
    calibrated column is the fitted one).
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    if rule_name == "g10_gt_0.15":
        return np.clip(fractions[:, 0] / 0.15, 0.0, 1.0)
    if rule_name == COMPOSITE_RULE:
        return np.clip(np.maximum(fractions[:, 0] / COMPOSITE_G10,
                                  fractions[:, 1] / COMPOSITE_G15), 0.0, 1.0)
    raise KeyError(f"{rule_name} has a dedicated event head, not a derived score")


def metrics(y: np.ndarray, p: np.ndarray, rule: tuple[str, float] | None = None,
            event_probability: np.ndarray | None = None,
            rule_name: str | None = None, truth: np.ndarray | None = None) -> dict:
    result = {
        "mae": float(np.abs(p - y).mean()),
        "rmse": float(np.sqrt(((p - y) ** 2).mean())),
    }
    if rule is not None and event_probability is not None:
        name, threshold = rule
        if truth is None:
            truth = (np.asarray(y) > threshold).astype(np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        probability = np.clip(event_probability, 1e-7, 1 - 1e-7)
        result.update({
            "rule": rule_name or name,
            "target": name,
            "threshold": threshold,
            "unsafe_rate": float(truth.mean()),
            "brier": float(((probability - truth) ** 2).mean()),
            "nll": float(-(truth * np.log(probability) + (1 - truth) * np.log(1 - probability)).mean()),
            "auc": float(roc_auc_score(truth, probability)) if 0 < truth.sum() < len(truth) else None,
            "brier_constant": float(((truth.mean() - truth) ** 2).mean()),
        })
    return result


def risk_coverage(y: np.ndarray, unsafe_probability: np.ndarray, rule_name: str,
                  coverages=COVERAGES, truth: np.ndarray | None = None) -> list[dict]:
    """Accept lowest P(unsafe) first; do not pass P(safe)."""
    _, threshold = RULES[rule_name]
    if truth is None:
        truth = (np.asarray(y) > threshold).astype(int)
    else:
        truth = np.asarray(truth).astype(int)
    unsafe_probability = np.asarray(unsafe_probability, dtype=np.float64)
    if unsafe_probability.shape != truth.shape or not np.isfinite(unsafe_probability).all():
        raise ValueError("unsafe_probability must be a finite vector aligned with y")
    if np.any((unsafe_probability < 0) | (unsafe_probability > 1)):
        raise ValueError("unsafe_probability must lie in [0, 1]")
    order = np.argsort(unsafe_probability, kind="mergesort")
    ranked_scores = unsafe_probability[order]
    ranked_truth = truth[order]
    unsafe_cumulative = np.cumsum(ranked_truth)
    rows = []
    for coverage in coverages:
        k = max(1, int(round(coverage * len(y))))
        # If the coverage boundary cuts through equal scores, take the expected
        # unsafe count under uniform random tie-breaking, not CSV row order.
        cutoff = ranked_scores[k - 1]
        lo = int(np.searchsorted(ranked_scores, cutoff, side="left"))
        hi = int(np.searchsorted(ranked_scores, cutoff, side="right"))
        unsafe_below = int(unsafe_cumulative[lo - 1]) if lo else 0
        tie_at_boundary = lo < k < hi
        if tie_at_boundary:
            unsafe_in_tie = int(ranked_truth[lo:hi].sum())
            unsafe_accepted = unsafe_below + (k - lo) / (hi - lo) * unsafe_in_tie
        else:
            unsafe_accepted = float(unsafe_cumulative[k - 1])
        rows.append({
            "rule": rule_name,
            "threshold": threshold,
            "coverage_requested": coverage,
            "coverage_actual": round(k / len(y), 4),
            "accepted_n": k,
            "test_n": len(y),
            "unsafe_accepted_n": unsafe_accepted,
            "boundary_tie_expected": tie_at_boundary,
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
                   rule_name: str, cal_scores=None, cal_truth=None,
                   event_truth: np.ndarray | None = None) -> dict:
    """Uniform payload: primary (calibrated when possible) plus raw metrics."""
    rule = RULES[rule_name]
    index = rule_column_index(rule_name) if rule_name != COMPOSITE_RULE else 0
    raw_probability = 1 / (1 + np.exp(-raw_scores))
    payload = {
        "metrics_raw": metrics(y_true, fractions[:, index], rule, raw_probability, rule_name,
                               truth=event_truth),
        "risk_coverage_raw": risk_coverage(y_true, raw_probability, rule_name, truth=event_truth),
    }
    if cal_scores is not None and cal_truth is not None:
        slope, intercept, degenerate = platt_scale(cal_scores, cal_truth)
        calibrated = 1 / (1 + np.exp(-(slope * raw_scores + intercept)))
        payload["platt"] = {"slope": slope, "intercept": intercept, "degenerate": degenerate,
                            "cal_unsafe_rate": float(cal_truth.mean())}
        payload["metrics_calibrated"] = metrics(y_true, fractions[:, index], rule, calibrated,
                                                rule_name, truth=event_truth)
        payload["risk_coverage_calibrated"] = risk_coverage(y_true, calibrated, rule_name,
                                                            truth=event_truth)
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


HEAD_RULES = ("g10_gt_0.05", "g15_gt_0.02")


def evaluate_all_rules(fractions_test: np.ndarray, fractions_cal: np.ndarray,
                       head_test: dict, head_cal: dict,
                       events_test: dict, events_cal: dict,
                       values_test: dict) -> dict:
    """Evaluate every rule with one uniform mapping: rule -> score, cal score, truth."""
    payload = {}
    for rule_name in RULES:
        value = np.asarray(values_test[rule_name], dtype=np.float64)
        if rule_name in HEAD_RULES and head_test.get(rule_name) is not None:
            raw_scores = head_test[rule_name]
            cal_scores = head_cal[rule_name]
        else:
            raw_scores = logit(rule_score(rule_name, fractions_test))
            cal_scores = logit(rule_score(rule_name, fractions_cal))
        payload[rule_name] = evaluate_model(
            value, fractions_test, raw_scores, rule_name,
            cal_scores=cal_scores, cal_truth=events_cal[rule_name].astype(int),
            event_truth=events_test[rule_name])
    return payload


def rule_values(frame: pd.DataFrame) -> dict:
    """The fraction column each rule's MAE is measured against."""
    return {
        name: (frame["slope3m_g10"] if (target.endswith("g10") or name == COMPOSITE_RULE)
               else frame["slope3m_g15"]).to_numpy(dtype=np.float64)
        for name, (target, threshold) in RULES.items()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels_hydrated.csv"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache_v3.npz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--augment", action="store_true", default=False,
                        help="enable flips/rotations ONLY when --no-geometry is active")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--no-geometry", action="store_true",
                        help="ablate incidence/emission/phase/solar azimuth scalars")
    parser.add_argument("--no-nac-photometry", action="store_true",
                        help="ablate NAC absolute contrast image channel and photometry scalars")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--min-nac-valid", type=float, default=0.99,
                        help="minimum finite fraction of the aligned NAC layer")
    parser.add_argument("--variants", default="nac,coarse_grid,fusion")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.augment and not args.no_geometry:
        parser.error("--augment with solar azimuth geometry is invalid: spatial flips/rot90 "
                     "do not rotate the geometry scalar. Use --no-augment (default), "
                     "or explicitly --no-geometry --augment for a geometry-free control.")

    labels = pd.read_csv(args.labels)
    labels["layers_nac"] = labels["sample_id"].map(lambda s: f"data/hydrated/{s}/nac.nc")
    labels["layers_slope_3m"] = labels["sample_id"].map(lambda s: f"data/hydrated/{s}/slope_3m.nc")
    labels["layers_coarse_slope"] = labels["sample_id"].map(lambda s: f"data/hydrated/{s}/slope_60m.nc")
    missing = [path for path in labels["layers_nac"].head(1) if not Path(path).exists()]
    if missing:
        raise SystemExit(f"hydrated tiles not found: {missing[0]}")
    required = ["incidence_angle_deg", "emission_angle_deg", "phase_angle_deg",
                "sub_solar_ground_azimuth_deg", "dtm3m_roughness_9m_m"]
    absent = [c for c in required if c not in labels.columns]
    if absent:
        raise SystemExit(f"labels file lacks {absent}; run "
                         "`python3 scripts/hydrate_tiles.py --enrich` first")
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

    scalar_table = build_scalar_table(data, cache_meta)
    scalar_scaler = StandardScaler().fit(scalar_table.loc[train_rows["sample_id"].tolist()])
    scalar_table_scaled = pd.DataFrame(scalar_scaler.transform(scalar_table),
                                       index=scalar_table.index, columns=scalar_table.columns)

    results = {
        "protocol": {
            "models": ["constant", "coarse_ridge"] + [f"cnn_{v}" for v in CNN_VARIANTS],
            "shared_train_tiles": int(len(train_rows)),
            "shared_test_tiles": int(len(test_rows)),
            "test_groups": int(test_rows["group_id"].nunique()),
            "seed": args.seed,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "batch": args.batch,
            "lr": args.lr,
            "min_nac_valid": args.min_nac_valid,
            "input_size": args.size,
            "augment": args.augment,
            "augmentation_policy": ("geometry-free flips/rot90" if args.augment
                                    else "no spatial transforms; solar azimuth unchanged"),
            "ablation": {"use_geometry": not args.no_geometry,
                         "use_nac_photometry": not args.no_nac_photometry},
            "channels": list(CHANNELS),
            "variant_channels": {
                k: [CHANNELS[c] for c in ablation_features(
                    k, use_geometry=not args.no_geometry,
                    use_nac_photometry=not args.no_nac_photometry)[0]]
                for k in CNN_VARIANTS
            },
            "variant_scalars": {
                k: list(ablation_features(
                    k, use_geometry=not args.no_geometry,
                    use_nac_photometry=not args.no_nac_photometry)[1])
                for k in CNN_VARIANTS
            },
            "scalar_scaling": "StandardScaler fitted on the train split only",
            "calibration": "Platt scaling fitted on the cal split, applied to cnn_* probabilities",
            "rules": {k: {"target": v[0], "threshold": v[1]} for k, v in RULES.items()},
            "composite_rule": COMPOSITE_RULE,
            "contrast_channel": "(I - median(I)) / median(I), clipped to [-1, 3]",
            "provenance": {
                "manifest": file_fingerprint(args.manifest),
                "labels": file_fingerprint(args.labels),
                "cache": file_fingerprint(args.cache),
            },
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
            "excluded_unsafe_rate": float((excluded["slope3m_g10"] > 0.05).mean()) if len(excluded) else None,
        },
        "alignment_check": {row["sample_id"]: cache_meta.get(row["sample_id"], {})
                            for _, row in data.head(5).iterrows()},
        "coarse_alignment": {
            "max_dx_m": max((m.get("coarse_offsets_m", {}).get("dx_max_m", 0.0)
                             for m in cache_meta.values()), default=0.0),
            "max_dy_m": max((m.get("coarse_offsets_m", {}).get("dy_max_m", 0.0)
                             for m in cache_meta.values()), default=0.0),
            "tiles_with_prefill_nan": sum(1 for m in cache_meta.values()
                                          if m.get("coarse_valid_fraction_prefill", 1.0) < 1.0),
        },
        "models": {},
    }

    # ---- Baseline A: constant ----
    train_events = rule_event(train_rows)
    cal_events = rule_event(cal_rows)
    test_events = rule_event(test_rows)
    values_test = rule_values(test_rows)
    constant = {
        "y1": float(y_train[:, 0].mean()),
        "y2": float(y_train[:, 1].mean()),
        "prevalence": {name: float(train_events[name].mean()) for name in RULES},
    }
    constant_fractions_test = np.full((len(test_rows), 2), [constant["y1"], constant["y2"]])
    constant_fractions_cal = np.full((len(cal_rows), 2), [constant["y1"], constant["y2"]])
    constant_head_test = {name: logit(np.full(len(test_rows), constant["prevalence"][name]))
                          for name in HEAD_RULES}
    constant_head_cal = {name: logit(np.full(len(cal_rows), constant["prevalence"][name]))
                         for name in HEAD_RULES}
    payload = evaluate_all_rules(constant_fractions_test, constant_fractions_cal,
                                 constant_head_test, constant_head_cal,
                                 test_events, cal_events, values_test)
    results["models"]["constant"] = {"params": constant, "test": payload}

    # ---- Baseline B: coarse 60 m slope summary features ----
    features = data[COARSE_FEATURES].to_numpy(dtype=np.float64)
    medians = np.nanmedian(features, axis=0)
    features = np.where(np.isfinite(features), features, medians)
    scaler = StandardScaler().fit(features[data["split"] == "train"])
    features_scaled = scaler.transform(features)
    split_index = {s: np.where(data["split"].to_numpy() == s)[0] for s in ("train", "cal", "val", "test")}
    ridge = Ridge(alpha=1.0).fit(features_scaled[split_index["train"]], y_train)
    # The ridge is trained on the same usable train tiles as the CNNs; its event
    # probabilities get the same cal-split Platt treatment.
    ridge_predictions = ridge.predict(features_scaled[split_index["test"]])
    ridge_cal_predictions = ridge.predict(features_scaled[split_index["cal"]])
    degenerate_flags = {}
    ridge_head_test, ridge_head_cal = {}, {}
    for name in HEAD_RULES:
        model, degenerate = fit_logistic(features_scaled[split_index["train"]],
                                        train_events[name].astype(int))
        degenerate_flags[name] = degenerate
        fallback = float(train_events[name].mean())
        ridge_head_test[name] = logit(logistic_probability(
            model, features_scaled[split_index["test"]], fallback))
        ridge_head_cal[name] = logit(logistic_probability(
            model, features_scaled[split_index["cal"]], fallback))
    payload = evaluate_all_rules(ridge_predictions, ridge_cal_predictions,
                                 ridge_head_test, ridge_head_cal,
                                 test_events, cal_events, values_test)
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
        channels, columns = ablation_features(
            variant, use_geometry=not args.no_geometry,
            use_nac_photometry=not args.no_nac_photometry)
        print(f"training cnn_{variant}: channels {[CHANNELS[c] for c in channels]}, "
              f"scalars {list(columns)}")
        trained = train_variant(cache, data, args, device, channels,
                                scalar_table=scalar_table_scaled, scalar_columns=columns)
        test_fractions, test_scores = predict_variant(trained, test_rows, cache, channels, args.batch, device)
        cal_fractions, cal_scores = predict_variant(trained, cal_rows, cache, channels, args.batch, device)
        payload = evaluate_all_rules(
            test_fractions, cal_fractions,
            {"g10_gt_0.05": test_scores[:, 0], "g15_gt_0.02": test_scores[:, 1]},
            {"g10_gt_0.05": cal_scores[:, 0], "g15_gt_0.02": cal_scores[:, 1]},
            test_events, cal_events, values_test)
        results["models"][f"cnn_{variant}"] = {
            "channels": [CHANNELS[c] for c in channels],
            "scalars": list(columns),
            "seed": trained["seed"],
            "best_epoch": trained["best_epoch"],
            "best_val_mae": trained["best_val_mae"],
            "history": trained["history"],
            "test": payload,
            "_predictions": {"fractions": test_fractions, "scores": test_scores},
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }

    # ---- Grouped bootstrap on the primary rule ----
    target, threshold = RULES[PRIMARY_RULE]
    y_true = test_rows[target].to_numpy(dtype=np.float64)
    truth = test_events[PRIMARY_RULE].astype(np.float64)
    predictions, probabilities = {}, {}
    for name, model in results["models"].items():
        payload = model["test"][PRIMARY_RULE]
        probabilities[name] = payload["probability"]
        if name == "constant":
            predictions[name] = np.full(len(y_true), model["params"]["y1"])
        elif name == "coarse_ridge":
            predictions[name] = ridge_predictions[:, 0]
        else:
            predictions[name] = model["_predictions"]["fractions"][:, 0]
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

    # ---- Label distributions and rule prevalences ----
    distribution, prevalence = {}, {}
    for split in ("train", "cal", "val", "test"):
        part = data[data["split"] == split]
        events = rule_event(part)
        distribution[split] = {
            "tiles": int(len(part)),
            "g10_quantiles": {str(q): round(float(part["slope3m_g10"].quantile(q)), 5)
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "g15_quantiles": {str(q): round(float(part["slope3m_g15"].quantile(q)), 5)
                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "g10_mean": round(float(part["slope3m_g10"].mean()), 5),
            "roughness_mean_m": round(float(part["dtm3m_roughness_9m_m"].mean()), 5),
            "relief_p95_p5_mean_m": round(float(part["dtm3m_relief_p95_p5_m"].mean()), 5),
            "ltm_zones": int(part["LTM_CODE"].nunique()),
        }
        prevalence[split] = {name: round(float(events[name].mean()), 4) for name in RULES}
    results["label_distribution"] = distribution
    results["rule_prevalence"] = prevalence

    # ---- Write artifacts ----
    args.out.mkdir(parents=True, exist_ok=True)
    export = {
        "sample_id": test_rows["sample_id"], "group_id": test_rows["group_id"],
        "LTM_CODE": test_rows["LTM_CODE"], "y_g10": test_rows["slope3m_g10"],
        "y_g15": test_rows["slope3m_g15"],
        "slope60m_frac_gt_10": test_rows["slope60m_frac_gt_10"],
        "incidence_angle_deg": test_rows["incidence_angle_deg"],
        "dtm3m_roughness_9m_m": test_rows["dtm3m_roughness_9m_m"],
        "dtm3m_relief_p95_p5_m": test_rows["dtm3m_relief_p95_p5_m"],
        "constant_g10": predictions["constant"],
        "constant_g15": np.full(len(test_rows), constant["y2"]),
        "coarse_g10": ridge_predictions[:, 0],
        "coarse_g15": ridge_predictions[:, 1],
    }
    for variant in variant_names:
        model_key = f"cnn_{variant}"
        export[f"cnn_{variant}_g10"] = results["models"][model_key]["_predictions"]["fractions"][:, 0]
        export[f"cnn_{variant}_g15"] = results["models"][model_key]["_predictions"]["fractions"][:, 1]
        export[f"cnn_{variant}_event_prob"] = results["models"][model_key]["test"][PRIMARY_RULE]["probability"]
        # Raw scores are necessary to audit the *classification heads*, not
        # the two correlated slope-fraction regression outputs.
        scores = results["models"][model_key]["_predictions"]["scores"]
        for index, rule_name in enumerate(HEAD_RULES):
            export[f"{model_key}_raw_event_prob_{rule_name}"] = 1 / (1 + np.exp(-scores[:, index]))
    for model_name, model in results["models"].items():
        for rule_name, payload in model["test"].items():
            export[f"{model_name}_prob_{rule_name}"] = payload["probability"]
    for name in RULES:
        export[f"event_{name}"] = test_events[name].astype(int)
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
        for payload in model["test"].values():
            payload.pop("probability", None)
    for model in results["models"].values():
        model.pop("_predictions", None)
    results["environment"] = {
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": sys.version.split()[0],
    }
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = {
        "test_tiles": len(test_rows),
        "test_groups": int(test_rows["group_id"].nunique()),
        "excluded_tiles": int((~usable).sum()),
        "rule_prevalence_test": prevalence["test"],
        "mae_g10": {name: round(model["test"][PRIMARY_RULE]["metrics"]["mae"], 5)
                    for name, model in results["models"].items()},
        "brier_g10": {name: round(model["test"][PRIMARY_RULE]["metrics"]["brier"], 5)
                      for name, model in results["models"].items()},
        "brier_balanced_g10_0.15": {name: round(model["test"]["g10_gt_0.15"]["metrics"]["brier"], 5)
                                    for name, model in results["models"].items()},
        "brier_composite": {name: round(model["test"][COMPOSITE_RULE]["metrics"]["brier"], 5)
                            for name, model in results["models"].items()},
        "best_epochs": {name: model.get("best_epoch") for name, model in results["models"].items()
                        if "best_epoch" in model},
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}/metrics.json, test_predictions.csv, risk_coverage.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
