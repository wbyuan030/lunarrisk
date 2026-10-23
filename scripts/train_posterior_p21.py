"""P2.1 A3: train and evaluate one posterior variant for one seed on the 760 development tiles.

Reuses the existing inputs (``outputs/cache_v3.npz`` and the P1.8 scalar table) instead of rebuilding
them, so nothing about the historical baselines changes and no new download is needed.

Evaluation is deliberately split by role:

* ``train`` fits the encoder, the head and the scalar scaler.
* ``val``  only early-stops the epoch count.
* ``cal``  is *not* touched here; calibration is a separate step and must not be folded into training.
* ``test`` is reported as EXPLORATORY, because the old 110-tile test has been inspected many times.

Every number is reported per split with the number of independent groups, and the event metrics use
the rule AST and posterior integration rather than a thresholded classification.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lunarrisk.probabilistic import (condition_posterior, export_state, from_internal,  # noqa: E402
                                     nll_internal, posterior_from_state, sample_internal,
                                     to_internal)
from lunarrisk.rules import prob_event, standard_rules  # noqa: E402

TARGET_COLUMNS = {"g10": "slope3m_g10", "g15": "slope3m_g15",
                  "roughness_9m": "dtm3m_roughness_9m_m", "relief_p95_p5": "dtm3m_relief_p95_p5_m"}
COVERAGES = (0.10, 0.20, 0.40)


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("train_baseline", ROOT / "scripts" / "train_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset(labels_path: Path, cache_path: Path, baseline):
    import pandas as pd
    labels = pd.read_csv(labels_path)
    labels = labels[labels["split"].isin(["train", "val", "cal", "test"])].reset_index(drop=True)
    data = np.load(cache_path, allow_pickle=True)
    meta = json.loads(str(data["__meta__"]))
    cache = {str(k): data[k] for k in data.files if k != "__meta__"}
    scalars = baseline.build_scalar_table(labels, meta)
    present = [s for s in labels["sample_id"].astype(str) if s in cache and s in scalars.index]
    labels = labels[labels["sample_id"].astype(str).isin(present)].reset_index(drop=True)
    return labels, cache, scalars


def target_matrix(labels):
    import torch
    values = {key: labels[column].astype(float).to_numpy() for key, column in TARGET_COLUMNS.items()}
    z = to_internal(values["g10"], values["g15"], values["roughness_9m"], values["relief_p95_p5"])
    return values, torch.tensor(np.stack([z["z1"], z["z2"], z["z3"], z["z4"]], axis=1),
                                dtype=torch.float32)


def batch(model, ids, cache, scalars, channels, device):
    import torch
    x = np.stack([cache[i][list(channels)] for i in ids]).astype(np.float32)
    s = scalars.loc[ids].to_numpy(dtype=np.float32)
    return torch.tensor(x, device=device), torch.tensor(s, device=device)


def train(model, ids, cache, scalars, z, channels, *, epochs: int, lr: float, device,
          batch_size: int = 32, seed: int = 0, val=None, patience: int = 4):
    import torch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = np.random.default_rng(seed)
    best_state, best_val, stale = None, float("inf"), 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(len(ids))
        total = 0.0
        for start in range(0, len(order), batch_size):
            pick = [ids[i] for i in order[start:start + batch_size]]
            x, s = batch(model, pick, cache, scalars, channels, device)
            zz = z[[order[start:start + batch_size]]].squeeze(0).to(device)
            optimizer.zero_grad()
            internal, _ = nll_internal(model, x, zz, s)
            loss = internal.mean()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(pick)
        train_loss = total / len(ids)
        record = {"epoch": epoch + 1, "train_nll": round(train_loss, 6)}
        if val is not None:
            v_ids, v_z = val
            model.eval()
            with torch.no_grad():
                x, s = batch(model, v_ids, cache, scalars, channels, device)
                v_internal, _ = nll_internal(model, x, v_z.to(device), s)
            record["val_nll"] = round(float(v_internal.mean()), 6)
            if record["val_nll"] < best_val - 1e-4:
                best_val, stale = record["val_nll"], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model, ids, cache, scalars, channels, device, draws=256, seed=0, point_sigma=None,
            temperature=None, covariance_mode="full"):
    """Posterior draws on every split.

    The point variant has no dispersion of its own, so it is evaluated as a Gaussian with the
    *training residual* standard deviation (``point_sigma``). Without that its rule probabilities are
    hard 0/1 and its Brier/NLL would look artificially perfect or perfectly wrong.
    """
    import torch
    if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("calibration temperature must be finite and positive")
    x, s = batch(model, ids, cache, scalars, channels, device)
    if model.variant == "point" and point_sigma is not None:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            mu = model(x, s)
        sigma = torch.as_tensor(point_sigma, dtype=mu.dtype, device=mu.device)
        # Point residual noise is a distribution too: the same cal-fitted T
        # must scale its standard deviation, not just the diagonal/joint models.
        sigma = sigma * (1.0 if temperature is None else temperature)
        noise = torch.randn((draws,) + mu.shape, generator=generator).to(mu.device) * sigma
        z = (mu.unsqueeze(0) + noise).cpu().numpy()
    else:
        if model.variant == "point" and temperature is not None:
            raise ValueError("point model requires point_sigma for temperature calibration")
        z = sample_internal(model, x, s, draws=draws, seed=seed,
                            covariance_mode=covariance_mode).cpu().numpy()
        if temperature is not None:
            # Use the exact latent mean, not the random MC sample average;
            # keeps the same checkpoint's mean invariant under U=0 ablation.
            with torch.no_grad():
                mu = model.parameters_dict(model(x, s))["mu"].cpu().numpy()[None, ...]
            z = mu + temperature * (z - mu)
    dense = {f"z{i + 1}": z[..., i] for i in range(4)}
    return from_internal(dense), z


def attribute_metrics(truth, predicted):
    """MAE / RMSE for the posterior mean, per attribute."""
    out = {}
    for key in TARGET_COLUMNS:
        estimate = np.mean(predicted[key], axis=0) if predicted[key].ndim > 1 else predicted[key]
        error = estimate - truth[key]
        out[key] = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}
    return out


def event_metrics(truth, predicted, rules, seed=0):
    """Brier / NLL per rule, plus UAR/HMR at fixed acceptance coverages."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, rule in rules.items():
        realised = np.asarray(rule.evaluate(truth), dtype=float)
        draws_here = next(iter(predicted.values()))
        if draws_here.ndim > 1:
            probability = np.asarray(prob_event(predicted, rule, axis=0), dtype=float)
        else:                                   # no draws available: point event, reported as such
            probability = np.asarray(rule.evaluate(predicted), dtype=float)
        probability = np.atleast_1d(probability)
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        per_row = -np.where(realised > 0.5, np.log(clipped), np.log(1 - clipped))
        out[name] = {
            "prevalence": float(realised.mean()),
            "brier": float(np.mean((probability - realised) ** 2)),
            "nll": float(per_row.mean()),
        }
        probability = np.atleast_1d(probability)
        for coverage in COVERAGES:
            accepted = np.argsort(probability)[:max(1, int(round(coverage * probability.size)))]
            unsafe = realised > 0.5
            out[name][f"uar@{int(coverage * 100)}"] = (
                float(unsafe[accepted].mean()) if unsafe.sum() else None)
            out[name][f"hmr@{int(coverage * 100)}"] = (
                float(unsafe[accepted].sum() / unsafe.sum()) if unsafe.sum() else None)
            out[name][f"accepted@{int(coverage * 100)}"] = int(accepted.size)
            out[name][f"unsafe_accepted@{int(coverage * 100)}"] = int(unsafe[accepted].sum())
    return out


def per_sample_frame(labels, rows, truth, predicted, probabilities):
    """One row per site: ids, split, truth, posterior mean and per-rule event probability.

    Paired statistics must be computed from these rows (never from a table of means), so every run
    writes them and the comparison script reads them back by ``sample_id``.
    """
    import pandas as pd
    frame = pd.DataFrame({
        "sample_id": labels["sample_id"].astype(str).to_numpy()[rows],
        "group_id": labels["group_id"].astype(str).to_numpy()[rows],
        "split": labels["split"].astype(str).to_numpy()[rows],
    })
    for key in TARGET_COLUMNS:
        frame[f"truth_{key}"] = np.asarray(truth[key], dtype=float)
        draws = np.asarray(predicted[key])
        frame[f"mean_{key}"] = draws.mean(axis=0) if draws.ndim > 1 else draws
    for name, probability in probabilities.items():
        frame[f"p_{name}"] = np.asarray(probability, dtype=float)
    return frame


def group_bootstrap(labels, values, predicted, groups, statistic, draws=200, seed=0):
    """Group-level bootstrap CI for a scalar statistic (independent unit = group)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    estimates = []
    for _ in range(draws):
        pick = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[g] for g in pick])
        estimates.append(statistic(rows))
    return {"p05": float(np.percentile(estimates, 5)), "p50": float(np.percentile(estimates, 50)),
            "p95": float(np.percentile(estimates, 95)), "groups": int(len(unique))}


def fit_temperature(mu, cov, z_true):
    """Maximum-likelihood isotropic temperature for a Gaussian posterior, in closed form.

    Scaling the covariance by T^2 changes the NLL by ``quad / (2 T^2) + k log T + const``, so
    ``dNLL/dT = 0`` gives ``T^2 = mean(quad) / k``. Fitted on the ``cal`` split only - never on test.
    """
    diff = z_true - mu
    chol = np.linalg.cholesky(cov)
    solved = np.linalg.solve(chol, diff[..., None])
    quad = float(np.mean((solved ** 2).sum(axis=(-1, -2))))
    k = mu.shape[-1]
    return float(np.sqrt(max(quad, 1e-12) / k))


def calibrate(model, ids_cal, cache, scalars, z_cal, channels, device, point_sigma=None):
    """Return (temperature, diagnostics) from the calibration split.

    The point model is calibrated too: its dispersion comes from training residuals, which says
    nothing about whether that dispersion is right on held-out rows, so the same temperature is fitted.
    """
    import torch
    with torch.no_grad():
        x, s = batch(model, ids_cal, cache, scalars, channels, device)
        params = model.parameters_dict(model(x, s))
        mu = params["mu"].cpu().numpy()
        if model.variant == "point":
            if point_sigma is None:
                return None, {"reason": "point model without residual sigma"}
            sigma = np.asarray(point_sigma, dtype=np.float64)
            cov = np.tile(np.diag(sigma ** 2), (mu.shape[0], 1, 1))
        else:
            cov = model.covariance(params).cpu().numpy()
    temperature = fit_temperature(mu, cov, z_cal.cpu().numpy())
    return temperature, {"method": "gaussian isotropic temperature (closed form)",
                         "fit_on": "cal", "k": int(mu.shape[-1])}


def zero_low_rank(head):
    """Zero the low-rank columns of the head and return what is needed to restore them."""
    import torch
    start = 2 * 4
    with torch.no_grad():
        saved = (head.weight[start:].clone(), head.bias[start:].clone())
        head.weight[start:].zero_()
        head.bias[start:].zero_()
    return saved


def restore_low_rank(head, saved):
    import torch
    start = 2 * 4
    with torch.no_grad():
        head.weight[start:] = saved[0]
        head.bias[start:] = saved[1]


def file_sha256(path):
    import hashlib
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha():
    """Code revision for provenance; the ROCm container copy has no .git, so env wins first."""
    import os
    import subprocess
    if os.environ.get("LUNARRISK_GIT_SHA"):
        return os.environ["LUNARRISK_GIT_SHA"].strip()
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_checkpoint(path, model, *, scalar_columns, scalar_mean, scalar_std, point_sigma,
                    temperature, seed, labels_path, cache_path, channels, extra=None):
    """Write a reloadable ``*.pt`` payload (the JSON report only keeps a weight hash).

    The payload carries the whole inference state: weights, variant geometry, the train-fitted
    scalar standardisation, the point residual sigma, every calibrated temperature and the
    provenance of the inputs, so a fresh process can reproduce event probabilities exactly.
    """
    import torch
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "variant": model.variant,
        "rank": model.rank,
        "width": model.base_width,
        "in_channels": model.in_channels,
        "n_scalars": model.n_scalars,
        "scalar_columns": [str(c) for c in scalar_columns],
        "scalar_mean": [float(v) for v in scalar_mean],
        "scalar_std": [float(v) for v in scalar_std],
        "channels": [int(c) for c in channels],
        "point_sigma": None if point_sigma is None else [float(v) for v in point_sigma],
        "temperature": None if temperature is None else float(temperature),
        "target_eps": 1e-6,
        "seeds": [int(seed)],
        "labels_sha256": file_sha256(labels_path),
        "cache_sha256": file_sha256(cache_path),
        "git_sha": git_sha(),
        "weights_sha256": export_state(model)["weights_sha256"],
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_checkpoint(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return posterior_from_state(payload), payload


def standardize_with(payload, scalars):
    """Apply the checkpoint's train-fitted standardisation to a raw scalar table."""
    columns = list(payload["scalar_columns"])
    mean = np.asarray(payload["scalar_mean"], dtype=np.float64)
    std = np.asarray(payload["scalar_std"], dtype=np.float64)
    std = np.where(std == 0, 1.0, std)
    table = scalars.copy()
    table[columns] = (table[columns].to_numpy(dtype=np.float64) - mean) / std
    return table


def predict_from_checkpoint(path, ids, cache, scalars, rules, device, draws=256, seed=0,
                            covariance_mode="full"):
    """Reload a saved model and return posterior draws plus per-rule event probabilities."""
    model, payload = load_checkpoint(path)
    model.to(device).eval()
    table = standardize_with(payload, scalars)
    predicted, _ = predict(model, ids, cache, table, tuple(payload["channels"]), device,
                           draws=draws, seed=seed, point_sigma=payload["point_sigma"],
                           temperature=payload["temperature"], covariance_mode=covariance_mode)
    probabilities = {name: np.atleast_1d(np.asarray(prob_event(predicted, rule, axis=0),
                                                    dtype=float))
                     for name, rule in rules.items()}
    return predicted, probabilities, payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "outputs" / "labels_hydrated.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs" / "cache_v3.npz")
    parser.add_argument("--variant", choices=("point", "diagonal", "joint"), default="diagonal")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true",
                        help="torch.use_deterministic_algorithms(True); on this ROCm build it makes "
                             "two same-seed runs bit-identical, which plain seeding does not")
    parser.add_argument("--limit", type=int, default=0, help="cap rows per split (smoke runs)")
    parser.add_argument("--calibrate", action="store_true",
                        help="fit one Gaussian temperature on cal and apply it when sampling")
    parser.add_argument("--ablate-u", action="store_true",
                        help="re-evaluate the same checkpoint with the low-rank factor zeroed")
    parser.add_argument("--ablate-u-calibration-mode", choices=("shared", "refit", "both"),
                        default="shared",
                        help="shared: reuse the full-joint cal T; refit: fit a new T on the zeroed "
                             "model's cal covariance; both: report the two as separate branches. "
                             "The two must never share one comparison table")
    parser.add_argument("--matched-marginals", action="store_true",
                        help="same checkpoint with diag(Sigma_joint): keeps marginal variances and "
                             "removes only the off-diagonal covariance")
    parser.add_argument("--save-checkpoint", type=Path, default=None,
                        help="write a reloadable *.pt with weights, scaler, sigma and all cal temperatures")
    parser.add_argument("--predictions-out", type=Path, default=None,
                        help="per-sample CSV: ids, truth, posterior means and per-rule probabilities")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    baseline = load_baseline_module()
    labels, cache, scalars = load_dataset(args.labels, args.cache, baseline)
    if args.limit:
        labels = labels.groupby("split", group_keys=False).head(args.limit).reset_index(drop=True)

    truth, z = target_matrix(labels)
    ids_all = labels["sample_id"].astype(str).tolist()
    split_of = dict(zip(ids_all, labels["split"]))
    groups = labels["group_id"].astype(str).to_numpy()

    # scalars are standardised on TRAIN only, then applied everywhere
    train_rows = [i for i, s in enumerate(ids_all) if split_of[s] == "train"]
    mean = scalars.iloc[train_rows].mean()
    std = scalars.iloc[train_rows].std().replace(0.0, 1.0)
    scalars = (scalars - mean) / std

    channels = tuple(range(len(baseline.CHANNELS)))
    model = condition_posterior(args.variant, rank=args.rank, n_scalars=scalars.shape[1],
                                in_channels=len(channels)).to(device)

    ids_train = [s for s in ids_all if split_of[s] == "train"]
    ids_val = [s for s in ids_all if split_of[s] == "val"]
    index_of = {s: i for i, s in enumerate(ids_all)}
    z_train = z[[index_of[s] for s in ids_train]]
    z_val = z[[index_of[s] for s in ids_val]]
    history = train(model, ids_train, cache, scalars, z_train, channels, epochs=args.epochs,
                    lr=args.lr, device=device, batch_size=args.batch_size, seed=args.seed,
                    val=(ids_val, z_val) if ids_val else None)

    report = {"variant": args.variant, "rank": args.rank, "seed": args.seed,
              "device": str(device), "deterministic": bool(args.deterministic),
              "draws": int(args.draws), "epochs_run": len(history), "history": history,
              "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
              "rows": {s: int((labels["split"] == s).sum()) for s in ("train", "val", "cal", "test")},
              "splits": {}}
    rules = standard_rules()

    model.eval()
    point_sigma = None
    if model.variant == "point":
        with torch.no_grad():
            x_train, s_train = batch(model, ids_train, cache, scalars, channels, device)
            residual = z_train.to(device) - model(x_train, s_train)
        point_sigma = residual.std(dim=0, unbiased=True).cpu().numpy().tolist()
        report["point_sigma_internal"] = point_sigma

    temperature = None
    calibration = None
    if args.calibrate:
        ids_cal = [s for s in ids_all if split_of[s] == "cal"]
        if ids_cal:
            temperature, calibration = calibrate(model, ids_cal, cache, scalars,
                                                z[[index_of[s] for s in ids_cal]],
                                                channels, device, point_sigma=point_sigma)
            report["calibration"] = calibration
            report["calibration"]["temperature"] = round(float(temperature), 6)

    def evaluate_splits(tag: str, temperature_here=None, covariance_mode="full") -> dict:
        out = {}
        for split in ("train", "val", "cal", "test"):
            rows = [i for i, s in enumerate(ids_all) if split_of[s] == split]
            if len(rows) < 4:
                continue
            split_ids = [ids_all[i] for i in rows]
            split_truth = {k: v[rows] for k, v in truth.items()}
            predicted, _ = predict(model, split_ids, cache, scalars, channels, device,
                                   draws=args.draws, seed=args.seed, point_sigma=point_sigma,
                                   temperature=temperature_here, covariance_mode=covariance_mode)
            metrics = {"n_rows": len(rows),
                       "n_groups": int(len(np.unique(groups[rows]))),
                       "attributes": attribute_metrics(split_truth, predicted),
                       "events": event_metrics(split_truth, predicted, rules, seed=args.seed),
                       "tag": tag}

            def rule_statistic(selection, rule_name="g10_gt_0.05"):
                realised = np.asarray(rules[rule_name].evaluate(split_truth), dtype=float)
                return float(realised[selection].mean())

            metrics["prevalence_ci"] = group_bootstrap(labels.iloc[rows], split_truth, predicted,
                                                       groups[rows], rule_statistic, seed=args.seed)
            out[split] = metrics
        return out

    report["splits"] = evaluate_splits("as_trained", temperature_here=temperature)

    temperatures_u0 = {}
    if args.ablate_u and model.rank:
        # same checkpoint, correlation removed: the only way to attribute a joint gain to the
        # covariance structure rather than to the extra head parameters and their optimisation path
        saved = zero_low_rank(model.head)
        modes = (["shared", "refit"] if args.ablate_u_calibration_mode == "both"
                 else [args.ablate_u_calibration_mode])
        temperatures_u0 = {}
        try:
            for mode in modes:
                temperature_u0 = temperature
                if mode == "refit" and args.calibrate:
                    ids_cal = [s for s in ids_all if split_of[s] == "cal"]
                    temperature_u0, calibration_u0 = calibrate(
                        model, ids_cal, cache, scalars, z[[index_of[s] for s in ids_cal]],
                        channels, device, point_sigma=point_sigma)
                    report["calibration_u0_refit"] = dict(calibration_u0)
                    report["calibration_u0_refit"]["temperature"] = round(float(temperature_u0), 6)
                key = "splits_u0" if mode == "shared" else "splits_u0_refit"
                report[key] = evaluate_splits(f"u_zeroed_{mode}", temperature_here=temperature_u0)
                temperatures_u0[mode] = temperature_u0
        finally:
            restore_low_rank(model.head, saved)
        # U = 0 shrinks the marginal variances as well: report that as a separate matched-marginals
        # control before any claim that "the correlation itself" is harmful.
        report["ablation"] = {
            "kind": "same-checkpoint U=0",
            "calibration_mode": args.ablate_u_calibration_mode,
            "temperature_policy": {
                "shared": "shared_T_fitted_on_full_joint_cal",
                "refit": "refit_T_on_cal_of_the_u_zero_model",
                "both": "reported separately: shared_T and refit_T_on_cal",
            }[args.ablate_u_calibration_mode],
            "temperature_full": temperature,
            "temperature_u0": {k: v for k, v in temperatures_u0.items()},
            "note": "covariance reduced to diag(sigma^2); encoder and mu unchanged. "
                    "shared_T and refit_T_on_cal must never be mixed inside one comparison table."}

    if args.matched_marginals and model.rank:
        report["splits_matched_marginals"] = evaluate_splits(
            "matched_marginals", temperature_here=temperature, covariance_mode="matched_marginals")
        report["matched_marginals"] = {
            "covariance": "diag(diag(Sigma_joint)) - marginal variances untouched",
            "temperature": "same cal-fitted T as the full joint model",
            "note": "isolates the off-diagonal covariance; compare with U=0 only as a second control"}

    if args.save_checkpoint:
        saved_payload = save_checkpoint(
            args.save_checkpoint, model,
            scalar_columns=list(scalars.columns),
            scalar_mean=[float(v) for v in mean[scalars.columns]],
            scalar_std=[float(v) for v in std[scalars.columns]],
            point_sigma=point_sigma, temperature=temperature, seed=args.seed,
            labels_path=args.labels, cache_path=args.cache, channels=channels,
            extra={"ablate_u_calibration_mode": args.ablate_u_calibration_mode})
        report["checkpoint_path"] = str(args.save_checkpoint)
        report["checkpoint_sha256"] = saved_payload["weights_sha256"]

    if args.predictions_out:
        # every split is exported: paired statistics need test rows, and a deployable risk threshold
        # must be chosen from cal probabilities without ever looking at test danger labels
        rows_export = list(range(len(ids_all)))
        ids_export = [ids_all[i] for i in rows_export]
        export_truth = {k: v[rows_export] for k, v in truth.items()}

        def branch_frame(temperature_here, covariance_mode="full", suffix=""):
            predicted, _ = predict(model, ids_export, cache, scalars, channels, device,
                                   draws=args.draws, seed=args.seed, point_sigma=point_sigma,
                                   temperature=temperature_here, covariance_mode=covariance_mode)
            probabilities = {name: np.atleast_1d(np.asarray(prob_event(predicted, rule, axis=0),
                                                            dtype=float))
                             for name, rule in rules.items()}
            frame = per_sample_frame(labels, rows_export, export_truth, predicted, probabilities)
            if suffix:                      # truth stays in the first branch only
                frame = frame.drop(columns=[c for c in frame.columns if c.startswith("truth_")])
                frame = frame.rename(columns={c: f"{c}__{suffix}" for c in frame.columns
                                              if c.startswith("p_") or c.startswith("mean_")})
            return frame

        frame = branch_frame(temperature)
        if args.ablate_u and model.rank:
            saved = zero_low_rank(model.head)
            try:
                for mode, temperature_here in temperatures_u0.items():
                    suffix = "u0" if mode == "shared" else "u0_refit"
                    extra = branch_frame(temperature_here, suffix=suffix)
                    frame = frame.merge(extra, on=["sample_id", "group_id", "split"], how="left")
            finally:
                restore_low_rank(model.head, saved)
        if args.matched_marginals and model.rank:
            extra = branch_frame(temperature, covariance_mode="matched_marginals",
                                 suffix="matched")
            frame = frame.merge(extra, on=["sample_id", "group_id", "split"], how="left")
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.predictions_out, index=False)
        report["predictions_path"] = str(args.predictions_out)

    report["checkpoint"] = {k: v for k, v in export_state(model, seeds=[args.seed]).items()
                            if k != "state_dict"}
    report["notes"] = {
        "cal_role": "not used in training; calibration is a separate step",
        "test_role": "EXPLORATORY - the old 120-tile development_test has been inspected repeatedly",
        "nll_space": "training uses internal-space NLL; physical-space NLL needs the Jacobian term"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("variant", "seed", "epochs_run",
                                             "trainable_parameters", "rows")}, indent=2))
    for split, metrics in report["splits"].items():
        attrs = metrics["attributes"]
        print(f"{split}: n={metrics['n_rows']} groups={metrics['n_groups']} "
              + " ".join(f"{k}_MAE={v['mae']:.4f}" for k, v in attrs.items())
              + f" brier(g10>0.05)={metrics['events']['g10_gt_0.05']['brier']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
