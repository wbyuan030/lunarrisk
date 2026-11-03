"""P2.2: one fair comparison matrix for the probability models on a fixed sample set.

What this script refuses to do, because the P2.1 drafts did it:

* compare models trained on different tiles (the P1.8 CNN baselines skip tiles whose NAC layer is
  less than 99 % finite, so every model here is restricted to the *intersection*);
* mix calibration policies (``shared_T``, ``refit_T_on_cal`` and ``uncalibrated`` are separate
  columns and never one table);
* explain a ``U = 0`` gain as "correlation is harmful" while the ablation also shrinks the marginal
  variances - ``matched_marginals`` keeps latent ``diag(Sigma_z)``; physical g15 margins can change;
* choose a deployment threshold from test danger labels (thresholds come from ``cal`` only).

Every model writes per-sample rows (``split``, truth, posterior mean, per-rule probability). The
aggregate statistics are computed from those rows with a **group** paired bootstrap over the 25
development-test groups, never over tiles or over rules.

``development_test`` is the old 110-tile / 25-group set. It has been inspected many times and is a
development set, not an external OOD result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_posterior_p21 as runner  # noqa: E402

from lunarrisk.rules import prob_event, standard_rules  # noqa: E402

TARGETS = runner.TARGET_COLUMNS
COVERAGES = (0.10, 0.20, 0.40)
MODELS = ("coarse_ridge", "coarse_logistic", "direct_event", "point", "diagonal", "joint")
HEAD_RULES = ("g10_gt_0.05", "g15_gt_0.02")
RIDGE_NOTE = ("Misspecified physical-space independent Gaussian residuals, clipped per attribute; "
              "does not enforce g15 <= g10. Separate from the historical coarse logistic heads.")
UNCERTAINTY = "test groups only, conditional on trained models and fixed Cal thresholds"


def load_cache_meta(cache_path: Path) -> dict:
    with np.load(cache_path, allow_pickle=True) as data:
        if "__meta__" not in data:
            raise ValueError("cache is missing __meta__")
        metadata = json.loads(str(data["__meta__"]))
    if not isinstance(metadata, dict):
        raise ValueError("cache metadata must be an object")
    return metadata


def validate_identity(frame):
    for column in ("sample_id", "group_id", "split"):
        if column not in frame or frame[column].isna().any() or \
                frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"missing/empty {column}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("duplicate sample_id")
    if not frame["split"].isin(("train", "val", "cal", "test")).all():
        raise ValueError("unknown split")


def canonical_labels(labels: pd.DataFrame, cache_meta: dict, min_nac_valid: float,
                     cache_ids=None):
    """Fail closed on missing metadata/assets, then apply the documented NAC filter."""
    validate_identity(labels)
    if not 0 <= min_nac_valid <= 1:
        raise ValueError("min_nac_valid must be in [0, 1]")
    ids = labels["sample_id"].astype(str)
    if cache_ids is not None and not set(ids).issubset(set(cache_ids)):
        raise ValueError("canonical labels contain missing cache assets")
    fractions = []
    for sample in ids:
        if not isinstance(cache_meta.get(sample), dict) or \
                "nac_valid_fraction" not in cache_meta[sample]:
            raise ValueError(f"missing nac_valid_fraction metadata for {sample}")
        value = float(cache_meta[sample]["nac_valid_fraction"])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid nac_valid_fraction for {sample}")
        fractions.append(value)
    fraction = pd.Series(fractions, index=labels.index)
    keep = fraction >= min_nac_valid
    excluded = labels.loc[~keep, ["sample_id", "split", "group_id"]].copy()
    excluded["nac_valid_fraction"] = fraction[~keep].to_numpy()
    used = labels.loc[keep].reset_index(drop=True)
    report = {
        "threshold": min_nac_valid,
        "rows_total": int(len(labels)),
        "rows_used": int(len(used)),
        "excluded": excluded.to_dict(orient="records"),
        "excluded_by_split": excluded.groupby("split").size().to_dict(),
        "rows_by_split": used.groupby("split").size().to_dict(),
        "groups_by_split": used.groupby("split")["group_id"].nunique().to_dict(),
        "sample_id_sha256": text_sha256(sorted(used["sample_id"].astype(str))),
    }
    return used, report


def train_ridge(labels: pd.DataFrame, seed: int):
    """Misspecified four-target physical Gaussian; retain as a separately labeled reference."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    features = labels[runner.load_baseline_module().COARSE_FEATURES].to_numpy(dtype=np.float64)
    train = labels["split"].to_numpy() == "train"
    cal = labels["split"].to_numpy() == "cal"
    medians = np.nanmedian(features[train], axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("coarse features require finite training medians")
    features = np.where(np.isfinite(features), features, medians)
    scaler = StandardScaler().fit(features[train])
    x = scaler.transform(features)
    y = labels[[c for c in TARGETS.values()]].to_numpy(dtype=np.float64)
    model = Ridge(alpha=1.0).fit(x[train], y[train])
    residual = y[train] - model.predict(x[train])
    sigma = residual.std(axis=0, ddof=1)
    sigma = np.maximum(sigma, 1e-9)
    mu_cal = model.predict(x[cal])
    covariance = np.tile(np.diag(sigma ** 2), (int(cal.sum()), 1, 1))
    temperature = runner.fit_temperature(mu_cal, covariance, y[cal])
    return {"model": model, "scaler": scaler, "medians": medians, "sigma": sigma,
            "temperature": float(temperature), "seed": seed}


def ridge_predictions(fitted, labels: pd.DataFrame, draws: int, seed: int):
    features = labels[runner.load_baseline_module().COARSE_FEATURES].to_numpy(dtype=np.float64)
    features = np.where(np.isfinite(features), features, fitted["medians"])
    x = fitted["scaler"].transform(features)
    mu = fitted["model"].predict(x)
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal((draws,) + mu.shape) * fitted["sigma"] * fitted["temperature"]
    values = mu[None, ...] + noise
    values[..., 0:2] = np.clip(values[..., 0:2], 1e-6, 1 - 1e-6)
    values[..., 2:4] = np.maximum(values[..., 2:4], 1e-6)
    draws_by_target = {key: values[..., i] for i, key in enumerate(TARGETS)}
    return mu, draws_by_target


def frame_from_draws(labels, truth, mean, draw_map, rules):
    frame = pd.DataFrame({"sample_id": labels["sample_id"].astype(str).to_numpy(),
                          "group_id": labels["group_id"].astype(str).to_numpy(),
                          "split": labels["split"].astype(str).to_numpy()})
    for key in TARGETS:
        frame[f"truth_{key}"] = np.asarray(truth[key], dtype=float)
        frame[f"mean_{key}"] = np.asarray(mean[key], dtype=float)
    for name, rule in rules.items():
        frame[f"p_{name}"] = np.atleast_1d(np.asarray(prob_event(draw_map, rule, axis=0), dtype=float))
    return frame


def train_direct_event(labels, cache, cache_meta, seed, epochs, batch, lr, device):
    """P1.8 direct-event head: TinyCNN with per-rule BCE logits on two preregistered rules."""
    import torch
    baseline = runner.load_baseline_module()
    table = baseline.build_scalar_table(labels, cache_meta)
    train_ids = labels.loc[labels["split"] == "train", "sample_id"].astype(str).tolist()
    mean = table.loc[train_ids].mean()
    std = table.loc[train_ids].std().replace(0.0, 1.0)
    table = (table - mean) / std

    class Args:
        pass

    args = Args()
    args.epochs, args.batch, args.lr, args.seed = epochs, batch, lr, seed
    args.augment, args.patience, args.verbose = False, 4, False
    result = baseline.train_variant(cache, labels, args, device, (0, 1, 2),
                                    scalar_table=table, scalar_columns=tuple(baseline.ALL_SCALARS))
    fractions, logits = baseline.predict_variant(result, labels, cache, (0, 1, 2), batch, device)
    probabilities = {"g10_gt_0.05": 1 / (1 + np.exp(-logits[:, 0])),
                     "g15_gt_0.02": 1 / (1 + np.exp(-logits[:, 1]))}
    return result, probabilities, fractions, (mean, std)


def train_coarse_logistic(fitted, labels, rules):
    """Original coarse-feature event heads, with the original Cal-only Platt treatment."""
    baseline = runner.load_baseline_module()
    features = labels[baseline.COARSE_FEATURES].to_numpy(dtype=float)
    x = fitted["scaler"].transform(np.where(np.isfinite(features), features, fitted["medians"]))
    train, cal = (labels["split"].to_numpy() == split for split in ("train", "cal"))
    attributes = {key: labels[column].to_numpy(dtype=float) for key, column in TARGETS.items()}
    heads = {}
    for name in HEAD_RULES:
        truth = np.asarray(rules[name].evaluate(attributes), dtype=int)
        model, degenerate = baseline.fit_logistic(x[train], truth[train])
        fallback = float(truth[train].mean())
        scores = baseline.logit(baseline.logistic_probability(model, x, fallback))
        slope, intercept, cal_degenerate = baseline.platt_scale(scores[cal], truth[cal])
        heads[name] = {"coef": None if model is None else model.coef_[0].tolist(),
                       "intercept": None if model is None else float(model.intercept_[0]),
                       "fallback": fallback, "degenerate": bool(degenerate),
                       "platt": {"slope": slope, "intercept": intercept,
                                 "degenerate": bool(cal_degenerate)}}
    return heads


def coarse_probabilities(heads, x):
    from scipy.special import expit
    probabilities = {}
    for name, head in heads.items():
        raw = (np.full(len(x), head["fallback"]) if head["coef"] is None else
               expit(x @ np.asarray(head["coef"]) + head["intercept"]))
        raw = np.clip(raw, 1e-7, 1 - 1e-7)
        score = np.log(raw / (1 - raw))
        platt = head["platt"]
        probabilities[name] = expit(platt["slope"] * score + platt["intercept"])
    return probabilities


def event_frame(labels, probabilities, fractions=None):
    frame = labels[["sample_id", "group_id", "split"]].copy()
    for i, (key, column) in enumerate(TARGETS.items()):
        frame[f"truth_{key}"] = labels[column].to_numpy(dtype=float)
        if fractions is not None and i < 2:
            frame[f"mean_{key}"] = fractions[:, i]
    for name, probability in probabilities.items():
        frame[f"p_{name}"] = probability
    return frame


def coarse_state(fitted):
    return {"feature_columns": list(runner.load_baseline_module().COARSE_FEATURES),
            "medians": fitted["medians"].tolist(),
            "scalar_mean": fitted["scaler"].mean_.tolist(),
            "scalar_std": fitted["scaler"].scale_.tolist()}


def save_baseline_checkpoint(path, model_name, state, labels, *, seed, config, provenance):
    """No pickled sklearn models: payload contains tensors and plain Python values only."""
    import torch
    payload = {"format_version": 1, "model": model_name, "seed": int(seed),
               "config": dict(config), "source_sha256": runner.source_fingerprints(),
               "git_sha": runner.git_sha(), "provenance": dict(provenance),
               "sample_ids": labels["sample_id"].astype(str).tolist(),
               "ids_by_split": {split: labels.loc[labels["split"] == split, "sample_id"]
                                .astype(str).tolist() for split in ("train", "val", "cal", "test")},
               "canonical": labels[["sample_id", "group_id", "split", *TARGETS.values()]]
                            .to_dict(orient="records"),
               "rules": {name: rule.to_dict() for name, rule in standard_rules().items()},
               **state}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def predict_baseline_checkpoint(path, labels, cache=None, cache_meta=None, device="cpu",
                                draws=None, seed=None, rules=None):
    """Replay the saved full sample order; subset-dependent RNG streams are never implicit."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    canonical = pd.DataFrame(payload["canonical"])
    validate_identity(labels)
    reference = event_frame(canonical, {})
    candidate = event_frame(labels, {})
    aligned_frames(reference, candidate)
    labels = labels.set_index("sample_id").loc[payload["sample_ids"]].reset_index()
    draws = payload["config"]["draws"] if draws is None else draws
    seed = payload["seed"] if seed is None else seed
    if rules is not None and payload["model"] != "coarse_ridge":
        raise ValueError("dedicated event heads cannot transfer to new rules")
    if payload["model"] == "direct_event":
        baseline = runner.load_baseline_module()
        columns = payload["scalar_columns"]
        table = baseline.build_scalar_table(labels, cache_meta)
        table[columns] = ((table[columns] - np.asarray(payload["scalar_mean"])) /
                          np.asarray(payload["scalar_std"]))
        model = baseline.TinyCNN(in_channels=len(payload["channels"]), n_scalars=len(columns))
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        result = {"model": model, "scalar_table": table, "scalar_columns": tuple(columns)}
        fractions, logits = baseline.predict_variant(result, labels, cache,
                          tuple(payload["channels"]), payload["config"]["batch_size"], device)
        probabilities = {name: 1 / (1 + np.exp(-logits[:, i]))
                         for i, name in enumerate(HEAD_RULES)}
        return event_frame(labels, probabilities, fractions), payload
    features = labels[payload["feature_columns"]].to_numpy(dtype=float)
    features = np.where(np.isfinite(features), features, np.asarray(payload["medians"]))
    x = (features - np.asarray(payload["scalar_mean"])) / np.asarray(payload["scalar_std"])
    if payload["model"] == "coarse_logistic":
        return event_frame(labels, coarse_probabilities(payload["heads"], x)), payload
    if payload["model"] != "coarse_ridge":
        raise ValueError("unknown baseline checkpoint model")
    mu = x @ np.asarray(payload["coef"]).T + np.asarray(payload["intercept"])
    values = mu[None, ...] + np.random.default_rng(seed).standard_normal((draws,) + mu.shape) * \
        np.asarray(payload["sigma"]) * payload["temperature"]
    values[..., :2] = np.clip(values[..., :2], 1e-6, 1 - 1e-6)
    values[..., 2:] = np.maximum(values[..., 2:], 1e-6)
    truth = {key: labels[column].to_numpy(dtype=float) for key, column in TARGETS.items()}
    frame = frame_from_draws(labels, truth, {key: mu[:, i] for i, key in enumerate(TARGETS)},
                            {key: values[..., i] for i, key in enumerate(TARGETS)},
                            standard_rules() if rules is None else rules)
    return frame, payload


def checked_probabilities(probability):
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return p


def brier_and_nll(probability, truth):
    raw = checked_probabilities(probability)
    p = np.clip(raw, 1e-6, 1 - 1e-6)
    y = np.asarray(truth, dtype=float)
    return float(np.mean((raw - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def aligned_frames(frame_a, frame_b):
    """Require one matching canonical record per ID, including truths and cluster assignment."""
    for frame in (frame_a, frame_b):
        validate_identity(frame)
    a = frame_a.set_index("sample_id").sort_index()
    b = frame_b.set_index("sample_id").sort_index()
    if list(a.index) != list(b.index):
        raise ValueError("prediction frames cover different sample ids")
    for column in ("group_id", "split"):
        if not a[column].astype(str).equals(b[column].astype(str)):
            raise ValueError(f"prediction frames disagree on {column}")
    for key in TARGETS:
        column = f"truth_{key}"
        if column not in a or column not in b:
            raise ValueError(f"prediction frames missing {column}")
        av, bv = (frame[column].to_numpy(dtype=float) for frame in (a, b))
        # CSV round trips can change the last floating-point bit, but never the label meaning.
        if not np.isfinite(av).all() or not np.isfinite(bv).all() or \
                not np.allclose(av, bv, rtol=1e-12, atol=1e-12):
            raise ValueError(f"prediction frames disagree on {column}")
    return a, b


def validate_prediction_frame(frame, labels, model, rules):
    aligned_frames(event_frame(labels, {}), frame)
    names = HEAD_RULES if model in ("direct_event", "coarse_logistic") else tuple(rules)
    expected = {f"p_{name}" for name in names}
    if model == "joint":
        expected |= {f"p_{name}__{suffix}" for name in names
                     for suffix in ("u0", "u0_refit", "matched")}
    actual = {column for column in frame if column.startswith("p_")}
    if actual != expected:
        raise ValueError(f"{model}: missing/stale probability columns: {actual ^ expected}")
    for column in actual:
        checked_probabilities(frame[column])


def check_model_files(outdir, seeds, complete=False):
    expected = {f"{model}_seed{seed}" for model in MODELS for seed in seeds}
    for directory, extension in (("pred", "csv"), ("ckpt", "pt"), ("report", "json")):
        found = {path.stem for path in (Path(outdir) / directory).glob(f"*.{extension}")}
        if found - expected or (complete and expected - found):
            raise ValueError(f"{directory}: missing/stale model files: "
                             f"{sorted((found - expected) | (expected - found if complete else set()))}")


def validate_saved_run(record, identity, outdir, seeds):
    if record.get("identity") != identity or record.get("status") != "complete":
        raise ValueError("saved run has stale source/data/configuration or is incomplete")
    check_model_files(outdir, seeds, complete=True)
    expected = {f"{directory}/{model}_seed{seed}.{extension}"
                for directory, extension in (("pred", "csv"), ("ckpt", "pt"), ("report", "json"))
                for model in MODELS for seed in seeds}
    if set(record.get("files", {})) != expected:
        raise ValueError("saved run is missing model file fingerprints")
    for relative, digest in record["files"].items():
        if runner.file_sha256(Path(outdir) / relative) != digest:
            raise ValueError(f"saved model file changed: {relative}")


def text_sha256(lines):
    import hashlib
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def paired_group_bootstrap(delta, groups, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    estimates = np.empty(draws)
    for i in range(draws):
        pick = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[g] for g in pick])
        estimates[i] = np.mean(delta[rows])
    return {"mean": float(np.mean(delta)), "p025": float(np.percentile(estimates, 2.5)),
            "p975": float(np.percentile(estimates, 97.5)), "groups": int(len(unique)),
            "bootstrap_draws": int(draws), "uncertainty": UNCERTAINTY}


def coverage_metrics(frame, rule_name, rule, coverage, threshold=None):
    """Accept the lowest-risk ``coverage`` fraction; threshold from cal when supplied.

    ``oracle`` ranks by the model's own probabilities; ``deployable`` uses a threshold fitted on
    ``cal`` and never looks at test danger labels.
    """
    truth = np.asarray(rule.evaluate({k: frame[f"truth_{k}"].to_numpy() for k in TARGETS}), dtype=bool)
    probability = checked_probabilities(frame[f"p_{rule_name}"])
    if not 0 <= coverage <= 1:
        raise ValueError("coverage must be in [0, 1]")
    if threshold is None:
        # Deterministic, label-free ties. floor(n*c) is explicit, including k=0.
        ids = frame["sample_id"].astype(str).to_numpy()
        if len(np.unique(ids)) != len(ids):
            raise ValueError("oracle ranking requires unique sample_id")
        order = np.lexsort((ids, probability))
        accepted = np.zeros(len(frame), dtype=bool)
        accepted[order[:int(np.floor(len(frame) * coverage))]] = True
    else:
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        accepted = probability <= threshold
    return risk_counts(truth, accepted)


def risk_counts(truth, accepted):
    unsafe = np.asarray(truth, dtype=bool)
    accepted = np.asarray(accepted, dtype=bool)
    count = int(accepted.sum())
    total = int(unsafe.sum())
    unsafe_count = int((unsafe & accepted).sum())
    return {"rows": int(len(unsafe)), "unsafe_total": total, "accepted": count,
            "unsafe_accepted": unsafe_count,
            "uar": float(unsafe_count / count) if count else None,
            "hmr": float(unsafe_count / total) if total else None,
            "undefined": {"uar": "no accepted rows" if not count else None,
                          "hmr": "no unsafe rows" if not total else None},
            "coverage": float(accepted.mean()) if len(accepted) else None}


def paired_deployable_bootstrap(frame_a, frame_b, rule_name, rule, threshold_a, threshold_b,
                                draws=2000, seed=0):
    """Paired test-group uncertainty; never refit Cal cutoffs in bootstrap replicates."""
    a, b = aligned_frames(frame_a, frame_b)
    if not a["split"].eq("test").all() or not np.isfinite([threshold_a, threshold_b]).all():
        raise ValueError("deployable bootstrap requires test rows and finite fixed thresholds")
    truth = np.asarray(rule.evaluate({key: a[f"truth_{key}"].to_numpy() for key in TARGETS}),
                       dtype=bool)
    accept_a = checked_probabilities(a[f"p_{rule_name}"]) <= threshold_a
    accept_b = checked_probabilities(b[f"p_{rule_name}"]) <= threshold_b
    groups = a["group_id"].astype(str).to_numpy()
    unique = np.unique(groups)
    if not len(unique) or draws < 1:
        raise ValueError("bootstrap requires test groups and positive draws")
    # Resample sufficient counts per group; the two models always share the same picked groups.
    group_counts = []
    for group in unique:
        selected = groups == group
        group_counts.append([int(truth[selected].sum()), int(accept_a[selected].sum()),
            int((truth & accept_a)[selected].sum()), int(accept_b[selected].sum()),
            int((truth & accept_b)[selected].sum())])
    rng = np.random.default_rng(seed)
    picked = rng.integers(len(unique), size=(draws, len(unique)))
    totals = np.asarray(group_counts)[picked].sum(axis=1).astype(float)
    observed = [risk_counts(truth, accepted) for accepted in (accept_a, accept_b)]
    valid_uar = (totals[:, 1] > 0) & (totals[:, 3] > 0)
    valid_hmr = totals[:, 0] > 0
    estimates = {
        "uar": (totals[valid_uar, 2] / totals[valid_uar, 1] -
                totals[valid_uar, 4] / totals[valid_uar, 3]),
        "hmr": (totals[valid_hmr, 2] - totals[valid_hmr, 4]) / totals[valid_hmr, 0]}
    result = {"groups": int(len(unique)), "bootstrap_draws": int(draws),
              "uncertainty": UNCERTAINTY, "model_a": observed[0], "model_b": observed[1],
              "cal_threshold_a": float(threshold_a), "cal_threshold_b": float(threshold_b),
              "interval_conditioning": "replicates with both ratio denominators nonzero"}
    for metric, values in estimates.items():
        left, right = observed[0][metric], observed[1][metric]
        result[metric] = {"mean": None if left is None or right is None else left - right,
                          "p025": float(np.percentile(values, 2.5)) if len(values) else None,
                          "p975": float(np.percentile(values, 97.5)) if len(values) else None,
                          "valid_draws": len(values), "undefined_draws": draws - len(values)}
    return result


def calibrate_threshold(cal_frame, rule_name, coverage):
    """Deployable threshold: chosen on cal to accept roughly ``coverage`` of cal rows."""
    probability = checked_probabilities(cal_frame[f"p_{rule_name}"])
    if not len(probability) or not 0 < coverage <= 1:
        raise ValueError("Cal threshold requires nonempty Cal rows and coverage in (0, 1]")
    return float(np.quantile(probability, coverage, method="lower"))


def evaluate_frame(frame, rules):
    """Brier/NLL per rule on the test rows of one prediction frame."""
    test = frame[frame["split"] == "test"]
    out = {}
    for name, rule in rules.items():
        column = f"p_{name}"
        if column not in test:
            continue
        truth = np.asarray(rule.evaluate({k: test[f"truth_{k}"].to_numpy() for k in TARGETS}))
        brier, nll = brier_and_nll(test[column], truth)
        out[name] = {"brier": brier, "nll": nll, "prevalence": float(truth.mean()),
                     "rows": int(len(test))}
    return out


def analytic_g10_probability(checkpoint_path, ids, cache, scalars, threshold=0.05, device="cpu"):
    """Closed-form P(g10 > threshold) for the primary rule, used to check the MC estimator.

    A Gaussian posterior on the internal chain gives ``z1 ~ N(mu1, var1)`` and
    ``P(sigmoid(z1) > t) = P(z1 > logit(t))``, so no Monte-Carlo error enters this number. If the
    MC estimate at 4096 draws agrees with it, the draw-based AST integration is trustworthy for the
    single-attribute rules; it is not a substitute for the AST on composite rules.
    """
    import math
    import torch
    from scipy.special import erf
    model, payload = runner.load_checkpoint(checkpoint_path)
    model.to(device).eval()
    table = runner.standardize_with(payload, scalars)
    x, s = runner.batch(model, ids, cache, table, tuple(payload["channels"]), device)
    with torch.no_grad():
        params = model.parameters_dict(model(x, s))
        mu1 = params["mu"][:, 0].cpu().numpy().astype(np.float64)
        temperature = payload["temperature"] if payload["temperature"] is not None else 1.0
        if model.variant == "point":
            sigma = float(payload["point_sigma"][0]) * temperature
            variance = np.full_like(mu1, sigma ** 2)
        else:
            variance = model.covariance(params)[:, 0, 0].cpu().numpy().astype(np.float64) * temperature ** 2
    z_threshold = math.log(threshold / (1 - threshold))
    return 0.5 * (1 - erf((z_threshold - mu1) / np.sqrt(2 * variance)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "outputs" / "labels_hydrated.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs" / "cache_v3.npz")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "p22")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--min-nac-valid", type=float, default=0.99)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true",
                        help="re-aggregate existing prediction CSVs without retraining")
    args = parser.parse_args(argv)

    import torch
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be distinct nonnegative integers")
    if min(args.epochs, args.batch_size, args.draws, args.bootstrap_draws) < 1 or \
            not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("epochs/batch/draws/bootstrap draws/lr must be positive")
    config = {key: getattr(args, key) for key in
              ("epochs", "batch_size", "lr", "draws", "bootstrap_draws", "min_nac_valid", "device")}
    config["seeds"] = seeds
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "report").mkdir(exist_ok=True)
    (args.outdir / "pred").mkdir(exist_ok=True)
    (args.outdir / "ckpt").mkdir(exist_ok=True)

    check_model_files(args.outdir, seeds, complete=args.skip_train)
    labels_all = pd.read_csv(args.labels, dtype={"sample_id": str, "group_id": str, "split": str})
    cache_meta = load_cache_meta(args.cache)
    with np.load(args.cache, allow_pickle=True) as archive:
        cache_ids = set(archive.files) - {"__meta__"}
    labels, inventory = canonical_labels(labels_all, cache_meta, args.min_nac_valid, cache_ids)
    aligned_frames(event_frame(labels, {}), event_frame(labels, {}))
    if any(int((labels["split"] == split).sum()) < (2 if split == "train" else 1)
           for split in ("train", "val", "cal", "test")):
        raise ValueError("canonical data require all four nonempty splits and >=2 training rows")
    labels_path = args.outdir / "labels_p22.csv"
    labels.to_csv(labels_path, index=False)
    inventory["labels_path"] = str(labels_path)
    inventory["labels_file_sha256"] = runner.file_sha256(labels_path)
    rules = standard_rules()

    manifest = {"inventory": inventory, "rules": {k: v.to_dict() for k, v in rules.items()},
                "models": {}, "device": None, "code_sha": runner.git_sha(),
                "source_sha256": runner.source_fingerprints(), "config": config,
                "bootstrap_draws": args.bootstrap_draws,
                "runtime": {"python": sys.version, "torch": str(torch.__version__),
                            "hip": torch.version.hip, "numpy": np.__version__,
                            "pandas": pd.__version__},
                "uncertainty": UNCERTAINTY,
                "ci_scope": "pointwise 95% intervals; no across-seed or multiplicity-adjusted inference",
                "coarse_ridge_note": RIDGE_NOTE,
                "ast_scan": "deferred; five existing primary rules only, no Test-driven expansion",
                "matched_marginals": "latent z margins only; physical g15 marginal can change",
                "labels_source_sha256": runner.file_sha256(args.labels),
                "cache_sha256": runner.file_sha256(args.cache)}
    provenance = {key: manifest[key] for key in ("labels_source_sha256", "cache_sha256")}
    provenance["labels_sha256"] = inventory["labels_file_sha256"]
    run_identity = {"config": config, "provenance": provenance,
                    "source_sha256": manifest["source_sha256"], "rules": manifest["rules"]}
    run_path = args.outdir / "matrix_run.json"
    if args.skip_train:
        if not run_path.exists():
            raise ValueError("missing matrix_run.json; cannot authenticate existing predictions")
        validate_saved_run(json.loads(run_path.read_text()), run_identity, args.outdir, seeds)
    else:
        run_path.write_text(json.dumps({"status": "incomplete", "identity": run_identity}, indent=2))

    if not args.skip_train:
        device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                              else "cpu")
        # Enable before GPU work; fresh-process repetitions check this configuration.
        # The mechanism behind earlier same-seed differences has not been isolated.
        torch.use_deterministic_algorithms(True)
        manifest["device"] = str(device)
        manifest["deterministic"] = True

        # --- baselines -------------------------------------------------------
        from sklearn.linear_model import Ridge  # noqa: F401  (fail early, not mid-run)
        _, cache, _ = runner.load_dataset(args.labels, args.cache, runner.load_baseline_module())
        for seed in seeds:
            fitted = train_ridge(labels, seed)
            mean, draws = ridge_predictions(fitted, labels, args.draws, seed)
            truth = {k: labels[c].to_numpy(dtype=float) for k, c in TARGETS.items()}
            frame = frame_from_draws(labels, truth, {k: mean[:, i] for i, k in enumerate(TARGETS)},
                                     draws, rules)
            frame.to_csv(args.outdir / "pred" / f"coarse_ridge_seed{seed}.csv", index=False)
            (args.outdir / "report" / f"coarse_ridge_seed{seed}.json").write_text(json.dumps({
                "model": "coarse_ridge", "seed": seed, "temperature": fitted["temperature"],
                "sigma_physical": fitted["sigma"].tolist(),
                "ridge_coef": fitted["model"].coef_.tolist(),
                "intercept": fitted["model"].intercept_.tolist(),
                "features": runner.load_baseline_module().COARSE_FEATURES,
                "config": config, "source_sha256": manifest["source_sha256"],
                "note": RIDGE_NOTE}, indent=2), encoding="utf-8")
            saved_coarse = coarse_state(fitted)
            save_baseline_checkpoint(args.outdir / "ckpt" / f"coarse_ridge_seed{seed}.pt",
                "coarse_ridge", {**saved_coarse, "coef": fitted["model"].coef_.tolist(),
                "intercept": fitted["model"].intercept_.tolist(),
                "sigma": fitted["sigma"].tolist(), "temperature": fitted["temperature"],
                "calibration_policy": "Cal-only residual Gaussian temperature", "note": RIDGE_NOTE},
                labels, seed=seed, config=config, provenance=provenance)

            heads = train_coarse_logistic(fitted, labels, rules)
            features = labels[saved_coarse["feature_columns"]].to_numpy(dtype=float)
            x = fitted["scaler"].transform(np.where(np.isfinite(features), features, fitted["medians"]))
            event_frame(labels, coarse_probabilities(heads, x)).to_csv(
                args.outdir / "pred" / f"coarse_logistic_seed{seed}.csv", index=False)
            save_baseline_checkpoint(args.outdir / "ckpt" / f"coarse_logistic_seed{seed}.pt",
                "coarse_logistic", {**saved_coarse, "heads": heads,
                "calibration_policy": "original coarse logistic heads + Cal-only Platt"},
                labels, seed=seed, config=config, provenance=provenance)
            (args.outdir / "report" / f"coarse_logistic_seed{seed}.json").write_text(json.dumps({
                "model": "coarse_logistic", "seed": seed, "heads": heads, "config": config,
                "source_sha256": manifest["source_sha256"], "rules": list(HEAD_RULES)}, indent=2))

            result, probabilities, fractions, scalar_stats = train_direct_event(
                labels, cache, cache_meta, seed, args.epochs, args.batch_size, args.lr, device)
            base = event_frame(labels, probabilities, fractions)
            base.to_csv(args.outdir / "pred" / f"direct_event_seed{seed}.csv", index=False)
            scalar_mean, scalar_std = scalar_stats
            columns = list(result["scalar_columns"])
            save_baseline_checkpoint(args.outdir / "ckpt" / f"direct_event_seed{seed}.pt",
                "direct_event", {"state_dict": {key: value.detach().cpu().clone()
                    for key, value in result["model"].state_dict().items()},
                "channels": [0, 1, 2], "scalar_columns": columns,
                "scalar_mean": scalar_mean[columns].tolist(),
                "scalar_std": scalar_std[columns].tolist(),
                "calibration_policy": "raw dedicated event-head sigmoid; no calibration"},
                labels, seed=seed, config=config, provenance=provenance)
            (args.outdir / "report" / f"direct_event_seed{seed}.json").write_text(json.dumps({
                "model": "direct_event", "seed": seed, "parameters": result["parameters"],
                "best_epoch": result["best_epoch"], "rules": list(probabilities),
                "config": config, "source_sha256": manifest["source_sha256"],
                "note": "only the two preregistered event heads; no probabilities for other rules"}, indent=2),
                encoding="utf-8")

        # --- posterior variants ---------------------------------------------
        for variant in ("point", "diagonal", "joint"):
            for seed in seeds:
                stem = f"{variant}_seed{seed}"
                argv_run = ["--labels", str(labels_path), "--cache", str(args.cache),
                            "--variant", variant, "--seed", str(seed),
                            "--epochs", str(args.epochs), "--draws", str(args.draws),
                            "--batch-size", str(args.batch_size), "--lr", str(args.lr),
                            "--device", str(device), "--calibrate", "--deterministic",
                            "--save-checkpoint", str(args.outdir / "ckpt" / f"{stem}.pt"),
                            "--predictions-out", str(args.outdir / "pred" / f"{stem}.csv"),
                            "--out", str(args.outdir / "report" / f"{stem}.json")]
                if variant == "joint":
                    argv_run += ["--ablate-u", "--ablate-u-calibration-mode", "both",
                                 "--matched-marginals"]
                print(f"[compare_p22] training {stem}", flush=True)
                runner.main(argv_run)

        check_model_files(args.outdir, seeds, complete=True)
        files = {str(path.relative_to(args.outdir)): runner.file_sha256(path)
                 for directory, extension in (("pred", "csv"), ("ckpt", "pt"), ("report", "json"))
                 for path in sorted((args.outdir / directory).glob(f"*.{extension}"))}
        run_path.write_text(json.dumps({"status": "complete", "identity": run_identity,
                                        "files": files}, indent=2))

    # --- aggregation ---------------------------------------------------------
    frames = {}
    for path in sorted((args.outdir / "pred").glob("*.csv")):
        frame = pd.read_csv(path, dtype={"sample_id": str, "group_id": str, "split": str})
        validate_prediction_frame(frame, labels, path.stem.rsplit("_seed", 1)[0], rules)
        frames[path.stem] = frame
    baseline_names = sorted({name.rsplit("_seed", 1)[0] for name in frames})
    manifest["models"] = baseline_names
    manifest["prediction_sha256"] = {name: runner.file_sha256(
        args.outdir / "pred" / f"{name}.csv") for name in sorted(frames)}

    # sample ids must be identical across models
    id_sets = {name: set(frame["sample_id"]) for name, frame in frames.items()}
    union = set().union(*id_sets.values())
    intersection = set.intersection(*id_sets.values())
    manifest["sample_sets"] = {"union": len(union), "intersection": len(intersection),
                               "per_model": {k: len(v) for k, v in sorted(id_sets.items())}}

    summary = {"models": {}, "pairs": {}, "coverage": {}, "deployable_pairs": {},
               "monte_carlo": {}, "baseline_checkpoint_replay": {}}
    replay_device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                                 else "cpu")
    if args.skip_train:
        _, cache, _ = runner.load_dataset(labels_path, args.cache, runner.load_baseline_module())
    for seed in seeds:
        for model in ("coarse_ridge", "coarse_logistic", "direct_event"):
            name = f"{model}_seed{seed}"
            replay, _ = predict_baseline_checkpoint(args.outdir / "ckpt" / f"{name}.pt",
                                labels, cache, cache_meta, replay_device)
            validate_prediction_frame(replay, labels, model, rules)
            reference, actual = aligned_frames(frames[name], replay)
            columns = [column for column in reference
                       if column.startswith("p_") or column.startswith("mean_")]
            differences = {column: float(np.max(np.abs(reference[column].to_numpy(dtype=float) -
                              actual[column].to_numpy(dtype=float)))) for column in columns}
            maximum = max(differences.values(), default=0.0)
            summary["baseline_checkpoint_replay"][name] = {
                "max_abs_diff": maximum, "per_column": differences, "atol": 1e-7,
                "scope": "reloaded weights/scaler/calibration versus exported full-order CSV"}
            if maximum > 1e-7:
                raise ValueError(f"{name} checkpoint replay differs from export by {maximum}")
    for name, frame in sorted(frames.items()):
        summary["models"][name] = evaluate_frame(frame, rules)

    primary = "g10_gt_0.05"

    def paired_brier(name_a, name_b, rule_key, split="test"):
        """Per-row Brier and NLL differences; negative means model A is better."""
        a = frames[name_a][frames[name_a]["split"] == split]
        b = frames[name_b][frames[name_b]["split"] == split]
        column = f"p_{rule_key}"
        if column not in a or column not in b:
            return None
        a, b = aligned_frames(a, b)
        truth = np.asarray(rules[rule_key].evaluate(
            {k: a[f"truth_{k}"].to_numpy() for k in TARGETS}), dtype=float)
        pa = a[column].to_numpy(dtype=float)
        pb = b[column].to_numpy(dtype=float)
        groups = a["group_id"].astype(str).to_numpy()
        return bootstrap_two(pa, pb, truth, groups, args.bootstrap_draws)

    def bootstrap_two(pa, pb, truth, groups, draws):
        pa, pb = checked_probabilities(pa), checked_probabilities(pb)
        brier = (pa - truth) ** 2 - (pb - truth) ** 2
        pa, pb = np.clip(pa, 1e-6, 1 - 1e-6), np.clip(pb, 1e-6, 1 - 1e-6)
        nll = -(truth * np.log(pa) + (1 - truth) * np.log(1 - pa)) + \
              (truth * np.log(pb) + (1 - truth) * np.log(1 - pb))
        return {"brier": paired_group_bootstrap(brier, groups, draws=draws),
                "nll": paired_group_bootstrap(nll, groups, draws=draws),
                "rows": int(len(truth)), "prevalence": float(np.mean(truth))}

    comparisons = {"joint_vs_point": ("joint", "point"),
                   "joint_vs_diagonal": ("joint", "diagonal"),
                   "diagonal_vs_point": ("diagonal", "point")}
    comparisons.update({f"{model}_vs_{baseline.removeprefix('coarse_')}": (model, baseline)
                        for model in ("point", "diagonal", "joint")
                        for baseline in ("coarse_ridge", "coarse_logistic", "direct_event")})
    pairs = {}
    for seed in seeds:
        pairs[f"seed{seed}"] = {}
        for label, (name_a, name_b) in comparisons.items():
            entry = {}
            for rule_name, _ in rules.items():
                value = paired_brier(f"{name_a}_seed{seed}", f"{name_b}_seed{seed}", rule_name)
                if value is None:
                    continue
                entry[rule_name] = value
            pairs[f"seed{seed}"][label] = entry
        # same-checkpoint ablations, kept apart from the cross-model table. They are reported per
        # rule: matched_marginals leaves P(g10 > t) exactly unchanged by construction (the first
        # Cholesky row only uses noise[0]), so a single-attribute g10 rule cannot attribute anything
        # and the g15 / composite rules are the informative ones.
        joint = frames[f"joint_seed{seed}"]
        ablations = {}
        test_joint = joint[joint["split"] == "test"].set_index("sample_id").sort_index()
        for rule_name, rule in rules.items():
            truth = np.asarray(rule.evaluate(
                {k: test_joint[f"truth_{k}"].to_numpy() for k in TARGETS}), dtype=float)
            pb = test_joint[f"p_{rule_name}"].to_numpy(dtype=float)
            ablations[rule_name] = {}
            for suffix, title in (("u0", "joint_u0_shared_T"), ("u0_refit", "joint_u0_refit_T"),
                                  ("matched", "joint_matched_marginals")):
                column = f"p_{rule_name}__{suffix}"
                if column not in test_joint:
                    continue
                pa = test_joint[column].to_numpy(dtype=float)
                ablations[rule_name][title] = bootstrap_two(
                    pa, pb, truth, test_joint["group_id"].astype(str).to_numpy(),
                    args.bootstrap_draws)
        pairs[f"seed{seed}"]["joint_ablations"] = ablations
    summary["pairs"] = pairs

    # coverage: oracle ranking + deployable cal threshold, for the primary rules
    for seed in seeds:
        summary["coverage"][f"seed{seed}"] = {}
        for model in MODELS:
            name = f"{model}_seed{seed}"
            if name not in frames:
                continue
            frame = frames[name]
            test = frame[frame["split"] == "test"]
            cal = frame[frame["split"] == "cal"]
            entry = {}
            for rule_name, rule in rules.items():
                if f"p_{rule_name}" not in frame:
                    continue
                entry[rule_name] = {}
                for coverage in COVERAGES:
                    entry[rule_name][f"oracle@{int(coverage * 100)}"] = coverage_metrics(
                        test, rule_name, rule, coverage)
                    threshold = calibrate_threshold(cal, rule_name, coverage)
                    entry[rule_name][f"deployable@{int(coverage * 100)}"] = coverage_metrics(
                        test, rule_name, rule, coverage, threshold=threshold)
                    entry[rule_name][f"deployable@{int(coverage * 100)}"]["cal_threshold"] = threshold
            summary["coverage"][f"seed{seed}"][model] = entry

    for seed in seeds:
        summary["deployable_pairs"][f"seed{seed}"] = {}
        for label, (model_a, model_b) in comparisons.items():
            frame_a, frame_b = (frames[f"{model}_seed{seed}"] for model in (model_a, model_b))
            cal_a, cal_b = (frame[frame["split"] == "cal"] for frame in (frame_a, frame_b))
            test_a, test_b = (frame[frame["split"] == "test"] for frame in (frame_a, frame_b))
            entry = {}
            for rule_name, rule in rules.items():
                if f"p_{rule_name}" not in frame_a or f"p_{rule_name}" not in frame_b:
                    continue
                entry[rule_name] = {}
                for coverage in COVERAGES:
                    entry[rule_name][f"deployable@{int(coverage * 100)}"] = paired_deployable_bootstrap(
                        test_a, test_b, rule_name, rule,
                        calibrate_threshold(cal_a, rule_name, coverage),
                        calibrate_threshold(cal_b, rule_name, coverage),
                        draws=args.bootstrap_draws)
            summary["deployable_pairs"][f"seed{seed}"][label] = entry

    # Monte-Carlo error of the event probabilities: same checkpoint, same seed, 256 vs 1024 draws.
    # This is *plain* Monte Carlo; a Sobol/QMC estimator is not claimed here.
    checkpoint = args.outdir / "ckpt" / "joint_seed42.pt"
    if checkpoint.exists():
        import torch
        device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                              else "cpu")
        _, cache, scalars = runner.load_dataset(labels_path, args.cache,
                                                runner.load_baseline_module())
        test = labels[labels["split"] == "test"]
        ids = test["sample_id"].astype(str).tolist()
        probabilities = {}
        for draws in (256, 1024, 4096):
            _, probability, _ = runner.predict_from_checkpoint(
                checkpoint, ids, cache, scalars, rules, device, draws=draws, seed=42)
            probabilities[draws] = probability
        summary["monte_carlo"] = {
            name: {"max_abs_diff_1024_vs_256": float(np.max(np.abs(probabilities[1024][name]
                                                                   - probabilities[256][name]))),
                   "max_abs_diff_4096_vs_256": float(np.max(np.abs(probabilities[4096][name]
                                                                   - probabilities[256][name])))}
            for name in rules}
        analytic = {}
        for variant in ("point", "diagonal", "joint"):
            path = args.outdir / "ckpt" / f"{variant}_seed42.pt"
            if not path.exists():
                continue
            try:
                _, probability, _ = runner.predict_from_checkpoint(
                    path, ids, cache, scalars, rules, device, draws=4096, seed=42)
                analytic[variant] = float(np.max(np.abs(
                    analytic_g10_probability(path, ids, cache, scalars, 0.05, device)
                    - probability["g10_gt_0.05"])))
            except (ValueError, IndexError) as error:  # pragma: no cover - diagnostic only
                analytic[variant] = f"unavailable: {error}"
        summary["monte_carlo"]["analytic_g10_check_max_abs_vs_4096_draws"] = analytic
        summary["monte_carlo"]["note"] = ("plain Monte Carlo with a shared seed; not QMC. "
                                          "analytic_g10 is the closed-form Gaussian CDF for the "
                                          "single-attribute rule")

    (args.outdir / "compare.json").write_text(json.dumps({"manifest": manifest, "summary": summary},
                                                         indent=2), encoding="utf-8")
    print(json.dumps({"models": baseline_names, "sample_sets": manifest["sample_sets"]}, indent=2))
    for name, metrics in sorted(summary["models"].items()):
        if primary in metrics:
            print(f"{name:24s} brier={metrics[primary]['brier']:.4f} nll={metrics[primary]['nll']:.4f} "
                  f"prevalence={metrics[primary]['prevalence']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
