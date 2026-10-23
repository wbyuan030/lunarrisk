"""P2.2: one fair comparison matrix for the probability models on a fixed sample set.

What this script refuses to do, because the P2.1 drafts did it:

* compare models trained on different tiles (the P1.8 CNN baselines skip tiles whose NAC layer is
  less than 99 % finite, so every model here is restricted to the *intersection*);
* mix calibration policies (``shared_T``, ``refit_T_on_cal`` and ``uncalibrated`` are separate
  columns and never one table);
* explain a ``U = 0`` gain as "correlation is harmful" while the ablation also shrinks the marginal
  variances - ``matched_marginals`` keeps ``diag(Sigma_joint)`` and removes only the covariance;
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


def load_cache_meta(cache_path: Path) -> dict:
    data = np.load(cache_path, allow_pickle=True)
    return json.loads(str(data["__meta__"]))


def canonical_labels(labels: pd.DataFrame, cache_meta: dict, min_nac_valid: float):
    """Restrict to the tiles the P1.8 baselines can actually read, and document the loss."""
    fraction = labels["sample_id"].astype(str).map(
        lambda s: float(cache_meta.get(s, {}).get("nac_valid_fraction", 1.0)))
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
    """Four-target ridge on the 60 m summary features + a cal-fitted Gaussian.

    The P1.8 ``coarse_ridge`` only regressed the two exceedance fractions; extending it to all four
    attributes makes it a *stronger* reference for the composite rules, not a weaker one.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    features = labels[runner.load_baseline_module().COARSE_FEATURES].to_numpy(dtype=np.float64)
    train = labels["split"].to_numpy() == "train"
    cal = labels["split"].to_numpy() == "cal"
    medians = np.nanmedian(features[train], axis=0)
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


def brier_and_nll(probability, truth):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(truth, dtype=float)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def aligned_frames(frame_a, frame_b):
    """Both frames restricted to the same sample ids and sorted by ``sample_id``."""
    a = frame_a.set_index("sample_id").sort_index()
    b = frame_b.set_index("sample_id").sort_index()
    if list(a.index) != list(b.index):
        raise ValueError("prediction frames cover different sample ids")
    return a, b


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
            "bootstrap_draws": int(draws)}


def coverage_metrics(frame, rule_name, rule, coverage, threshold=None):
    """Accept the lowest-risk ``coverage`` fraction; threshold from cal when supplied.

    ``oracle`` ranks by the model's own probabilities; ``deployable`` uses a threshold fitted on
    ``cal`` and never looks at test danger labels.
    """
    truth = np.asarray(rule.evaluate({k: frame[f"truth_{k}"].to_numpy() for k in TARGETS}), dtype=bool)
    probability = frame[f"p_{rule_name}"].to_numpy(dtype=float)
    if threshold is None:
        accepted = probability <= np.quantile(probability, coverage, method="lower")
    else:
        accepted = probability <= threshold
    unsafe = truth
    return {"accepted": int(accepted.sum()),
            "unsafe_accepted": int((unsafe & accepted).sum()),
            "uar": float(unsafe[accepted].mean()) if accepted.sum() else None,
            "hmr": float((unsafe & accepted).sum() / unsafe.sum()) if unsafe.sum() else None,
            "coverage": float(accepted.mean())}


def calibrate_threshold(cal_frame, rule_name, coverage):
    """Deployable threshold: chosen on cal to accept roughly ``coverage`` of cal rows."""
    probability = cal_frame[f"p_{rule_name}"].to_numpy(dtype=float)
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
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--min-nac-valid", type=float, default=0.99)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true",
                        help="re-aggregate existing prediction CSVs without retraining")
    args = parser.parse_args(argv)

    import torch
    seeds = [int(s) for s in args.seeds.split(",")]
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "report").mkdir(exist_ok=True)
    (args.outdir / "pred").mkdir(exist_ok=True)
    (args.outdir / "ckpt").mkdir(exist_ok=True)

    labels_all = pd.read_csv(args.labels)
    cache_meta = load_cache_meta(args.cache)
    labels, inventory = canonical_labels(labels_all, cache_meta, args.min_nac_valid)
    labels_path = args.outdir / "labels_p22.csv"
    labels.to_csv(labels_path, index=False)
    inventory["labels_path"] = str(labels_path)
    inventory["labels_file_sha256"] = runner.file_sha256(labels_path)
    rules = standard_rules()

    manifest = {"inventory": inventory, "rules": {k: v.to_dict() for k, v in rules.items()},
                "models": {}, "device": None, "code_sha": runner.git_sha(),
                "labels_source_sha256": runner.file_sha256(args.labels),
                "cache_sha256": runner.file_sha256(args.cache)}

    if not args.skip_train:
        device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                              else "cpu")
        manifest["device"] = str(device)

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
                "note": "Gaussian residual posterior in physical units, cal-fitted isotropic T"}, indent=2),
                encoding="utf-8")

            result, probabilities, fractions, _ = train_direct_event(
                labels, cache, cache_meta, seed, args.epochs, 32, 1e-3, device)
            truth = {k: labels[c].to_numpy(dtype=float) for k, c in TARGETS.items()}
            base = pd.DataFrame({"sample_id": labels["sample_id"].astype(str).to_numpy(),
                                 "group_id": labels["group_id"].astype(str).to_numpy(),
                                 "split": labels["split"].astype(str).to_numpy()})
            for key in TARGETS:
                base[f"truth_{key}"] = truth[key]
                base[f"mean_{key}"] = fractions[:, 0] if key == "g10" else (
                    fractions[:, 1] if key == "g15" else np.nan)
            for name, probability in probabilities.items():
                base[f"p_{name}"] = probability
            base.to_csv(args.outdir / "pred" / f"direct_event_seed{seed}.csv", index=False)
            (args.outdir / "report" / f"direct_event_seed{seed}.json").write_text(json.dumps({
                "model": "direct_event", "seed": seed, "parameters": result["parameters"],
                "best_epoch": result["best_epoch"], "rules": list(probabilities),
                "note": "only the two preregistered event heads; no probabilities for other rules"}, indent=2),
                encoding="utf-8")

        # --- posterior variants ---------------------------------------------
        for variant in ("point", "diagonal", "joint"):
            for seed in seeds:
                stem = f"{variant}_seed{seed}"
                argv_run = ["--labels", str(labels_path), "--cache", str(args.cache),
                            "--variant", variant, "--seed", str(seed),
                            "--epochs", str(args.epochs), "--draws", str(args.draws),
                            "--calibrate", "--deterministic",
                            "--save-checkpoint", str(args.outdir / "ckpt" / f"{stem}.pt"),
                            "--predictions-out", str(args.outdir / "pred" / f"{stem}.csv"),
                            "--out", str(args.outdir / "report" / f"{stem}.json")]
                if variant == "joint":
                    argv_run += ["--ablate-u", "--ablate-u-calibration-mode", "both",
                                 "--matched-marginals"]
                print(f"[compare_p22] training {stem}", flush=True)
                runner.main(argv_run)

    # --- aggregation ---------------------------------------------------------
    frames = {}
    for path in sorted((args.outdir / "pred").glob("*.csv")):
        frames[path.stem] = pd.read_csv(path)
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

    summary = {"models": {}, "pairs": {}, "coverage": {}, "monte_carlo": {}}
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
        pa = np.clip(a[column].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        pb = np.clip(b[column].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        groups = a["group_id"].astype(str).to_numpy()
        return bootstrap_two(pa, pb, truth, groups, args.bootstrap_draws)

    def bootstrap_two(pa, pb, truth, groups, draws):
        brier = (pa - truth) ** 2 - (pb - truth) ** 2
        nll = -(truth * np.log(pa) + (1 - truth) * np.log(1 - pa)) + \
              (truth * np.log(pb) + (1 - truth) * np.log(1 - pb))
        return {"brier": paired_group_bootstrap(brier, groups, draws=draws),
                "nll": paired_group_bootstrap(nll, groups, draws=draws),
                "rows": int(len(truth)), "prevalence": float(np.mean(truth))}

    pairs = {}
    for seed in seeds:
        pairs[f"seed{seed}"] = {}
        for label, (name_a, name_b) in {
                "joint_vs_point": ("joint", "point"),
                "joint_vs_diagonal": ("joint", "diagonal"),
                "point_vs_ridge": ("point", "coarse_ridge"),
                "diagonal_vs_ridge": ("diagonal", "coarse_ridge"),
                "joint_vs_ridge": ("joint", "coarse_ridge"),
                "point_vs_direct_event": ("point", "direct_event"),
                "diagonal_vs_direct_event": ("diagonal", "direct_event"),
                "joint_vs_direct_event": ("joint", "direct_event")}.items():
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
            pb = np.clip(test_joint[f"p_{rule_name}"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
            ablations[rule_name] = {}
            for suffix, title in (("u0", "joint_u0_shared_T"), ("u0_refit", "joint_u0_refit_T"),
                                  ("matched", "joint_matched_marginals")):
                column = f"p_{rule_name}__{suffix}"
                if column not in test_joint:
                    continue
                pa = np.clip(test_joint[column].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
                ablations[rule_name][title] = bootstrap_two(
                    pa, pb, truth, test_joint["group_id"].astype(str).to_numpy(),
                    args.bootstrap_draws)
        pairs[f"seed{seed}"]["joint_ablations"] = ablations
    summary["pairs"] = pairs

    # coverage: oracle ranking + deployable cal threshold, for the primary rules
    for seed in seeds:
        summary["coverage"][f"seed{seed}"] = {}
        for model in ("coarse_ridge", "direct_event", "point", "diagonal", "joint"):
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
