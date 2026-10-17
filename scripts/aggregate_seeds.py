#!/usr/bin/env python3
"""Aggregate the multi-seed post-fix baseline runs.

Reads ``outputs/baseline_fixed_seed<SEED>/{metrics.json,test_predictions.csv}``
for every requested seed and produces:

* per-model / per-rule mean and std across seeds,
* paired group-bootstrap CIs on the *differences* of interest
  (fusion minus coarse_grid, fusion minus coarse_ridge),
* a calibration audit for both event rules (raw vs calibrated),
* a check that the g15 head is not a copy of the g10 head,
* split-independence assertions on the hydrated labels.

All seeds must cover the identical tile set and group assignment; the script
fails loudly if they do not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_baseline as tb  # noqa: E402  (single source of truth for rules/variants)

RULES = tb.RULES
HEAD_RULES = tb.HEAD_RULES
COMPOSITE_RULE = tb.COMPOSITE_RULE
CNN_VARIANTS = tb.CNN_VARIANTS
MODELS = ("constant", "coarse_ridge", "cnn_nac", "cnn_coarse_grid", "cnn_fusion")
COMPARISONS = (
    ("cnn_fusion", "cnn_coarse_grid"),
    ("cnn_fusion", "coarse_ridge"),
    ("cnn_coarse_grid", "coarse_ridge"),
)
COVERAGES = (0.2, 0.4, 0.6, 0.8)
PREDICTION_COLUMNS = {
    "constant": "constant_g10",
    "coarse_ridge": "coarse_g10",
    **{f"cnn_{v}": f"cnn_{v}_g10" for v in CNN_VARIANTS},
}
# Each rule's MAE is measured against the fraction column of its target attribute.
FRACTION_COLUMNS = {}
for _rule, (_target, _threshold) in RULES.items():
    _suffix = "g15" if _target.endswith("g15") else "g10"
    FRACTION_COLUMNS[_rule] = {
        "constant": f"constant_{_suffix}",
        "coarse_ridge": f"coarse_{_suffix}",
        **{f"cnn_{v}": f"cnn_{v}_{_suffix}" for v in CNN_VARIANTS},
    }
G15_FRACTION_COLUMNS = {f"cnn_{v}": f"cnn_{v}_g15" for v in CNN_VARIANTS}


def load_seed(directory: Path) -> tuple[dict, pd.DataFrame]:
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(directory / "test_predictions.csv")
    return metrics, predictions


def assert_identical_cohorts(seeds: list[int], frames: dict[int, pd.DataFrame]) -> dict:
    reference = frames[seeds[0]]
    for seed in seeds[1:]:
        frame = frames[seed]
        if not frame["sample_id"].equals(reference["sample_id"]):
            raise SystemExit(f"seed {seed} covers a different tile set/order than seed {seeds[0]}")
        if not frame["group_id"].equals(reference["group_id"]):
            raise SystemExit(f"seed {seed} has a different group assignment")
    return {
        "tiles": int(len(reference)),
        "groups": int(reference["group_id"].nunique()),
        "zones": int(reference["LTM_CODE"].nunique()),
        "unsafe_total": int((reference["y_g10"] > RULES["g10_gt_0.05"][1]).sum()),
    }


def check_split_independence(labels: pd.DataFrame) -> dict:
    mapping = labels.groupby("group_id")["split"].nunique()
    overlap = mapping[mapping > 1]
    products = labels.groupby("NAC_EDR_ID")["split"].nunique()
    return {
        "groups": int(labels["group_id"].nunique()),
        "groups_in_multiple_splits": int(len(overlap)),
        "products_in_multiple_splits": int((products > 1).sum()),
        "splits": {k: int(v) for k, v in labels["split"].value_counts().items()},
        "usable": int(labels["nac_valid_fraction"].ge(0.99).sum()) if "nac_valid_fraction" in labels else None,
    }


def nll(probability: np.ndarray, truth: np.ndarray) -> float:
    """Binary NLL with clipping; the probability is the *primary* (calibrated) one."""
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    t = np.asarray(truth, dtype=np.float64)
    return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean())


def assert_truth_consistency(frames: dict[int, pd.DataFrame], seeds: list[int]) -> dict:
    """Every seed must carry identical truths, event labels, groups and zones."""
    reference = frames[seeds[0]]
    audit = {}
    for column in [c for c in reference.columns if c.startswith("event_")] + ["group_id", "LTM_CODE"]:
        if column not in reference.columns:
            raise SystemExit(f"test_predictions.csv is missing {column}; rerun the trainer")
        values = reference[column]
        for seed in seeds[1:]:
            other = frames[seed]
            if column not in other.columns:
                raise SystemExit(f"seed {seed} is missing {column}")
            if column in ("group_id", "LTM_CODE"):
                same = other[column].astype(str).equals(values.astype(str))
            else:
                same = np.array_equal(other[column].to_numpy(), values.to_numpy())
            if not same:
                raise SystemExit(f"seed {seed} disagrees with seed {seeds[0]} on {column}")
        audit[column] = {"checked_seeds": list(seeds)}
    for column in ("y_g10", "y_g15"):
        values = reference[column].to_numpy(dtype=np.float64)
        for seed in seeds[1:]:
            other = frames[seed][column].to_numpy(dtype=np.float64)
            if values.shape != other.shape or not np.allclose(values, other, rtol=0, atol=1e-12):
                raise SystemExit(f"seed {seed} disagrees with seed {seeds[0]} on {column}")
        audit[column] = {"checked_seeds": list(seeds)}
    audit["unsafe_total"] = int((reference["y_g10"] > RULES["g10_gt_0.05"][1]).sum())
    return audit


def assert_probability_quality(frames: dict[int, pd.DataFrame], seeds: list[int],
                               metrics: dict[int, dict]) -> dict:
    """Refuse missing/non-finite/out-of-range probabilities and uncalibrated 'primary' claims."""
    audit = {"checked": 0}
    for seed in seeds:
        frame = frames[seed]
        for model in MODELS:
            for rule_name in RULES:
                column = f"{model}_prob_{rule_name}"
                payload = metrics[seed]["models"][model]["test"][rule_name]
                if column not in frame.columns:
                    raise SystemExit(f"seed {seed}: missing probability column {column}")
                values = frame[column].to_numpy(dtype=np.float64)
                if not np.isfinite(values).all():
                    raise SystemExit(f"seed {seed}: non-finite probabilities in {column}")
                if values.min() < 0.0 or values.max() > 1.0:
                    raise SystemExit(f"seed {seed}: probabilities outside [0, 1] in {column}")
                if payload.get("primary") == "calibrated" and payload.get("metrics_calibrated") is None:
                    raise SystemExit(f"seed {seed}: {model}/{rule_name} claims calibrated primary "
                                     "but has no calibrated metrics")
                audit["checked"] += int(len(values))
    return audit


REQUIRED_IDENTICAL_PROTOCOL = (
    "input_size", "augment", "augmentation_policy", "epochs_max", "patience", "batch", "lr",
    "min_nac_valid", "channels", "scalar_scaling", "calibration", "rules", "composite_rule",
    "contrast_channel", "ablation",
)


def assert_seed_protocol_consistency(metrics: dict[int, dict], seeds: list[int]) -> dict:
    """Seeds of one config must share hyperparameters, ablation flags and data provenance."""
    missing_seeds = [seed for seed in seeds if seed not in metrics]
    if missing_seeds:
        raise SystemExit(f"missing metrics.json for seeds {missing_seeds}")
    reference = metrics[seeds[0]]["protocol"]
    audit = {"missing": [], "unknown": []}
    for key in REQUIRED_IDENTICAL_PROTOCOL:
        if key not in reference:
            audit["missing"].append(key)
            continue
        for seed in seeds[1:]:
            if metrics[seed]["protocol"].get(key) != reference[key]:
                raise SystemExit(f"seed {seed} protocol.{key} differs from seed {seeds[0]}")
    for seed in seeds:
        if int(metrics[seed]["protocol"].get("seed", -1)) != int(seed):
            raise SystemExit(f"run for seed {seed} recorded protocol.seed="
                             f"{metrics[seed]['protocol'].get('seed')}")
    reference_provenance = reference.get("provenance")
    if reference_provenance is None:
        audit["unknown"].append("provenance")
    else:
        for seed in seeds[1:]:
            other = metrics[seed]["protocol"].get("provenance")
            if other != reference_provenance:
                raise SystemExit(f"seed {seed} was trained on different data provenance than seed {seeds[0]}")
    audit["provenance"] = reference_provenance if reference_provenance is not None else "UNKNOWN"
    audit["ablation"] = reference.get("ablation", "UNKNOWN")
    return audit


def require_split_independence(audit: dict) -> None:
    """Fail rather than silently publishing metrics from leaked split groups."""
    if audit["groups_in_multiple_splits"] or audit["products_in_multiple_splits"]:
        raise SystemExit(
            "split leakage detected: "
            f"{audit['groups_in_multiple_splits']} groups and "
            f"{audit['products_in_multiple_splits']} source products span multiple splits"
        )


def event_head_divergence(frames: dict[int, pd.DataFrame], seeds: list[int]) -> dict:
    """Audit raw binary-event outputs, NOT correlated g10/g15 regression fractions."""
    result = {}
    for variant in CNN_VARIANTS:
        model = f"cnn_{variant}"
        result[model] = {}
        for seed in seeds:
            frame = frames[seed]
            left_key = f"{model}_raw_event_prob_g10_gt_0.05"
            right_key = f"{model}_raw_event_prob_g15_gt_0.02"
            missing = [key for key in (left_key, right_key) if key not in frame.columns]
            if missing:
                raise SystemExit(f"seed {seed}: missing {missing}; rerun the P1.7 trainer")
            left = frame[left_key].to_numpy(dtype=np.float64)
            right = frame[right_key].to_numpy(dtype=np.float64)
            identical = bool(np.allclose(left, right))
            pearson = (float(np.corrcoef(left, right)[0, 1])
                       if np.std(left) > 0 and np.std(right) > 0 else None)
            result[model][str(seed)] = {
                "mean_abs_gap": float(np.abs(left - right).mean()),
                "pearson": pearson,
                "identical": identical,
            }
            if identical:
                raise SystemExit(f"seed {seed} {model}: g10/g15 raw event heads are identical")
    return result


def mae(values: np.ndarray, prediction: np.ndarray, index: np.ndarray) -> float:
    return float(np.abs(values[index] - prediction[index]).mean())


def brier(truth: np.ndarray, probability: np.ndarray, index: np.ndarray) -> float:
    return float(((probability[index] - truth[index]) ** 2).mean())


def paired_bootstrap_difference(statistic, y_true: np.ndarray,
                                per_seed: dict[int, tuple[np.ndarray, np.ndarray]],
                                groups: np.ndarray, n: int, seed: int) -> dict:
    """statistic(y, prediction, index) evaluated on both models, per seed, then averaged."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique}
    point = {
        seed_id: statistic(y_true, left, np.arange(len(y_true))) - statistic(y_true, right, np.arange(len(y_true)))
        for seed_id, (left, right) in per_seed.items()
    }
    differences = []
    for _ in range(n):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in chosen])
        per_seed_difference = [
            statistic(y_true, left, idx) - statistic(y_true, right, idx)
            for left, right in per_seed.values()
        ]
        differences.append(float(np.mean(per_seed_difference)))
    differences = np.asarray(differences)
    return {
        "mean_over_seeds": float(np.mean(list(point.values()))),
        "per_seed": {str(k): float(v) for k, v in point.items()},
        "lo95": float(np.percentile(differences, 2.5)),
        "hi95": float(np.percentile(differences, 97.5)),
        "n_groups": int(len(unique)),
        "n_resamples": n,
        "excludes_zero": bool(np.percentile(differences, 2.5) > 0 or np.percentile(differences, 97.5) < 0),
    }


def environment_info(seeds: list[int], pattern: str) -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        import platform
        info["platform"] = platform.platform()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3)
            if torch.cuda.is_available() else None,
        })
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    elapsed = {}
    for seed in seeds:
        path = Path(pattern.format(seed=seed)) / "elapsed_sec.txt"
        if path.exists():
            elapsed[str(seed)] = float(path.read_text().strip())
    info["elapsed_sec_per_seed"] = elapsed
    info["elapsed_sec_total"] = float(sum(elapsed.values())) if elapsed else None
    for name in ("outputs/cache_v3.npz", "outputs/labels_hydrated.csv"):
        path = Path(name)
        info[f"bytes_{Path(name).name}"] = path.stat().st_size if path.exists() else None
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="outputs/baseline_v3_seed{seed}")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels_hydrated.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/multiseed_summary"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    metrics, frames = {}, {}
    for seed in seeds:
        directory = Path(args.pattern.format(seed=seed))
        if not (directory / "metrics.json").exists():
            raise SystemExit(f"missing {directory}/metrics.json")
        metrics[seed], frames[seed] = load_seed(directory)

    cohort = assert_identical_cohorts(seeds, frames)
    truth_audit = assert_truth_consistency(frames, seeds)
    probability_audit = assert_probability_quality(frames, seeds, metrics)
    protocol_audit = assert_seed_protocol_consistency(metrics, seeds)
    labels = pd.read_csv(args.labels)
    split_audit = check_split_independence(labels)
    require_split_independence(split_audit)
    head_audit = event_head_divergence(frames, seeds)
    reference = frames[seeds[0]]
    y_g10 = reference["y_g10"].to_numpy(dtype=np.float64)
    y_g15 = reference["y_g15"].to_numpy(dtype=np.float64)
    groups = reference["group_id"].to_numpy()
    truth_g10 = (y_g10 > RULES["g10_gt_0.05"][1]).astype(np.float64)
    truth_g15 = (y_g15 > RULES["g15_gt_0.02"][1]).astype(np.float64)

    # ---- per-model summaries across seeds ----
    summary = {}
    for model in MODELS:
        summary[model] = {}
        for rule_name in RULES:
            keys = ("mae", "rmse", "brier", "brier_raw", "auc", "auc_raw",
                    "brier_constant", "nll_primary", "nll_raw")
            collected = {k: [] for k in keys}
            for seed in seeds:
                payload = metrics[seed]["models"][model]["test"][rule_name]
                primary = payload["metrics"]
                raw = payload["metrics_raw"]
                frame = frames[seed]
                truth = frame[f"event_{rule_name}"].to_numpy(dtype=np.float64)
                probability = frame[f"{model}_prob_{rule_name}"].to_numpy(dtype=np.float64)
                # Cross-check the trainer's primary Brier against the exported probabilities.
                recomputed = float(((probability - truth) ** 2).mean())
                if abs(recomputed - float(primary["brier"])) > 1e-6:
                    raise SystemExit(f"seed {seed} {model}/{rule_name}: exported probabilities give "
                                     f"Brier {recomputed:.6f} but metrics.json reports "
                                     f"{primary['brier']:.6f}; refusing to publish")
                collected["mae"].append(primary.get("mae"))
                collected["rmse"].append(primary.get("rmse"))
                collected["brier"].append(primary.get("brier"))
                collected["brier_raw"].append(raw.get("brier"))
                collected["auc"].append(raw.get("auc"))
                collected["auc_raw"].append(raw.get("auc"))
                collected["brier_constant"].append(primary.get("brier_constant"))
                collected["nll_primary"].append(nll(probability, truth))
                collected["nll_raw"].append(raw.get("nll"))
            summary[model][rule_name] = {
                key: {
                    "mean": float(np.mean(clean)) if (clean := [v for v in values if v is not None]) else None,
                    "std": float(np.std(clean)) if clean else None,
                    "per_seed": values,
                } for key, values in collected.items()
            }

    # ---- calibration audit ----
    calibration = {}
    for rule_name in RULES:
        calibration[rule_name] = {}
        for model in MODELS:
            payload = metrics[seeds[0]]["models"][model]["test"][rule_name]
            calibration[rule_name][model] = {
                "raw": {k: payload["metrics_raw"].get(k) for k in ("brier", "nll", "auc")},
                "calibrated": None if payload["metrics_calibrated"] is None else {
                    k: payload["metrics_calibrated"].get(k) for k in ("brier", "nll", "auc")
                },
                "platt": payload["platt"],
                "primary": payload["primary"],
                "brier_constant": payload["metrics_raw"].get("brier_constant"),
            }

    # ---- paired differences (primary probability of each model) ----
    def primary_probability(seed: int, model: str, rule_name: str) -> np.ndarray:
        column = f"{model}_prob_{rule_name}"
        frame = frames[seed]
        if column not in frame.columns:
            raise SystemExit(f"{column} missing from test_predictions.csv; rerun train_baseline.py")
        return frame[column].to_numpy(dtype=np.float64)

    differences = {}
    for left_model, right_model in COMPARISONS:
        key = f"{left_model}_minus_{right_model}"
        differences[key] = {}
        for rule_name, (target, threshold) in RULES.items():
            values = y_g10 if target.endswith("g10") else y_g15
            truth = (values > threshold).astype(np.float64)
            fraction_pairs = {}
            for seed in seeds:
                frame = frames[seed]
                left = frame[FRACTION_COLUMNS[rule_name][left_model]].to_numpy(dtype=np.float64)
                right = frame[FRACTION_COLUMNS[rule_name][right_model]].to_numpy(dtype=np.float64)
                fraction_pairs[seed] = (left, right)
            probability_pairs = {
                seed: (primary_probability(seed, left_model, rule_name),
                       primary_probability(seed, right_model, rule_name))
                for seed in seeds
            }
            differences[key][rule_name] = {
                "mae": paired_bootstrap_difference(mae, values, fraction_pairs,
                                                   groups, args.bootstrap, 42),
                "brier": paired_bootstrap_difference(brier, truth, probability_pairs,
                                                     groups, args.bootstrap, 42),
            }

    # ---- risk-coverage at fixed coverage, mean across seeds ----
    risk_coverage = {}
    for model in MODELS:
        risk_coverage[model] = {}
        for rule_name in RULES:
            rows_by_coverage = {c: [] for c in COVERAGES}
            for seed in seeds:
                for entry in metrics[seed]["models"][model]["test"][rule_name]["risk_coverage"]:
                    if entry["coverage_requested"] in rows_by_coverage:
                        rows_by_coverage[entry["coverage_requested"]].append(entry)
            risk_coverage[model][rule_name] = {
                str(coverage): {
                    "UAR_mean": float(np.mean([e["UAR"] for e in entries])),
                    "UAR_std": float(np.std([e["UAR"] for e in entries])),
                    "HMR_mean": float(np.mean([e["HMR"] for e in entries if e["HMR"] is not None])),
                    "accepted_n": int(entries[0]["accepted_n"]),
                    "unsafe_accepted_n_mean": float(np.mean([e["unsafe_accepted_n"] for e in entries])),
                } for coverage, entries in rows_by_coverage.items()
            }

    # ---- rule prevalence (from the exported event columns) ----
    prevalence = {}
    for rule_name in RULES:
        column = f"event_{rule_name}"
        if column in reference.columns:
            prevalence[rule_name] = float(reference[column].mean())
        else:
            target, threshold = RULES[rule_name]
            values = y_g15 if target.endswith("g15") else y_g10
            prevalence[rule_name] = float((values > threshold).mean()) if target != "composite" else None

    # ---- g15 vs g10 head divergence ----
    divergence = {}
    for model, g15_column in G15_FRACTION_COLUMNS.items():
        g10_column = PREDICTION_COLUMNS[model]
        per_seed = {}
        for seed in seeds:
            frame = frames[seed]
            g10_values = frame[g10_column].to_numpy(dtype=np.float64)
            g15_values = frame[g15_column].to_numpy(dtype=np.float64)
            per_seed[str(seed)] = {
                "mean_abs_gap": float(np.abs(g10_values - g15_values).mean()),
                "pearson": float(np.corrcoef(g10_values, g15_values)[0, 1]),
                "identical": bool(np.allclose(g10_values, g15_values)),
            }
        divergence[model] = per_seed

    protocol = {
        "seeds": seeds,
        "seed_count": len(seeds),
        "git_sha": _git_sha(),
        "rules": {k: {"target": v[0], "threshold": v[1]} for k, v in RULES.items()},
        "bootstrap_resamples": args.bootstrap,
        "models": list(MODELS),
    }
    payload = {
        "protocol": protocol,
        "environment": environment_info(seeds, args.pattern),
        "cohort": cohort,
        "truth_consistency": truth_audit,
        "probability_quality": probability_audit,
        "seed_protocol": protocol_audit,
        "field_sources": {
            "mae": "metrics.json metrics.mae (primary probability)",
            "brier": "metrics.json metrics.brier (primary) + recomputed cross-check",
            "brier_raw": "metrics.json metrics_raw.brier",
            "nll_primary": "recomputed from test_predictions {model}_prob_{rule} vs event_{rule}",
            "nll_raw": "metrics.json metrics_raw.nll",
            "auc": "metrics.json metrics_raw.auc",
        },
        "split_independence": split_audit,
        "event_head_divergence": head_audit,
        "test_rule_prevalence": prevalence,
        "per_model": summary,
        "calibration_audit": calibration,
        "paired_differences": differences,
        "risk_coverage_mean_across_seeds": risk_coverage,
        "g15_vs_g10_divergence": divergence,
        "source_directories": [args.pattern.format(seed=seed) for seed in seeds],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out / "summary.md").write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"wrote {args.out}/summary.json and {args.out}/summary.md")
    return 0


def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def render_markdown(payload: dict) -> str:
    lines = ["# Post-fix multi-seed summary", "",
             f"- seeds: {payload['protocol']['seeds']} (n={payload['protocol']['seed_count']})",
             f"- git: `{payload['protocol']['git_sha']}`",
             f"- cohort: {payload['cohort']['tiles']} tiles / {payload['cohort']['groups']} groups / "
             f"{payload['cohort']['zones']} zones / {payload['cohort']['unsafe_total']} unsafe",
             f"- split independence: {payload['split_independence']}",
             f"- environment: {payload.get('environment', {})}",
             f"- test rule prevalence: {payload.get('test_rule_prevalence', {})}", ""]
    lines += ["## Mean (std) across seeds, primary probability", "",
              "| model | rule | MAE | RMSE | Brier (primary) | Brier (raw) | AUC | NLL (primary) | NLL (raw) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for model, rules in payload["per_model"].items():
        for rule_name, values in rules.items():
            def cell(key):
                entry = values[key]
                return f"{entry['mean']:.4f} ({entry['std']:.4f})"
            auc = values["auc"]["mean"]
            lines.append(f"| {model} | {rule_name} | {cell('mae')} | {cell('rmse')} | "
                         f"{cell('brier')} | {cell('brier_raw')} | {auc if auc is None else f'{auc:.4f}'} | "
                         f"{cell('nll_primary')} | {cell('nll_raw')} |")
    lines += ["", "## Paired group-bootstrap differences (primary probabilities)", "",
              "| comparison | rule | ΔMAE mean [95% CI] | ΔBrier mean [95% CI] | excludes 0 |",
              "|---|---|---|---|---|"]
    for comparison, rules in payload["paired_differences"].items():
        for rule_name, entry in rules.items():
            mae_entry, brier_entry = entry["mae"], entry["brier"]
            lines.append(
                f"| {comparison} | {rule_name} | "
                f"{mae_entry['mean_over_seeds']:+.4f} [{mae_entry['lo95']:+.4f}, {mae_entry['hi95']:+.4f}] | "
                f"{brier_entry['mean_over_seeds']:+.4f} [{brier_entry['lo95']:+.4f}, {brier_entry['hi95']:+.4f}] | "
                f"{mae_entry['excludes_zero'] or brier_entry['excludes_zero']} |")
    lines += ["", "## Risk-coverage (mean UAR across seeds, primary probability)", "",
              "| model | coverage | UAR | HMR | accepted_n | unsafe_accepted_n |", "|---|---|---|---|---|---|"]
    for model, rules in payload["risk_coverage_mean_across_seeds"].items():
        for rule_name, coverages in rules.items():
            for coverage, entry in coverages.items():
                lines.append(f"| {model} ({rule_name}) | {coverage} | {entry['UAR_mean']:.3f} | "
                             f"{entry['HMR_mean']:.3f} | {entry['accepted_n']} | "
                             f"{entry['unsafe_accepted_n_mean']:.1f} |")
    lines += ["", "## g15 vs g10 fraction-regression divergence (not event heads)", "",
              "| model | mean |Δ| | pearson | identical |", "|---|---|---|---|"]
    for model, per_seed in payload["g15_vs_g10_divergence"].items():
        for seed, entry in per_seed.items():
            lines.append(f"| {model} (seed {seed}) | {entry['mean_abs_gap']:.4f} | "
                         f"{entry['pearson']:.4f} | {entry['identical']} |")
    lines += ["", "## Dedicated binary-event head divergence (raw, before calibration)", "",
              "| model | mean |Δ P(unsafe)| | pearson | identical |",
              "|---|---|---|---|"]
    for model, per_seed in payload["event_head_divergence"].items():
        for seed, entry in per_seed.items():
            corr = entry["pearson"]
            lines.append(f"| {model} (seed {seed}) | {entry['mean_abs_gap']:.4f} | "
                         f"{'n/a' if corr is None else f'{corr:.4f}'} | {entry['identical']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
