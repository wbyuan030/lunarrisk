#!/usr/bin/env python3
"""Cross-configuration comparison of the P1.7 controlled ablations.

Each ablation config is a set of per-seed runs produced by ``train_baseline.py``
(``outputs/<config>_seed<seed>/``). This script answers two questions:

1. How does the *NAC increment* (``cnn_fusion`` minus ``cnn_coarse_grid``) change when
   observation geometry and/or NAC photometry channels are removed?
2. How does the single-modality ``cnn_nac`` model behave across the same configs?

It reads the per-tile predictions directly (not the per-config summaries), asserts that
every config/seed covers exactly the same cohort, and reports group-level paired
bootstrap CIs for each difference, plus a difference-of-differences against the
reference config.
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
import train_baseline as tb  # noqa: E402
import aggregate_seeds as agg  # noqa: E402  (single definition of the NLL formula)

MODELS = ("constant", "coarse_ridge", "cnn_nac", "cnn_coarse_grid", "cnn_fusion")
FRACTION_COLUMNS = {
    "constant": "constant_g10",
    "coarse_ridge": "coarse_g10",
    **{f"cnn_{v}": f"cnn_{v}_g10" for v in tb.CNN_VARIANTS},
}
PROBABILITY_COLUMNS = {model: f"{model}_prob_{tb.PRIMARY_RULE}" for model in MODELS}
PRIMARY = tb.PRIMARY_RULE
TARGET, THRESHOLD = tb.RULES[PRIMARY]
# Truth columns as exported in test_predictions.csv.
TARGET_COLUMNS = {"slope3m_g10": "y_g10", "slope3m_g15": "y_g15"}
TARGET_COLUMN = TARGET_COLUMNS[TARGET]


def load_configs(pattern: str, configs: list[str], seeds: list[int]) -> tuple[dict, dict, dict]:
    frames, metrics, ablation = {}, {}, {}
    reference_ids = None
    group_ids = None
    for config in configs:
        for seed in seeds:
            directory = Path(pattern.format(config=config, seed=seed))
            if not (directory / "metrics.json").exists():
                raise SystemExit(f"missing {directory}/metrics.json")
            frame = pd.read_csv(directory / "test_predictions.csv")
            payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            if reference_ids is None:
                reference_ids = frame["sample_id"].astype(str)
                group_ids = frame["group_id"].astype(str).to_numpy()
                reference_frame = frame
            else:
                if not frame["sample_id"].astype(str).equals(reference_ids):
                    raise SystemExit(f"{directory} covers a different cohort/order")
                if not np.array_equal(frame["group_id"].astype(str).to_numpy(), group_ids):
                    raise SystemExit(f"{directory} has a different group assignment")
                if not frame["LTM_CODE"].astype(str).equals(reference_frame["LTM_CODE"].astype(str)):
                    raise SystemExit(f"{directory} has different LTM zones")
                for column in ("y_g10", "y_g15"):
                    if not np.allclose(frame[column].to_numpy(dtype=np.float64),
                                       reference_frame[column].to_numpy(dtype=np.float64),
                                       rtol=0, atol=1e-12):
                        raise SystemExit(f"{directory} has different truth values in {column}")
                for column in [c for c in reference_frame.columns if c.startswith("event_")]:
                    if column not in frame.columns:
                        raise SystemExit(f"{directory} is missing {column}")
                    if not np.array_equal(frame[column].to_numpy(), reference_frame[column].to_numpy()):
                        raise SystemExit(f"{directory} has different event labels in {column}")
            frames[(config, seed)] = frame
            metrics[(config, seed)] = payload
            ablation[config] = payload.get("protocol", {}).get("ablation", {})
    return frames, metrics, ablation


COMPARABLE_PROTOCOL = (
    "input_size", "augment", "augmentation_policy", "epochs_max", "patience", "batch", "lr",
    "min_nac_valid", "channels", "scalar_scaling", "calibration", "rules", "composite_rule",
    "contrast_channel",
)


def assert_config_protocol_consistency(metrics: dict, ablation: dict, configs: list[str],
                                       seeds: list[int]) -> dict:
    """Only the geometry/photometry switches may differ between configs.

    Everything else (hyperparameters, rules, input size, augmentation policy) must match,
    provenance must be identical, and the recorded channel/scalar lists must mirror the
    recorded ablation flags for every variant.
    """
    reference = metrics[(configs[0], seeds[0])]["protocol"]
    audit = {"missing": [], "configs": {}}
    for config in configs:
        for seed in seeds:
            protocol = metrics[(config, seed)]["protocol"]
            if int(protocol.get("seed", -1)) != int(seed):
                raise SystemExit(f"{config}/seed{seed} recorded protocol.seed="
                                 f"{protocol.get('seed')}")
            for key in COMPARABLE_PROTOCOL:
                if key not in reference or key not in protocol:
                    audit["missing"].append(f"{config}/{seed}:{key}")
                    continue
                if protocol[key] != reference[key]:
                    raise SystemExit(f"{config}/seed{seed} differs from the reference config on "
                                     f"protocol.{key}: {protocol[key]!r} vs {reference[key]!r}")
            if protocol.get("provenance") != reference.get("provenance"):
                raise SystemExit(f"{config}/seed{seed} was trained on different data provenance")
            if protocol.get("ablation") != ablation.get(config):
                raise SystemExit(f"{config}/seed{seed} ablation flags disagree with the audit")
            flags = ablation.get(config) or {}
            for variant in tb.CNN_VARIANTS:
                expected_channels, expected_scalars = tb.ablation_features(
                    variant,
                    use_geometry=flags.get("use_geometry", True),
                    use_nac_photometry=flags.get("use_nac_photometry", True))
                if protocol.get("variant_channels", {}).get(variant) != [tb.CHANNELS[c] for c in expected_channels]:
                    raise SystemExit(f"{config}/seed{seed} variant_channels for {variant} do not "
                                     "match the recorded ablation flags")
                if protocol.get("variant_scalars", {}).get(variant) != list(expected_scalars):
                    raise SystemExit(f"{config}/seed{seed} variant_scalars for {variant} do not "
                                     "match the recorded ablation flags")
            audit["configs"].setdefault(config, set()).add(json.dumps(protocol.get("ablation"), sort_keys=True))
    for config, seen in audit["configs"].items():
        if len(seen) > 1:
            raise SystemExit(f"{config} mixes several ablation flags across seeds: {sorted(seen)}")
        audit["configs"][config] = json.loads(next(iter(seen)))
    audit["missing"] = sorted(set(audit["missing"]))
    audit["provenance"] = reference.get("provenance", "UNKNOWN")
    return audit


def assert_probability_quality(frames: dict, metrics: dict, configs: list[str],
                               seeds: list[int]) -> dict:
    """Refuse missing, non-finite or out-of-range probabilities before combining configs."""
    audit = {"checked": 0}
    for config in configs:
        for seed in seeds:
            frame = frames[(config, seed)]
            for model in MODELS:
                for rule_name in tb.RULES:
                    column = f"{model}_prob_{rule_name}"
                    if column not in frame.columns:
                        raise SystemExit(f"{config}/seed{seed}: missing probability column {column}")
                    values = frame[column].to_numpy(dtype=np.float64)
                    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                        raise SystemExit(f"{config}/seed{seed}: invalid probabilities in {column}")
                    payload = metrics[(config, seed)]["models"][model]["test"][rule_name]
                    if payload.get("primary") == "calibrated" and payload.get("metrics_calibrated") is None:
                        raise SystemExit(f"{config}/seed{seed}: {model}/{rule_name} claims calibrated "
                                         "primary without calibrated metrics")
                    audit["checked"] += int(len(values))
            for variant in tb.CNN_VARIANTS:
                for rule_name in tb.HEAD_RULES:
                    column = f"cnn_{variant}_raw_event_prob_{rule_name}"
                    if column not in frame.columns:
                        raise SystemExit(f"{config}/seed{seed}: missing {column}")
                    values = frame[column].to_numpy(dtype=np.float64)
                    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                        raise SystemExit(f"{config}/seed{seed}: invalid raw event probabilities in {column}")
    return audit


def cohort_table(frames: dict, metrics: dict, configs: list[str], seeds: list[int]) -> dict:
    """Per-config, per-model headline metrics with the seed spread."""
    table = {}
    for config in configs:
        table[config] = {}
        for model in MODELS:
            rows = {"mae": [], "brier": [], "brier_raw": [], "nll_primary": [], "nll_raw": [],
                    "auc": [], "uar20": [], "uar40": []}
            for seed in seeds:
                frame = frames[(config, seed)]
                payload = metrics[(config, seed)]["models"][model]["test"][PRIMARY]
                rows["mae"].append(payload["metrics"]["mae"])
                rows["brier"].append(payload["metrics"]["brier"])
                rows["brier_raw"].append(payload["metrics_raw"]["brier"])
                truth = frame[f"event_{PRIMARY}"].to_numpy(dtype=np.float64)
                probability = frame[PROBABILITY_COLUMNS[model]].to_numpy(dtype=np.float64)
                recomputed = float(((probability - truth) ** 2).mean())
                if abs(recomputed - float(payload["metrics"]["brier"])) > 1e-6:
                    raise SystemExit(f"{config}/seed{seed} {model}: exported probabilities give "
                                     f"Brier {recomputed:.6f} but metrics.json reports "
                                     f"{payload['metrics']['brier']:.6f}")
                rows["nll_primary"].append(agg.nll(probability, truth))
                rows["nll_raw"].append(payload["metrics_raw"]["nll"])
                rows["auc"].append(payload["metrics_raw"]["auc"])
                y = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
                probability = frame[PROBABILITY_COLUMNS[model]].to_numpy(dtype=np.float64)
                coverage = tb.risk_coverage(y, probability, PRIMARY, coverages=(0.2, 0.4))
                rows["uar20"].append(coverage[0]["UAR"])
                rows["uar40"].append(coverage[1]["UAR"])
            table[config][model] = {
                key: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "per_seed": [float(v) for v in values],
                } for key, values in rows.items()
            }
    return table


def paired_difference(frames: dict, config: str, seeds: list[int], groups: np.ndarray,
                      left: str, right: str, statistic: str, n: int, seed: int) -> dict:
    """Group-bootstrap CI for mean_over_seeds(stat(left) - stat(right))."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique}

    def value(frame: pd.DataFrame, index: np.ndarray) -> tuple[float, float]:
        y = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)[index]
        truth = (y > THRESHOLD).astype(np.float64)
        if statistic == "mae":
            left_values = frame[FRACTION_COLUMNS[left]].to_numpy(dtype=np.float64)[index]
            right_values = frame[FRACTION_COLUMNS[right]].to_numpy(dtype=np.float64)[index]
            return float(np.abs(y - left_values).mean()), float(np.abs(y - right_values).mean())
        left_probability = frame[PROBABILITY_COLUMNS[left]].to_numpy(dtype=np.float64)[index]
        right_probability = frame[PROBABILITY_COLUMNS[right]].to_numpy(dtype=np.float64)[index]
        return (float(((left_probability - truth) ** 2).mean()),
                float(((right_probability - truth) ** 2).mean()))

    full = np.arange(len(groups))
    point = {seed_id: value(frames[(config, seed_id)], full) for seed_id in seeds}
    point_difference = float(np.mean([a - b for a, b in point.values()]))
    samples = []
    for _ in range(n):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in chosen])
        per_seed = [value(frames[(config, seed_id)], idx) for seed_id in seeds]
        samples.append(float(np.mean([a - b for a, b in per_seed])))
    samples = np.asarray(samples)
    return {
        "statistic": statistic,
        "config": config,
        "left": left,
        "right": right,
        "mean_over_seeds": point_difference,
        "lo95": float(np.percentile(samples, 2.5)),
        "hi95": float(np.percentile(samples, 97.5)),
        "excludes_zero": bool(np.percentile(samples, 2.5) > 0 or np.percentile(samples, 97.5) < 0),
        "per_seed": {str(k): float(a - b) for k, (a, b) in point.items()},
    }


def difference_of_differences(frames: dict, configs: list[str], reference: str, seeds: list[int],
                              groups: np.ndarray, statistic: str, n: int, seed: int) -> list[dict]:
    """Is the fusion-minus-coarse_grid gap different from the reference config's gap?"""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique}
    full = np.arange(len(groups))

    def gap(config: str, index: np.ndarray) -> float:
        per_seed = []
        for seed_id in seeds:
            frame = frames[(config, seed_id)]
            y = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)[index]
            truth = (y > THRESHOLD).astype(np.float64)
            fusion = frame[FRACTION_COLUMNS["cnn_fusion"]].to_numpy(dtype=np.float64)[index]
            coarse = frame[FRACTION_COLUMNS["cnn_coarse_grid"]].to_numpy(dtype=np.float64)[index]
            if statistic == "mae":
                per_seed.append(float(np.abs(y - fusion).mean() - np.abs(y - coarse).mean()))
            else:
                fusion_probability = frame[PROBABILITY_COLUMNS["cnn_fusion"]].to_numpy(dtype=np.float64)[index]
                coarse_probability = frame[PROBABILITY_COLUMNS["cnn_coarse_grid"]].to_numpy(dtype=np.float64)[index]
                per_seed.append(float(((fusion_probability - truth) ** 2).mean()
                                      - ((coarse_probability - truth) ** 2).mean()))
        return float(np.mean(per_seed))

    reference_gap = gap(reference, full)
    results = []
    for config in configs:
        if config == reference:
            continue
        point = gap(config, full) - reference_gap
        samples = []
        for _ in range(n):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([index_by_group[g] for g in chosen])
            samples.append(gap(config, idx) - gap(reference, idx))
        samples = np.asarray(samples)
        results.append({
            "statistic": statistic,
            "config": config,
            "reference": reference,
            "mean": float(point),
            "lo95": float(np.percentile(samples, 2.5)),
            "hi95": float(np.percentile(samples, 97.5)),
            "excludes_zero": bool(np.percentile(samples, 2.5) > 0 or np.percentile(samples, 97.5) < 0),
        })
    return results


def render(payload: dict) -> str:
    lines = ["# P1.7 controlled ablations — cross-config comparison", "",
             f"- configs: {payload['configs']} (reference: {payload['reference']})",
             f"- seeds: {payload['seeds']}",
             f"- cohort: {payload['cohort']['tiles']} test tiles / {payload['cohort']['groups']} groups",
             f"- ablation flags: {json.dumps(payload['ablation'], sort_keys=True)}",
             f"- bootstrap resamples: {payload['bootstrap_resamples']}", "",
             "## Headline metrics by config (mean (std) over seeds)", "",
             "| config | model | MAE g10 | Brier (primary) | Brier (raw) | NLL (primary) | NLL (raw) | AUC | UAR@20 | UAR@40 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for config, models in payload["per_config"].items():
        for model, values in models.items():
            lines.append(
                f"| {config} | {model} | {values['mae']['mean']:.4f} ({values['mae']['std']:.4f}) | "
                f"{values['brier']['mean']:.4f} ({values['brier']['std']:.4f}) | "
                f"{values['brier_raw']['mean']:.4f} | {values['nll_primary']['mean']:.4f} | "
                f"{values['nll_raw']['mean']:.4f} | {values['auc']['mean']:.4f} | "
                f"{values['uar20']['mean']:.3f} | {values['uar40']['mean']:.3f} |")
    lines += ["", "## NAC increment per config (cnn_fusion minus cnn_coarse_grid, paired group bootstrap)", "",
              "| config | statistic | mean [95% CI] | excludes 0 |", "|---|---|---|---|"]
    for entry in payload["fusion_minus_coarse_grid"]:
        lines.append(f"| {entry['config']} | d{entry['statistic'].upper()} | "
                     f"{entry['mean_over_seeds']:+.4f} [{entry['lo95']:+.4f}, {entry['hi95']:+.4f}] | "
                     f"{entry['excludes_zero']} |")
    lines += ["", f"## Difference of differences vs `{payload['reference']}`", "",
              "| config | statistic | mean [95% CI] | excludes 0 |", "|---|---|---|---|"]
    for entry in payload["difference_of_differences"]:
        lines.append(f"| {entry['config']} | d{entry['statistic'].upper()} | "
                     f"{entry['mean']:+.4f} [{entry['lo95']:+.4f}, {entry['hi95']:+.4f}] | "
                     f"{entry['excludes_zero']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="outputs/{config}_seed{seed}")
    parser.add_argument("--configs", default="p17_full,p17_no_geo,p17_no_photo,p17_no_both")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--reference", default="p17_full")
    parser.add_argument("--out", type=Path, default=Path("outputs/p17_ablation_summary"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.reference not in configs:
        raise SystemExit(f"reference {args.reference} is not in --configs")

    frames, metrics, ablation = load_configs(args.pattern, configs, seeds)
    groups = frames[(configs[0], seeds[0])]["group_id"].astype(str).to_numpy()
    cohort = {
        "tiles": int(len(groups)),
        "groups": int(len(np.unique(groups))),
        "unsafe": int(((frames[(configs[0], seeds[0])][TARGET_COLUMN] > THRESHOLD)).sum()),
    }

    protocol_audit = assert_config_protocol_consistency(metrics, ablation, configs, seeds)
    probability_audit = assert_probability_quality(frames, metrics, configs, seeds)
    per_config = cohort_table(frames, metrics, configs, seeds)
    fusion_differences, dod = [], []
    for config in configs:
        for statistic in ("mae", "brier"):
            fusion_differences.append(paired_difference(
                frames, config, seeds, groups, "cnn_fusion", "cnn_coarse_grid",
                statistic, args.bootstrap, 42))
        for statistic in ("mae", "brier"):
            dod.extend(difference_of_differences(
                frames, [config], args.reference, seeds, groups, statistic, args.bootstrap, 42))

    payload = {
        "configs": configs,
        "reference": args.reference,
        "seeds": seeds,
        "cohort": cohort,
        "ablation": ablation,
        "protocol_consistency": protocol_audit,
        "probability_quality": probability_audit,
        "field_sources": {
            "mae": "metrics.json metrics.mae (primary probability)",
            "brier_primary": "metrics.json metrics.brier + recomputed cross-check from the CSV",
            "brier_raw": "metrics.json metrics_raw.brier",
            "nll_primary": "recomputed from {model}_prob_{rule} vs event_{rule} (aggregate_seeds.nll)",
            "nll_raw": "metrics.json metrics_raw.nll",
            "auc": "metrics.json metrics_raw.auc (raw ranking)",
            "uar20_uar40": "train_baseline.risk_coverage on the primary probability (tie-aware)",
        },
        "caveats": [
            "CIs are a 25-group cluster bootstrap averaged over 3 fixed seeds; this is not a "
            "coverage guarantee for new regions or random initialisations.",
            "The 110-tile test set has been inspected across several rounds: treat all "
            "comparisons as exploratory (development_test).",
            "Multi-config / multi-rule comparisons are exploratory and not pre-registered.",
        ],
        "bootstrap_resamples": args.bootstrap,
        "per_config": per_config,
        "fusion_minus_coarse_grid": fusion_differences,
        "difference_of_differences": dod,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out / "summary.md").write_text(render(payload), encoding="utf-8")
    print(render(payload))
    print(f"wrote {args.out}/summary.json and {args.out}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
