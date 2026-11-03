"""A fixed, Train-only AST diagnostic using existing P2.2 checkpoints.

This is retrospective development analysis, not preregistered confirmation. All thresholds and
compositions are written before evaluation; no rule, architecture or checkpoint is selected from
these results. The two fixed direct-event heads cannot transfer to the generated rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_posterior_p21 as runner
import compare_p22 as comparison
from lunarrisk.rules import And, Compare, Or


def train_rules(labels):
    """Evaluate every combination in this fixed specification; never inspect test quantiles."""
    train = labels[labels["split"] == "train"]
    if train.empty:
        raise ValueError("rules need Train rows")
    rules = {}
    medians = {}
    for attribute, column in runner.TARGET_COLUMNS.items():
        values = train[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("non-finite Train target")
        medians[attribute] = float(np.quantile(values, 0.5))
        for quantile in (0.25, 0.5, 0.75):
            rules[f"{attribute}_train_q{int(quantile*100)}"] = Compare(
                attribute, float(np.quantile(values, quantile)))
    for first, second in (("g10", "roughness_9m"), ("g15", "relief_p95_p5"),
                          ("roughness_9m", "relief_p95_p5")):
        children = (Compare(first, medians[first]), Compare(second, medians[second]))
        rules[f"{first}_AND_{second}_train_medians"] = And(children)
        rules[f"{first}_OR_{second}_train_medians"] = Or(children)
    return rules


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    args.outdir.mkdir(exist_ok=False, parents=True)
    labels_path = args.run / "labels_p22.csv"
    labels, cache, scalars = runner.load_dataset(labels_path, args.cache, runner.load_baseline_module())
    rules = train_rules(labels)
    manifest = {"status": "retrospective development diagnostic; no selection",
                "threshold_source": "train only", "quantiles": [0.25, 0.5, 0.75],
                "rules": {name: rule.to_dict() for name, rule in rules.items()},
                "labels_sha256": runner.file_sha256(labels_path),
                "script_sha256": runner.file_sha256(__file__)}
    rule_path = args.outdir / "rules.json"
    rule_path.write_text(json.dumps(manifest, indent=2) + "\n")
    test = labels[labels["split"] == "test"]
    ids = test["sample_id"].astype(str).tolist()
    truth = {key: test[column].to_numpy(dtype=float) for key, column in runner.TARGET_COLUMNS.items()}
    records = []
    frames = {}
    marginal_records = []
    torch.use_deterministic_algorithms(True)
    for checkpoint in sorted((args.run / "ckpt").glob("*.pt")):
        if checkpoint.stem.split("_seed")[0] not in ("point", "diagonal", "joint"):
            continue
        _, payload = runner.load_checkpoint(checkpoint)
        seed = payload["sampling_seed"]
        physical, probabilities, _ = runner.predict_from_checkpoint(
            checkpoint, ids, cache, scalars, rules, args.device, draws=args.draws, seed=seed)
        frame = pd.DataFrame({"sample_id": ids, "group_id": test["group_id"].astype(str).tolist()})
        for name, rule in rules.items():
            event = rule.evaluate(truth).astype(int)
            p = probabilities[name]
            frame[f"event_{name}"] = event
            frame[f"p_{name}"] = p
            clipped = np.clip(p, 1e-6, 1-1e-6)
            records.append({"model": checkpoint.stem, "rule": name, "rows": len(ids),
                            "unsafe_total": int(event.sum()), "brier": float(np.mean((p-event)**2)),
                            "nll": float(-np.mean(event*np.log(clipped)+(1-event)*np.log(1-clipped)))})
        frames[checkpoint.stem] = frame
        frame.to_csv(args.outdir / f"{checkpoint.stem}.csv", index=False)
        if payload["variant"] == "joint":
            matched, _, _ = runner.predict_from_checkpoint(
                checkpoint, ids, cache, scalars, rules, args.device, draws=args.draws, seed=seed,
                covariance_mode="matched_marginals")
            for attribute in runner.TARGET_COLUMNS:
                mean_delta = matched[attribute].mean(axis=0)-physical[attribute].mean(axis=0)
                variance_delta = matched[attribute].var(axis=0)-physical[attribute].var(axis=0)
                marginal_records.append({"model": checkpoint.stem, "attribute": attribute,
                                         "mean_abs_change_of_site_mean": float(np.mean(np.abs(mean_delta))),
                                         "max_abs_change_of_site_mean": float(np.max(np.abs(mean_delta))),
                                         "mean_abs_change_of_site_variance": float(np.mean(np.abs(variance_delta)))})
    for checkpoint in sorted((args.run / "ckpt").glob("coarse_ridge_seed*.pt")):
        replay, _ = comparison.predict_baseline_checkpoint(
            checkpoint, labels, draws=args.draws, rules=rules)
        replay = replay.set_index("sample_id").loc[ids].reset_index()
        frame = replay[["sample_id", "group_id"]].copy()
        for name, rule in rules.items():
            event = rule.evaluate(truth).astype(int)
            probability = replay[f"p_{name}"].to_numpy()
            frame[f"event_{name}"] = event
            frame[f"p_{name}"] = probability
            brier, nll = comparison.brier_and_nll(probability, event)
            records.append({"model": checkpoint.stem, "rule": name, "rows": len(ids),
                            "unsafe_total": int(event.sum()), "brier": brier, "nll": nll})
        frames[checkpoint.stem] = frame
        frame.to_csv(args.outdir / f"{checkpoint.stem}.csv", index=False)
    intervals = {}
    for seed in (42, 43, 44):
        for first, second in (("diagonal", "point"), ("joint", "diagonal"),
                              ("joint", "coarse_ridge")):
            a, b = (frames[f"{model}_seed{seed}"] for model in (first, second))
            for name in rules:
                event = a[f"event_{name}"].to_numpy()
                delta = ((a[f"p_{name}"].to_numpy()-event)**2 -
                         (b[f"p_{name}"].to_numpy()-event)**2)
                intervals[f"seed{seed}/{first}_vs_{second}/{name}"] = comparison.paired_group_bootstrap(
                    delta, a["group_id"].to_numpy(), draws=2000)
    (args.outdir / "paired_brier.json").write_text(json.dumps(intervals, indent=2) + "\n")
    pd.DataFrame(records).to_csv(args.outdir / "scores.csv", index=False)
    pd.DataFrame(marginal_records).to_csv(args.outdir / "physical_marginal_changes.csv", index=False)
    print(json.dumps({"rule_count": len(rules), "model_count": len(records)//len(rules),
                      "rule_sha256": runner.file_sha256(rule_path),
                      "status": manifest["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
