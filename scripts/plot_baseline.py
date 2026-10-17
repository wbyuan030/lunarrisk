#!/usr/bin/env python3
"""Figures for the P1 baseline: label distribution, predictions, coverage curve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RULE = "g10_gt_0.05"
MODELS = [
    ("constant", "constant_g10", "#888888"),
    ("coarse_ridge", "coarse_g10", "#1f77b4"),
    ("cnn_nac", "cnn_nac_g10", "#d62728"),
    ("cnn_coarse_grid", "cnn_coarse_grid_g10", "#2ca02c"),
    ("cnn_fusion", "cnn_fusion_g10", "#9467bd"),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/baseline_alltiles"))
    parser.add_argument("--out", type=Path, default=Path("outputs/baseline/figures.png"))
    args = parser.parse_args(argv)

    metrics = json.loads((args.baseline / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.baseline / "test_predictions.csv")
    ridge = pd.read_csv(args.reference / "test_predictions.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. label distribution with the pre-registered threshold
    ax = axes[0, 0]
    ax.hist(predictions["y_g10"], bins=25, range=(0, 1), color="#4c72b0")
    ax.axvline(metrics["rules"][RULE]["threshold"], color="crimson", linestyle="--",
               label=f"rule threshold {metrics['rules'][RULE]['threshold']}")
    ax.set_title(f"test cohort g(10 deg)  n={len(predictions)}  "
                 f"unsafe={predictions_y_unsafe(predictions):.0%}")
    ax.set_xlabel("g(10 deg) area fraction")
    ax.set_ylabel("tiles")
    ax.legend(fontsize=8)

    # 2. predicted vs true
    ax = axes[0, 1]
    for name, column, color in MODELS:
        ax.scatter(predictions["y_g10"], predictions[column], s=14, alpha=0.6,
                   color=color, label=f"{name} (MAE {mae_of(metrics, name):.3f})")
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, linestyle=":")
    ax.set_xlabel("true g(10 deg)")
    ax.set_ylabel("predicted g(10 deg)")
    ax.set_title("test predictions (shared 110-tile set, 25 groups)")
    ax.legend(fontsize=7)

    # 3. coverage vs UAR/HMR for the primary rule
    ax = axes[1, 0]
    for name, _, color in MODELS:
        rows = metrics["models"][name]["test"][RULE]["risk_coverage"]
        ax.plot([r["coverage_actual"] for r in rows], [r["UAR"] for r in rows],
                marker="o", color=color, label=f"{name} UAR")
    ax.axhline(predictions_y_unsafe(predictions), color="black", linestyle=":",
               label="unsafe rate (no screening)")
    ax.set_xlabel("coverage (accepted fraction)")
    ax.set_ylabel("UAR")
    ax.set_title("UAR at fixed coverage (lower is better)")
    ax.legend(fontsize=7)

    # 4. per-group MAE for every model
    ax = axes[1, 1]
    grouped = predictions.groupby("group_id")
    order = [name for name, _, _ in MODELS]
    columns = {name: column for name, column, _ in MODELS}
    values = [grouped.apply(lambda d, c=columns[name]: float(np.abs(d["y_g10"] - d[c]).mean()),
                            include_groups=False).to_numpy()
              for name in order]
    ax.boxplot(values, labels=order, vert=True)
    ax.set_ylabel("per-group MAE")
    ax.set_title("per-group MAE (each dot = one leakage group)")
    for index, group_values in enumerate(values, start=1):
        jitter = np.random.default_rng(index).normal(0, 0.05, len(group_values))
        ax.scatter(np.full(len(group_values), index) + jitter, group_values,
                   s=10, color="black", alpha=0.5)
    ax.tick_params(axis="x", labelrotation=20)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


def predictions_y_unsafe(predictions: pd.DataFrame) -> float:
    return float((predictions["y_g10"] > 0.05).mean())


def mae_of(metrics: dict, name: str) -> float:
    return float(metrics["models"][name]["test"][RULE]["metrics"]["mae"])


if __name__ == "__main__":
    raise SystemExit(main())
