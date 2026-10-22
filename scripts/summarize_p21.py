"""P2.1 A3: aggregate the seed runs into paired comparisons.

Reads the ``p21_*.json`` reports produced by ``scripts/train_posterior_p21.py`` and prints:

* per-variant mean and across-seed range for each metric (the range IS the honest error bar at this
  sample size - three seeds cannot support a confidence interval),
* paired per-seed differences between variants on the same seed, so training noise cancels,
* the counts the risk numbers depend on (rows, groups, accepted, unsafe accepted).

Nothing here tests significance, and the third block of the report says so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = ("point", "diagonal", "joint")
ATTRIBUTES = ("g10", "g15", "roughness_9m", "relief_p95_p5")
RULES = ("g10_gt_0.05", "g15_gt_0.02")


def load_runs(directory: Path):
    runs = {}
    for path in sorted(directory.glob("*_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[(payload["variant"], payload["seed"])] = payload
    return runs


def collect(runs, split: str, block: str):
    """{metric: {variant: [values per seed]}} for one split and one block of the report."""
    out = {}
    for (variant, seed), payload in runs.items():
        metrics = payload.get("splits", {}).get(split)
        if not metrics:
            continue
        if block == "attributes":
            pairs = metrics["attributes"]
            for attribute, values in pairs.items():
                out.setdefault(f"{attribute}_mae", {}).setdefault(variant, []).append(values["mae"])
                out.setdefault(f"{attribute}_rmse", {}).setdefault(variant, []).append(values["rmse"])
        else:
            for rule, values in metrics["events"].items():
                out.setdefault(f"{rule}_brier", {}).setdefault(variant, []).append(values["brier"])
                out.setdefault(f"{rule}_nll", {}).setdefault(variant, []).append(values["nll"])
                for key in ("uar@10", "uar@20", "uar@40", "hmr@10", "hmr@20", "hmr@40"):
                    if values.get(key) is not None:
                        out.setdefault(f"{rule}_{key}", {}).setdefault(variant, []).append(values[key])
    for metric in out:
        for variant in out[metric]:
            out[metric][variant].sort()
    return out


def summarise(metric_values):
    lines = []
    for metric, per_variant in metric_values.items():
        cells = []
        for variant in VARIANTS:
            values = per_variant.get(variant)
            if not values:
                cells.append("-")
            else:
                cells.append(f"{np.mean(values):.4f} [{min(values):.4f}, {max(values):.4f}]")
        lines.append((metric, cells))
    return lines


def paired(runs, split: str, block: str, variant_a: str, variant_b: str):
    """Per-seed differences (b - a) on the same seed, so training noise cancels."""
    seeds = sorted({seed for (variant, seed) in runs if variant in (variant_a, variant_b)})
    out = {}
    for seed in seeds:
        a = runs.get((variant_a, seed))
        b = runs.get((variant_b, seed))
        if not a or not b:
            continue
        for metric in metric_names(block):
            try:
                va = metric_value(a, split, block, metric)
                vb = metric_value(b, split, block, metric)
            except KeyError:
                continue
            out.setdefault(metric, []).append(vb - va)
    return out


def metric_names(block):
    if block == "attributes":
        return [f"{a}_mae" for a in ATTRIBUTES] + [f"{a}_rmse" for a in ATTRIBUTES]
    names = []
    for rule in RULES:
        names += [f"{rule}_brier", f"{rule}_nll", f"{rule}_uar@20", f"{rule}_hmr@20"]
    return names


def metric_value(payload, split, block, metric):
    metrics = payload["splits"][split]
    if block == "attributes":
        attribute, kind = metric.rsplit("_", 1)
        return metrics["attributes"][attribute][kind]
    rule, kind = metric.rsplit("_", 1)
    return metrics["events"][rule][kind]


def counts(runs, split: str):
    rows = groups = accepted = unsafe = None
    for payload in runs.values():
        metrics = payload.get("splits", {}).get(split)
        if not metrics:
            continue
        rows = metrics["n_rows"]
        groups = metrics["n_groups"]
        entry = metrics["events"]["g10_gt_0.05"]
        accepted = entry["accepted@20"]
        unsafe = entry["unsafe_accepted@20"]
    return {"rows": rows, "groups": groups, "accepted@20": accepted,
            "unsafe_accepted@20": unsafe}


def render(runs) -> str:
    lines = ["# P2.1 A3 — 开发集上的第一轮探索性实验（自动汇总）", "",
             "**开发集为 `development_test`（120 Tile / 25 组），已被反复查看。以下全部为探索性结果。**",
             ""]
    for split in ("val", "test"):
        lines += [f"## {split} 属性误差（均值 [最小值, 最大值]，3 seeds）", "",
                  "| 指标 | point | diagonal | joint |", "|---|---|---|---|"]
        for metric, cells in summarise(collect(runs, split, "attributes")):
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines += ["", f"## {split} 事件指标", "",
                  "| 指标 | point | diagonal | joint |", "|---|---|---|---|"]
        for metric, cells in summarise(collect(runs, split, "events")):
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 同 seed 配对差（joint − diagonal，负值表示 joint 更好）", "",
              "| 指标 | 每个 seed 的差 | 均值 |", "|---|---|---|"]
    for metric, values in paired(runs, "test", "events", "diagonal", "joint").items():
        formatted = " ".join(f"{v:+.4f}" for v in values)
        lines.append(f"| {metric} | {formatted} | {np.mean(values):+.4f} |")
    lines += ["", "## 同 seed 配对差（diagonal − point）", "",
              "| 指标 | 每个 seed 的差 | 均值 |", "|---|---|---|"]
    for metric, values in paired(runs, "test", "attributes", "point", "diagonal").items():
        formatted = " ".join(f"{v:+.4f}" for v in values)
        lines.append(f"| {metric} | {formatted} | {np.mean(values):+.4f} |")
    stats = counts(runs, "test")
    lines += ["", "## 风险数字的实际计数（test，@20% 覆盖率）", "",
              f"- 样本 {stats['rows']} 行，独立组 {stats['groups']}，放行 {stats['accepted@20']}，"
              f"其中危险 {stats['unsafe_accepted@20']}。", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--md", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit(f"no runs under {args.dir}")
    text = render(runs)

    summary = {"runs": [{"variant": v, "seed": s} for (v, s) in sorted(runs)],
               "test_attributes": {m: {v: values for v, values in per.items()}
                                   for m, per in collect(runs, "test", "attributes").items()},
               "test_events": {m: {v: values for v, values in per.items()}
                               for m, per in collect(runs, "test", "events").items()},
               "test_counts": counts(runs, "test")}
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text, encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(text[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
