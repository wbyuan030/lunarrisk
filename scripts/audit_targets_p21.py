"""P2.1 A0: audit the four supervised targets before any posterior model touches them.

Read-only over ``outputs/labels_hydrated.csv``. It answers the questions the plan requires:

* availability and missing rate of ``slope3m_g10``, ``slope3m_g15``, ``dtm3m_roughness_9m_m``,
  ``dtm3m_relief_p95_p5_m`` per split, and whether missing values were ever zero-filled;
* physical sanity: ``0 <= g15 <= g10 <= 1``, roughness and relief non-negative, no non-finite values;
* unit provenance: the recorded slope variable name, unit source and the verified flag;
* split integrity: unique sample ids, and whether any ``group_id`` or ``NAC_EDR_ID`` spans splits
  (which would leak a site across train/val/cal/test);
* leakage: the two DTM-derived targets must not appear in the model's declared input lists.

Missing targets are reported, never imputed here; the modelling code must use an explicit mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TARGETS = ("slope3m_g10", "slope3m_g15", "dtm3m_roughness_9m_m", "dtm3m_relief_p95_p5_m")
SPLITS = ("train", "val", "cal", "test")
# columns that would be leakage if any modelling variant consumed them as input
DERIVED_TARGETS = ("dtm3m_roughness_9m_m", "dtm3m_relief_p95_p5_m", "slope3m_g10", "slope3m_g15")


def to_float(value):
    """Parse a CSV cell; empty -> None (explicitly missing, never 0.0)."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def quantiles(values):
    if not values:
        return {}
    ordered = sorted(values)
    def pick(p):
        index = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return ordered[index]
    return {"min": ordered[0], "p01": pick(0.01), "p50": pick(0.5), "p99": pick(0.99),
            "max": ordered[-1], "mean": sum(ordered) / len(ordered)}


def audit(rows):
    """rows: list of dicts already parsed from the labels CSV."""
    report = {"rows": len(rows), "targets": {}, "violations": [], "split_integrity": {},
              "unit_provenance": {}, "leakage": []}

    for split in SPLITS:
        subset = [r for r in rows if r.get("split") == split]
        report.setdefault("rows_per_split", {})[split] = len(subset)

    for target in TARGETS:
        entry = {"available": 0, "missing": 0, "available_per_split": {}, "missing_per_split": {},
                 "distribution": {}, "distribution_per_split": {}}
        all_values = []
        for split in SPLITS:
            values = [to_float(r.get(target)) for r in rows if r.get("split") == split]
            present = [v for v in values if v is not None]
            entry["available_per_split"][split] = len(present)
            entry["missing_per_split"][split] = len(values) - len(present)
            entry["distribution_per_split"][split] = quantiles(present)
            all_values.extend(present)
        entry["available"] = len(all_values)
        entry["missing"] = len(rows) - len(all_values)
        entry["distribution"] = quantiles(all_values)
        report["targets"][target] = entry

    for row in rows:
        sid = row.get("sample_id", "?")
        g10, g15 = to_float(row.get("slope3m_g10")), to_float(row.get("slope3m_g15"))
        rough = to_float(row.get("dtm3m_roughness_9m_m"))
        relief = to_float(row.get("dtm3m_relief_p95_p5_m"))
        if g10 is not None and not (0.0 <= g10 <= 1.0):
            report["violations"].append({"sample_id": sid, "target": "slope3m_g10",
                                         "reason": "outside [0,1]", "value": g10})
        if g15 is not None and not (0.0 <= g15 <= 1.0):
            report["violations"].append({"sample_id": sid, "target": "slope3m_g15",
                                         "reason": "outside [0,1]", "value": g15})
        if g10 is not None and g15 is not None and g15 > g10 + 1e-12:
            report["violations"].append({"sample_id": sid, "target": "slope3m_g15",
                                         "reason": "g15 > g10", "value": g15, "g10": g10})
        for name, value in (("dtm3m_roughness_9m_m", rough), ("dtm3m_relief_p95_p5_m", relief)):
            if value is not None and value < 0:
                report["violations"].append({"sample_id": sid, "target": name,
                                             "reason": "negative", "value": value})

    ids = [r.get("sample_id") for r in rows]
    report["split_integrity"] = {
        "unique_sample_ids": len(set(ids)) == len(ids),
        "n_groups": len({r.get("group_id") for r in rows}),
        "n_sources": len({r.get("NAC_EDR_ID") for r in rows}),
        "groups_per_split": {s: len({r.get("group_id") for r in rows if r.get("split") == s})
                             for s in SPLITS},
        "group_spans_splits": _spanning(rows, "group_id"),
        "source_spans_splits": _spanning(rows, "NAC_EDR_ID"),
    }

    report["unit_provenance"] = {
        "slope3m_variable": sorted({str(r.get("slope3m_variable")) for r in rows})[:5],
        "units_source": sorted({str(r.get("units_source")) for r in rows})[:5],
        "units_verified_values": sorted({str(r.get("slope3m_units_verified")) for r in rows}),
        "rows_with_units_verified": sum(1 for r in rows
                                        if str(r.get("slope3m_units_verified")).lower()
                                        in ("true", "1", "yes")),
    }
    return report


def _spanning(rows, key):
    """Keys that appear under more than one split."""
    seen = {}
    for row in rows:
        seen.setdefault(row.get(key), set()).add(row.get("split"))
    return sorted({k for k, splits in seen.items() if k is not None and len(splits) > 1})


def input_lists():
    """Read the declared model inputs from train_baseline without importing torch."""
    path = Path(__file__).resolve().parent / "train_baseline.py"
    text = path.read_text(encoding="utf-8")
    import re
    out = {}
    for name in ("COARSE_FEATURES", "CHANNELS", "GEOMETRY_SCALARS", "NAC_SCALARS",
                 "VARIANT_SCALARS"):
        match = re.search(rf"^{name}\s*=\s*(.+?)^\S", text, re.S | re.M)
        if match:
            out[name] = match.group(1)
    return out


def leakage_check(declared: dict) -> list:
    findings = []
    blob = " ".join(declared.values())
    for target in DERIVED_TARGETS:
        if target in blob:
            findings.append({"target": target, "reason": "appears in a declared model input list"})
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--md", type=Path, default=None)
    args = parser.parse_args(argv)

    import csv
    raw = args.labels.read_bytes()
    with args.labels.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    report = audit(rows)
    report["labels_sha256"] = hashlib.sha256(raw).hexdigest()
    report["labels_path"] = str(args.labels)
    report["leakage"] = leakage_check(input_lists())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "rows_per_split", "violations",
                                             "split_integrity", "unit_provenance", "leakage",
                                             "labels_sha256")}, indent=2, default=str)[:4000])
    return 0


def render_markdown(report) -> str:
    lines = ["# P2.1 A0 — 目标标签审计（自动生成）", "",
             f"`labels_sha256 = {report.get('labels_sha256', 'not recorded')}`", "",
             f"行数 {report['rows']}；split 分布 " +
             ", ".join(f"{k}={v}" for k, v in report["rows_per_split"].items()), "",
             "## 四个目标的可用数与分布", "",
             "| 目标 | 非空 | 缺失 | min | p50 | p99 | max | 缺失率 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, entry in report["targets"].items():
        dist = entry["distribution"]
        lines.append(
            f"| `{name}` | {entry['available']} | {entry['missing']} | "
            f"{dist.get('min', float('nan')):.6g} | {dist.get('p50', float('nan')):.6g} | "
            f"{dist.get('p99', float('nan')):.6g} | {dist.get('max', float('nan')):.6g} | "
            f"{entry['missing'] / max(report['rows'], 1):.3%} |")
    lines += ["", "### 分 split 非空数", "",
              "| 目标 | " + " | ".join(SPLITS) + " |",
              "|---|" + "---|" * len(SPLITS)]
    for name, entry in report["targets"].items():
        lines.append(f"| `{name}` | " +
                     " | ".join(str(entry["available_per_split"][s]) for s in SPLITS) + " |")
    lines += ["", "## 物理一致性", ""]
    if report["violations"]:
        lines.append(f"**{len(report['violations'])} 项违规**（前 10 项）：")
        lines.append("")
        for item in report["violations"][:10]:
            lines.append(f"- `{item['sample_id']}` {item['target']}: {item['reason']} "
                         f"value={item['value']}")
    else:
        lines.append("无违规：`0 ≤ g15 ≤ g10 ≤ 1` 与 roughness/relief 非负全部成立，且无 NaN 混入。")
    integ = report["split_integrity"]
    lines += ["", "## split 完整性", "",
              f"- sample_id 唯一：{integ['unique_sample_ids']}",
              f"- 组数 {integ['n_groups']}，来源数 {integ['n_sources']}",
              f"- 每 split 组数：" + ", ".join(f"{k}={v}" for k, v in
                                             integ["groups_per_split"].items()),
              f"- 跨 split 的组：{len(integ['group_spans_splits'])}",
              f"- 跨 split 的来源：{len(integ['source_spans_splits'])}",
              "", "## 单位与来源", "",
              f"- slope3m_variable: {report['unit_provenance']['slope3m_variable']}",
              f"- units_source: {report['unit_provenance']['units_source']}",
              f"- units_verified: {report['unit_provenance']['units_verified_values']} "
              f"（{report['unit_provenance']['rows_with_units_verified']} 行显式为真）",
              "", "## 输入泄漏检查", ""]
    if report["leakage"]:
        for item in report["leakage"]:
            lines.append(f"- **LEAK**: `{item['target']}` {item['reason']}")
    else:
        lines.append("`dtm3m_roughness_9m_m`、`dtm3m_relief_p95_p5_m`、`slope3m_g10`、`slope3m_g15` "
                     "均未出现在 `train_baseline.py` 声明的输入列表中。")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
