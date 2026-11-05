"""Offline contracts and descriptive metrics for the paired NASA transfer run.

No defaults, training, model imports, or Test scoring are provided here. All
returned values are strict-JSON serializable, including a genuinely infinite NLL.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

_CLASSES = 49
_CHOICES = {
    "adaptation": ("frozen", "input_affine"),
    "precision": ("fp32",),
    "optimizer": ("AdamW",),
    "schedule": ("constant",),
    "selection": ("val_macro_f1_full_earliest_tie",),
}
_FIELDS = frozenset((*_CHOICES, "epochs", "patience", "batch_size", "lr",
                     "weight_decay", "grad_clip", "seeds", "test_scored"))


def _integer(value, field, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{field} must be in [{low}, {high}]")
    return int(value)


def _real(value, field):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{field} must be a finite real number")
    return result


def validate_budget(budget):
    """Return a canonical copy of an explicit, complete bounded run budget."""
    if not isinstance(budget, Mapping) or set(budget) != _FIELDS:
        raise ValueError("budget must contain exactly the declared fields")
    result = {}
    result["epochs"] = _integer(budget["epochs"], "epochs", 1, 50)
    result["patience"] = _integer(budget["patience"], "patience", 1, result["epochs"])
    result["batch_size"] = _integer(budget["batch_size"], "batch_size", 1, 16)
    for field in ("lr", "weight_decay", "grad_clip"):
        result[field] = _real(budget[field], field)
    if not 0 < result["lr"] <= .01:
        raise ValueError("lr must be in (0, .01]")
    if not 0 <= result["weight_decay"] <= 1:
        raise ValueError("weight_decay must be in [0, 1]")
    if result["grad_clip"] <= 0:
        raise ValueError("grad_clip must be positive")
    seeds = budget["seeds"]
    if not isinstance(seeds, list):
        raise ValueError("seeds must be a list")
    seeds = [_integer(seed, "seed", 42, 44) for seed in seeds]
    if seeds not in ([42], [42, 43, 44]):
        raise ValueError("seeds must be [42] or [42, 43, 44], in that order")
    result["seeds"] = seeds
    for field, choices in _CHOICES.items():
        if not isinstance(budget[field], str) or budget[field] not in choices:
            raise ValueError(f"unsupported {field}")
        result[field] = budget[field]
    if budget["test_scored"] is not False:
        raise ValueError("test_scored must be false; Test scoring is held")
    result["test_scored"] = False
    return result


def probability_metrics(probabilities, targets):
    """Score nonempty Nx49 probabilities without clipping or renormalization.

    Rows must sum to one within 1e-6 absolute tolerance (FP32 roundoff).
    Argmax ties select the lowest class index. Confusion rows are true classes,
    columns predicted classes. Brier is the mean per-row *sum* across 49 classes.
    Absent classes have F1 zero and remain in the full49 macro denominator.
    """
    try:
        raw = np.asarray(probabilities)
        y = np.asarray(targets)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities and targets must be rectangular arrays") from exc
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1] != _CLASSES or not raw.shape[0]:
        raise ValueError("probabilities must be a nonempty real numeric Nx49 array")
    p = raw.astype(np.float64)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-6):
        raise ValueError("probability rows must sum to one")
    if y.ndim != 1 or len(y) != len(p) or y.dtype.kind not in "iu":
        raise ValueError("targets must be a matching one-dimensional integer array")
    if np.any(y < 0) or np.any(y >= _CLASSES):
        raise ValueError("targets must be in [0, 48]")
    y = y.astype(np.int64)
    prediction = p.argmax(axis=1)
    confusion = np.bincount(y * _CLASSES + prediction, minlength=_CLASSES ** 2).reshape(_CLASSES, _CLASSES)
    support = confusion.sum(axis=1)
    denominator = support + confusion.sum(axis=0)
    f1 = np.divide(2. * np.diag(confusion), denominator,
                   out=np.zeros(_CLASSES), where=denominator > 0)
    true_probability = p[np.arange(len(p)), y]
    infinite = bool(np.any(true_probability == 0))
    residual = p.copy()
    residual[np.arange(len(p)), y] -= 1.
    return {"rows": len(p), "macro_f1_full": float(f1.mean()),
            "accuracy": float(np.mean(prediction == y)),
            "confusion": confusion.tolist(), "support": support.tolist(),
            "per_class_f1": f1.tolist(),
            "nll": None if infinite else float(-np.log(true_probability).mean()),
            "nll_is_infinite": infinite, "multiclass_brier_sum": float(np.square(residual).sum(axis=1).mean())}


def paired_summary(reports):
    """Summarize pretrained-minus-random Val full49 F1, not an inference test.

    Every supplied seed must have exactly one report from each arm. The seed
    cohort must be one of the allowed complete budgets, preventing partial
    three-seed results from silently masquerading as a completed comparison.
    """
    if not isinstance(reports, list) or not reports:
        raise ValueError("reports must be a nonempty list")
    pairs = {}
    for report in reports:
        if not isinstance(report, Mapping):
            raise ValueError("each report must be a mapping")
        seed = _integer(report.get("seed"), "seed", 42, 44)
        arm = report.get("arm")
        if not isinstance(arm, str) or arm not in ("random", "pretrained"):
            raise ValueError("arm must be random or pretrained")
        try:
            score = _real(report["metrics"]["val"]["macro_f1_full"], "macro_f1_full")
        except (KeyError, TypeError) as exc:
            raise ValueError("report requires metrics.val.macro_f1_full") from exc
        if not 0 <= score <= 1:
            raise ValueError("macro_f1_full must be in [0, 1]")
        pair = pairs.setdefault(seed, {})
        if arm in pair:
            raise ValueError(f"duplicate {arm} report for seed {seed}")
        pair[arm] = score
    if sorted(pairs) not in ([42], [42, 43, 44]):
        raise ValueError("report seeds must be [42] or [42, 43, 44]")
    per_seed = []
    for seed, pair in sorted(pairs.items()):
        if set(pair) != {"random", "pretrained"}:
            raise ValueError(f"seed {seed} is not paired")
        per_seed.append({"seed": seed, **pair, "delta": pair["pretrained"] - pair["random"]})
    deltas = np.array([row["delta"] for row in per_seed])
    return {"n_seeds": len(per_seed), "per_seed": per_seed,
            "delta_mean": float(deltas.mean()),
            "delta_sample_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
            "scope": "Exploratory reused Val only; no confidence interval or generalization claim."}
