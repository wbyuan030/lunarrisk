"""Deterministic rule AST and event-probability integration (P2.1 A2).

Rules are plain data (JSON round-trippable), evaluated by one function, and the *same* evaluator is
used both for ground-truth boolean events and for posterior draws. That is the point: a rule must not
be written as ``g10 > 0.05`` in one place and re-implemented slightly differently in another.

Only the nodes the plan allows exist: ``Compare``, ``And``, ``Or``, ``Not``. Anything else, a
non-finite threshold, an unknown attribute or a wrong unit is rejected at construction time, so a
malformed rule can never silently evaluate to something plausible.

``prob_event`` integrates a posterior by Monte Carlo over supplied draws and returns the event
probability; it never thresholds the draws and then reports a single classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

ATTRIBUTE_UNITS = {
    "g10": "fraction",
    "g15": "fraction",
    "roughness_9m": "m",
    "relief_p95_p5": "m",
    "slope60m_mean": "degree",
    "slope60m_q90": "degree",
    "incidence": "degree",
    "emission": "degree",
}
OPERATORS = ("gt", "ge", "lt", "le")
NODE_TYPES = ("compare", "and", "or", "not")


def _apply(op: str, values, threshold):
    if op == "gt":
        return values > threshold
    if op == "ge":
        return values >= threshold
    if op == "lt":
        return values < threshold
    if op == "le":
        return values <= threshold
    raise ValueError(f"unknown operator {op!r}")


@dataclass(frozen=True)
class Compare:
    attribute: str
    threshold: float
    op: str = "gt"

    def __post_init__(self):
        if self.attribute not in ATTRIBUTE_UNITS:
            raise ValueError(f"attribute {self.attribute!r} is not whitelisted")
        if self.op not in OPERATORS:
            raise ValueError(f"operator {self.op!r} not in {OPERATORS}")
        value = float(self.threshold)
        if not np.isfinite(value):
            raise ValueError(f"threshold for {self.attribute} must be finite, got {self.threshold}")

    def evaluate(self, attributes: Mapping[str, np.ndarray]):
        if self.attribute not in attributes:
            raise KeyError(f"attribute {self.attribute!r} missing from the evaluation bundle")
        return _apply(self.op, np.asarray(attributes[self.attribute]), float(self.threshold))

    def to_dict(self) -> dict:
        return {"type": "compare", "attribute": self.attribute,
                "op": self.op, "threshold": float(self.threshold)}


@dataclass(frozen=True)
class And:
    children: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if len(self.children) < 2:
            raise ValueError("and needs at least two children")
        object.__setattr__(self, "children", tuple(self.children))

    def evaluate(self, attributes):
        result = self.children[0].evaluate(attributes)
        for child in self.children[1:]:
            result = np.logical_and(result, child.evaluate(attributes))
        return result

    def to_dict(self) -> dict:
        return {"type": "and", "children": [c.to_dict() for c in self.children]}


@dataclass(frozen=True)
class Or:
    children: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if len(self.children) < 2:
            raise ValueError("or needs at least two children")
        object.__setattr__(self, "children", tuple(self.children))

    def evaluate(self, attributes):
        result = self.children[0].evaluate(attributes)
        for child in self.children[1:]:
            result = np.logical_or(result, child.evaluate(attributes))
        return result

    def to_dict(self) -> dict:
        return {"type": "or", "children": [c.to_dict() for c in self.children]}


@dataclass(frozen=True)
class Not:
    child: Any

    def evaluate(self, attributes):
        return np.logical_not(self.child.evaluate(attributes))

    def to_dict(self) -> dict:
        return {"type": "not", "child": self.child.to_dict()}


BUILDERS = {"compare": Compare, "and": And, "or": Or, "not": Not}


def from_dict(spec: dict):
    """Rebuild a rule, rejecting unknown types, stray keys and malformed thresholds."""
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError("rule must be a dict with a 'type'")
    kind = spec["type"]
    if kind not in NODE_TYPES:
        raise ValueError(f"unknown node type {kind!r}")
    allowed = {"compare": {"type", "attribute", "op", "threshold"},
               "and": {"type", "children"}, "or": {"type", "children"}, "not": {"type", "child"}}[kind]
    extra = set(spec) - allowed
    if extra:
        raise ValueError(f"unexpected keys {sorted(extra)} for node {kind}")
    if kind == "compare":
        return Compare(spec["attribute"], spec["threshold"], spec.get("op", "gt"))
    if kind in ("and", "or"):
        children = tuple(from_dict(c) for c in spec["children"])
        return And(children) if kind == "and" else Or(children)
    return Not(from_dict(spec["child"]))


def to_dict(rule) -> dict:
    return rule.to_dict()


def validate(spec: dict) -> dict:
    """Round-trip a spec and return it, raising on any structural problem."""
    return from_dict(spec).to_dict()


def evaluate(rule, attributes: Mapping[str, np.ndarray]):
    """Boolean event array; the single evaluator used for truth and for draws."""
    return rule.evaluate(attributes)


def prob_event(draws: Mapping[str, np.ndarray], rule, *, axis: int | None = None):
    """P(rule) estimated from posterior draws; returns a probability, not a classification.

    ``draws`` hold arrays of shape (..., S) or (S,) depending on whether several sites are integrated
    at once. ``axis`` may be given to average along a specific draw axis.
    """
    result = np.asarray(rule.evaluate(draws), dtype=np.float64)
    return result.mean(axis=axis)


def standard_rules() -> dict:
    """The preregistered first set; thresholds are the original fixed proxies."""
    return {
        "g10_gt_0.05": Compare("g10", 0.05, "gt"),
        "g15_gt_0.02": Compare("g15", 0.02, "gt"),
        "g10_gt_0.10_or_g15_gt_0.03": Or((Compare("g10", 0.10, "gt"),
                                          Compare("g15", 0.03, "gt"))),
        "g10_gt_0.10_and_rough_gt_0.05": And((Compare("g10", 0.10, "gt"),
                                             Compare("roughness_9m", 0.05, "gt"))),
        "not_both_safe": Not(And((Compare("g10", 0.05, "le"),
                                  Compare("roughness_9m", 0.02, "le")))),
    }
