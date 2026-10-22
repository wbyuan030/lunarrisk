"""Physical-support transforms for the four terrain attributes (P2.1 A1).

The two slope-exceedance fractions are ordered by construction: ``g15 <= g10`` always, because a
steeper threshold can only select a subset. Modelling them as two independent variables would waste
capacity and produce impossible draws, so they go through a chain that cannot leave the support::

    g10 = sigmoid(z1)
    g15 = g10 * sigmoid(z2)          # ensures 0 <= g15 <= g10 <= 1 for any finite z

Roughness and relief are positive lengths in metres, so a plain ``log`` maps them to the whole line
and its inverse ``exp`` can never return a negative length. ``log1p`` was tried first and rejected:
``expm1`` of a negative ``z`` lands in (-1, 0), i.e. a negative roughness, and a Gaussian on ``z`` does
produce negative values. A tile with an exact zero would need an explicit offset instead of quietly
turning ``log(0)`` into ``-inf``, so :func:`to_internal` refuses non-positive lengths.

``log_det_jacobian`` returns the exact log|det J| of the map above, so a Gaussian fitted on ``z`` can
report a density in *physical* units. The internal-space NLL and the physical-space NLL differ by
this term and the two must never be quoted interchangeably.
"""
from __future__ import annotations

import numpy as np

ATTRIBUTES = ("g10", "g15", "roughness_9m", "relief_p95_p5")
DEFAULT_EPS = 1e-6


def _clip(probability, eps):
    return np.clip(probability, eps, 1.0 - eps)


def to_internal(g10, g15, roughness, relief, eps: float = DEFAULT_EPS):
    """Map physical attributes to the unconstrained chain used by the posterior."""
    g10 = np.asarray(g10, dtype=np.float64)
    g15 = np.asarray(g15, dtype=np.float64)
    roughness = np.asarray(roughness, dtype=np.float64)
    relief = np.asarray(relief, dtype=np.float64)

    p10 = _clip(g10, eps)
    # ratio must stay inside (eps, 1): g15 <= g10 by construction, but guard the data anyway
    ratio = np.clip(g15 / np.maximum(g10, eps), eps, 1.0 - eps)
    z1 = np.log(p10 / (1.0 - p10))
    z2 = np.log(ratio / (1.0 - ratio))
    if np.any(roughness <= 0) or np.any(relief <= 0):
        raise ValueError("roughness and relief must be strictly positive; add an explicit offset")
    z3 = np.log(roughness)
    z4 = np.log(relief)
    return {"z1": z1, "z2": z2, "z3": z3, "z4": z4}


def from_internal(z, eps: float = DEFAULT_EPS):
    """Inverse of :func:`to_internal`; the returned attributes always satisfy physical support."""
    z1 = np.asarray(z["z1"], dtype=np.float64)
    z2 = np.asarray(z["z2"], dtype=np.float64)
    p10 = 1.0 / (1.0 + np.exp(-z1))
    ratio = 1.0 / (1.0 + np.exp(-z2))
    g10 = np.clip(p10, eps, 1.0 - eps)
    g15 = g10 * np.clip(ratio, eps, 1.0 - eps)
    return {"g10": g10, "g15": g15,
            "roughness_9m": np.exp(z["z3"]),
            "relief_p95_p5": np.exp(z["z4"])}


def log_det_jacobian(z, eps: float = DEFAULT_EPS):
    """log|det J| of z -> physical attributes, elementwise (summed over attributes).

    Chain rule: g10 = sigmoid(z1) gives dg10/dz1 = g10 (1 - g10); with g15 = g10 * s(z2) the pair
    contributes ``log[g10 (1-g10)] + log[s(z2)(1-s(z2))] + log g10``; the two log1p transforms give
    ``z3`` and ``z4`` terms (d expm1(z)/dz = exp(z)).
    """
    z1 = np.asarray(z["z1"], dtype=np.float64)
    z2 = np.asarray(z["z2"], dtype=np.float64)
    z3 = np.asarray(z["z3"], dtype=np.float64)
    z4 = np.asarray(z["z4"], dtype=np.float64)
    p10 = np.clip(1.0 / (1.0 + np.exp(-z1)), eps, 1.0 - eps)
    ratio = np.clip(1.0 / (1.0 + np.exp(-z2)), eps, 1.0 - eps)
    return (np.log(p10) + np.log(1.0 - p10)          # dg10/dz1
            + np.log(p10)                             # g15 is proportional to g10
            + np.log(ratio) + np.log(1.0 - ratio)     # s(z2) factor
            + z3 + z4)                                # d expm1(z)/dz = exp(z)
