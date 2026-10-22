#!/usr/bin/env python3
"""Tests for the deterministic rule AST (P2.1 A2)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lunarrisk.rules import (And, Compare, Not, Or, evaluate, from_dict,  # noqa: E402
                             prob_event, standard_rules, to_dict, validate)


def bundle(**overrides):
    base = {
        "g10": np.array([0.01, 0.2, 0.6]),
        "g15": np.array([0.0, 0.1, 0.1]),
        "roughness_9m": np.array([0.001, 0.06, 0.2]),
        "relief_p95_p5": np.array([3.0, 40.0, 120.0]),
    }
    base.update(overrides)
    return base


class TestCompare(unittest.TestCase):
    def test_threshold_operators(self):
        attributes = bundle()
        self.assertEqual(evaluate(Compare("g10", 0.05), attributes).tolist(),
                         [False, True, True])
        self.assertEqual(evaluate(Compare("g10", 0.2, "ge"), attributes).tolist(),
                         [False, True, True])
        self.assertEqual(evaluate(Compare("g10", 0.2, "lt"), attributes).tolist(),
                         [True, False, False])
        self.assertEqual(evaluate(Compare("g10", 0.2, "le"), attributes).tolist(),
                         [True, True, False])

    def test_unknown_attribute_is_rejected(self):
        with self.assertRaises(ValueError):
            Compare("dtm3m_roughness_9m_m", 0.1)

    def test_non_finite_threshold_is_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Compare("g10", bad)

    def test_missing_attribute_is_rejected_at_evaluation(self):
        with self.assertRaises(KeyError):
            evaluate(Compare("roughness_9m", 0.1), {"g10": np.array([0.1])})


class TestComposite(unittest.TestCase):
    def test_and_or_not(self):
        attributes = bundle()
        both = And((Compare("g10", 0.1), Compare("roughness_9m", 0.05)))
        self.assertEqual(evaluate(both, attributes).tolist(), [False, True, True])
        either = Or((Compare("g10", 0.5), Compare("roughness_9m", 0.15)))
        self.assertEqual(evaluate(either, attributes).tolist(), [False, False, True])
        negated = Not(Compare("g10", 0.05))
        self.assertEqual(evaluate(negated, attributes).tolist(), [True, False, False])

    def test_de_morgan(self):
        attributes = bundle()
        left = Not(And((Compare("g10", 0.1), Compare("roughness_9m", 0.05))))
        right = Or((Not(Compare("g10", 0.1)), Not(Compare("roughness_9m", 0.05))))
        self.assertTrue(np.array_equal(evaluate(left, attributes), evaluate(right, attributes)))

    def test_single_child_composite_is_rejected(self):
        with self.assertRaises(ValueError):
            And((Compare("g10", 0.1),))
        with self.assertRaises(ValueError):
            Or((Compare("g10", 0.1),))


class TestSerialisation(unittest.TestCase):
    def test_round_trip(self):
        inner = Or((Compare("g15", 0.03), Not(Compare("relief_p95_p5", 5.0))))
        rule = And((Compare("g10", 0.10), inner))
        spec = to_dict(rule)
        self.assertEqual(to_dict(from_dict(json.loads(json.dumps(spec)))), spec)

    def test_unknown_node_type_rejected(self):
        with self.assertRaises(ValueError):
            from_dict({"type": "xor", "children": []})

    def test_extra_keys_rejected(self):
        with self.assertRaises(ValueError):
            from_dict({"type": "compare", "attribute": "g10", "threshold": 0.1, "units": "m"})

    def test_validate_returns_canonical_spec(self):
        self.assertEqual(validate({"type": "compare", "attribute": "g10", "threshold": 0.05}),
                         {"type": "compare", "attribute": "g10", "op": "gt", "threshold": 0.05})


class TestProbEvent(unittest.TestCase):
    def test_probability_is_a_fraction_not_a_classification(self):
        draws = {"g10": np.array([0.01, 0.02, 0.09, 0.2]), "g15": np.zeros(4),
                 "roughness_9m": np.full(4, 0.01), "relief_p95_p5": np.full(4, 10.0)}
        self.assertAlmostEqual(prob_event(draws, Compare("g10", 0.05)), 0.5)

    def test_constant_rules(self):
        draws = {"g10": np.full((3, 100), 0.5), "g15": np.full((3, 100), 0.1),
                 "roughness_9m": np.zeros((3, 100)), "relief_p95_p5": np.zeros((3, 100))}
        self.assertTrue(np.allclose(prob_event(draws, Compare("g10", 0.0), axis=-1), 1.0))
        self.assertTrue(np.allclose(prob_event(draws, Compare("g10", 1.0), axis=-1), 0.0))

    def test_mutually_exclusive_events_sum_to_one(self):
        draws = {"g10": np.linspace(0, 1, 101), "g15": np.zeros(101),
                 "roughness_9m": np.zeros(101), "relief_p95_p5": np.zeros(101)}
        above = prob_event(draws, Compare("g10", 0.5, "gt"))
        below = prob_event(draws, Compare("g10", 0.5, "le"))
        self.assertAlmostEqual(above + below, 1.0)

    def test_deterministic_across_calls(self):
        draws = {"g10": np.random.default_rng(0).random(1000), "g15": np.zeros(1000),
                 "roughness_9m": np.zeros(1000), "relief_p95_p5": np.zeros(1000)}
        first = prob_event(draws, Compare("g10", 0.5))
        second = prob_event(draws, Compare("g10", 0.5))
        self.assertEqual(first, second)


class TestStandardRules(unittest.TestCase):
    def test_preregistered_rules_round_trip(self):
        for name, rule in standard_rules().items():
            self.assertEqual(to_dict(from_dict(to_dict(rule))), to_dict(rule), name)

    def test_standard_rules_evaluate_on_a_bundle(self):
        attributes = bundle()
        for name, rule in standard_rules().items():
            result = evaluate(rule, attributes)
            self.assertEqual(result.shape, (3,), name)


if __name__ == "__main__":
    import unittest as _u
    _u.main()
