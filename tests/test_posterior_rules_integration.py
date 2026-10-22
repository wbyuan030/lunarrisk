#!/usr/bin/env python3
"""End-to-end check: posterior draws -> physical attributes -> rule probability (P2.1 A1+A2)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lunarrisk.probabilistic import condition_posterior, from_internal, sample_internal  # noqa: E402
from lunarrisk.rules import Compare, Not, Or, prob_event, standard_rules  # noqa: E402


def sample_attributes(model, x, scalars, draws=256, seed=0):
    """Draw the internal chain from the model covariance, then map to physical attributes."""
    z = sample_internal(model, x, scalars, draws=draws, seed=seed)
    z_np = {f"z{i + 1}": z[..., i].numpy() for i in range(4)}
    return from_internal(z_np)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = condition_posterior("joint", width=16, rank=2, n_scalars=7)
        self.x = torch.randn(1, 3, 32, 32)
        self.scalars = torch.randn(1, 7)

    def test_draws_respect_physical_support(self):
        attributes = sample_attributes(self.model, self.x, self.scalars, draws=128)
        self.assertTrue(np.all(attributes["g15"] <= attributes["g10"] + 1e-12))
        self.assertTrue(np.all(attributes["g10"] >= 0.0) and np.all(attributes["g10"] <= 1.0))
        self.assertTrue(np.all(attributes["roughness_9m"] > 0.0))
        self.assertTrue(np.all(attributes["relief_p95_p5"] > 0.0))

    def test_rule_probabilities_are_fractions(self):
        attributes = sample_attributes(self.model, self.x, self.scalars, draws=128)
        for name, rule in standard_rules().items():
            probability = prob_event(attributes, rule, axis=0)
            self.assertGreaterEqual(float(probability.min()), 0.0, name)
            self.assertLessEqual(float(probability.max()), 1.0, name)

    def test_complementary_events_are_complementary(self):
        attributes = sample_attributes(self.model, self.x, self.scalars, draws=128)
        above = prob_event(attributes, Compare("g10", 0.05, "gt"), axis=0)
        below = prob_event(attributes, Not(Compare("g10", 0.05, "gt")), axis=0)
        self.assertTrue(np.allclose(above + below, 1.0))

    def test_monotone_threshold_probabilities(self):
        attributes = sample_attributes(self.model, self.x, self.scalars, draws=256)
        loose = prob_event(attributes, Compare("g10", 0.02, "gt"), axis=0)
        strict = prob_event(attributes, Compare("g10", 0.20, "gt"), axis=0)
        self.assertGreaterEqual(float(loose), float(strict))

    def test_or_event_exceeds_each_part(self):
        attributes = sample_attributes(self.model, self.x, self.scalars, draws=256)
        a = Compare("g10", 0.10, "gt")
        b = Compare("g15", 0.03, "gt")
        prob_a = float(prob_event(attributes, a, axis=0))
        prob_b = float(prob_event(attributes, b, axis=0))
        prob_or = float(prob_event(attributes, Or((a, b)), axis=0))
        self.assertGreaterEqual(prob_or + 1e-12, max(prob_a, prob_b))

    def test_sampling_is_reproducible_with_a_seed(self):
        first = sample_attributes(self.model, self.x, self.scalars, draws=32, seed=7)
        second = sample_attributes(self.model, self.x, self.scalars, draws=32, seed=7)
        self.assertTrue(np.array_equal(first["g10"], second["g10"]))


if __name__ == "__main__":
    unittest.main()
