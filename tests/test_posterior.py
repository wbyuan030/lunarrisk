#!/usr/bin/env python3
"""Tests for the joint posterior: transforms, support, covariance and a CPU smoke run (P2.1 A1)."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lunarrisk.probabilistic import (PARAM_COUNT_FIXED_MASK, condition_posterior,  # noqa: E402
                                     export_state, from_internal, log_det_jacobian,
                                     nll_internal, posterior_from_state, to_internal)


class TestTransforms(unittest.TestCase):
    def test_round_trip(self):
        g10 = np.array([0.0, 0.01, 0.3, 0.99, 1.0])
        g15 = np.array([0.0, 0.001, 0.1, 0.5, 1.0])
        rough = np.array([0.005, 0.02, 0.1, 0.3, 1e-4])      # strictly positive by contract
        relief = np.array([2.9, 30.0, 120.0, 300.0, 1e-3])
        z = to_internal(g10, g15, rough, relief)
        back = from_internal(z)
        self.assertTrue(np.allclose(back["g10"], np.clip(g10, 1e-6, 1 - 1e-6), atol=1e-6))
        self.assertTrue(np.allclose(back["roughness_9m"], rough, atol=1e-6))
        self.assertTrue(np.allclose(back["relief_p95_p5"], relief, atol=1e-6))

    def test_support_holds_for_any_z(self):
        rng = np.random.default_rng(0)
        z = {f"z{i}": rng.normal(scale=8.0, size=5000) for i in range(1, 5)}
        back = from_internal(z)
        self.assertTrue(np.all(back["g15"] <= back["g10"] + 1e-12))
        self.assertTrue(np.all(back["g10"] <= 1.0) and np.all(back["g10"] >= 0.0))
        self.assertTrue(np.all(back["roughness_9m"] >= 0.0))
        self.assertTrue(np.all(back["relief_p95_p5"] >= 0.0))

    def test_g15_equals_g10_when_ratio_saturates(self):
        z = {"z1": np.array([0.0]), "z2": np.array([50.0]),
             "z3": np.array([0.0]), "z4": np.array([0.0])}
        back = from_internal(z)
        self.assertLess(abs(back["g15"][0] - back["g10"][0]), 1e-6)

    def test_non_positive_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            to_internal(np.array([0.1]), np.array([0.01]), np.array([0.0]), np.array([10.0]))
        with self.assertRaises(ValueError):
            to_internal(np.array([0.1]), np.array([0.01]), np.array([0.02]), np.array([0.0]))

    def test_jacobian_matches_finite_differences(self):
        rng = np.random.default_rng(1)
        z = {f"z{i}": rng.normal(size=8) for i in range(1, 5)}
        analytic = log_det_jacobian(z)
        step = 1e-6
        numeric = np.zeros(8)
        for key in ("z1", "z2", "z3", "z4"):
            shifted = {k: v.copy() for k, v in z.items()}
            shifted[key] = shifted[key] + step
            plus = from_internal(shifted)
            minus = from_internal(z)
            if key == "z1":
                derivative = (plus["g10"] - minus["g10"]) / step
                numeric += np.log(np.abs(derivative))
            elif key == "z2":
                # d g15 / d z2 holding z1 (and hence g10) fixed = g10 * s(z2)(1 - s(z2))
                derivative = (plus["g15"] - minus["g15"]) / step
                numeric += np.log(np.abs(derivative))
            elif key == "z3":
                derivative = (plus["roughness_9m"] - minus["roughness_9m"]) / step
                numeric += np.log(np.abs(derivative))
            else:
                derivative = (plus["relief_p95_p5"] - minus["relief_p95_p5"]) / step
                numeric += np.log(np.abs(derivative))
        self.assertTrue(np.allclose(analytic, numeric, atol=5e-2),
                        f"analytic {analytic[:3]} numeric {numeric[:3]}")


class TestVariants(unittest.TestCase):
    def test_point_variant_matches_the_p18_parameter_count(self):
        model = condition_posterior("point")
        self.assertEqual(sum(p.numel() for p in model.parameters()), PARAM_COUNT_FIXED_MASK)

    def test_encoder_is_identical_across_variants(self):
        counts = {v: sum(p.numel() for p in condition_posterior(v).encoder.parameters())
                  for v in ("point", "diagonal", "joint")}
        self.assertEqual(len(set(counts.values())), 1)

    def test_heads_have_the_expected_output_counts(self):
        x = torch.zeros(2, 3, 32, 32)
        scalars = torch.zeros(2, 7)
        for variant, expected in (("point", 4), ("diagonal", 8), ("joint", 16)):
            out = condition_posterior(variant)(x, scalars)
            self.assertEqual(out.shape, (2, expected), variant)

    def test_joint_with_zero_u_equals_diagonal(self):
        torch.manual_seed(0)
        diagonal = condition_posterior("diagonal", width=16, n_scalars=7)
        joint = condition_posterior("joint", width=16, rank=2, n_scalars=7)
        joint.encoder.load_state_dict(diagonal.encoder.state_dict())
        with torch.no_grad():
            joint.head.weight.zero_()
            joint.head.bias.zero_()
            joint.head.weight[:8] = diagonal.head.weight
            joint.head.bias[:8] = diagonal.head.bias
        x = torch.randn(4, 3, 32, 32)
        scalars = torch.randn(4, 7)
        z = torch.randn(4, 4)
        d_nll = diagonal.nll(diagonal(x, scalars), z)
        j_nll = joint.nll(joint(x, scalars), z)
        for d_part, j_part in zip(d_nll, j_nll):
            self.assertTrue(torch.allclose(d_part, j_part, atol=1e-5))


class TestCovariance(unittest.TestCase):
    def test_covariance_is_symmetric_positive_definite(self):
        model = condition_posterior("joint", rank=2)
        out = model(torch.randn(6, 3, 32, 32), torch.randn(6, 7))
        cov = model.covariance(model.parameters_dict(out))
        self.assertTrue(torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6))
        torch.linalg.cholesky(cov)                       # raises if not PD

    def test_sigma_is_clamped_away_from_zero(self):
        model = condition_posterior("diagonal")
        out = torch.full((1, 8), -50.0)
        params = model.parameters_dict(out)
        self.assertTrue(torch.all(params["sigma"] > 0))

    def test_point_variant_has_no_covariance(self):
        model = condition_posterior("point")
        with self.assertRaises(ValueError):
            model.covariance(model.parameters_dict(torch.zeros(1, 4)))


class TestSmoke(unittest.TestCase):
    def test_cpu_training_step_reduces_nll(self):
        torch.manual_seed(0)
        model = condition_posterior("diagonal", width=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        x = torch.randn(32, 3, 32, 32)
        scalars = torch.randn(32, 7)
        z = torch.randn(32, 4) * 0.5
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            internal, _ = nll_internal(model, x, z, scalars)
            loss = internal.mean()
            loss.backward()
            self.assertIsNotNone(model.head.weight.grad)
            optimizer.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(math.isfinite(losses[-1]))

    def test_joint_variant_backpropagates_through_cholesky(self):
        model = condition_posterior("joint", width=16, rank=2)
        x = torch.randn(8, 3, 32, 32)
        internal, physical = nll_internal(model, x, torch.randn(8, 4), torch.randn(8, 7))
        internal.mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_physical_nll_differs_by_the_jacobian(self):
        model = condition_posterior("diagonal", width=16)
        x = torch.randn(4, 3, 32, 32)
        internal, physical = nll_internal(model, x, torch.full((4, 4), -1.0), torch.randn(4, 7))
        self.assertFalse(torch.allclose(internal, physical))


class TestExport(unittest.TestCase):
    def test_state_round_trip_preserves_outputs(self):
        torch.manual_seed(3)
        model = condition_posterior("joint", width=16, rank=2)
        payload = export_state(model, seeds=[42, 43, 44])
        self.assertEqual(len(payload["weights_sha256"]), 64)
        restored = posterior_from_state(payload)
        x = torch.randn(2, 3, 32, 32)
        scalars = torch.randn(2, 7)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x, scalars), restored(x, scalars)))
        self.assertEqual(payload["seeds"], [42, 43, 44])

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            condition_posterior("full_covariance")


if __name__ == "__main__":
    unittest.main()
