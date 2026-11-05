"""Synthetic-only, NumPy-only tests of the NASA paired experiment contract."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunarrisk.nasa_transfer.experiment import (
    paired_summary, probability_metrics, validate_budget,
)


def budget():
    return dict(epochs=50, patience=8, batch_size=16, lr=.001,
                weight_decay=.01, grad_clip=1., seeds=[42, 43, 44],
                adaptation="frozen", precision="fp32", optimizer="AdamW",
                schedule="constant", selection="val_macro_f1_full_earliest_tie",
                test_scored=False)


def report(seed, arm, score):
    return dict(seed=seed, arm=arm, metrics={"val": {"macro_f1_full": score}})


class BudgetTests(unittest.TestCase):
    def test_canonical_copy_and_strict_json(self):
        source = budget()
        got = validate_budget(source)
        self.assertEqual(got, source)
        self.assertIsNot(got["seeds"], source["seeds"])
        json.dumps(got, allow_nan=False)
        source.update(epochs=np.int64(1), patience=1, batch_size=1,
                      lr=.01, weight_decay=0, seeds=[42], adaptation="input_affine")
        self.assertEqual(validate_budget(source)["epochs"], 1)
        source["weight_decay"] = 1
        self.assertEqual(validate_budget(source)["weight_decay"], 1.)

    def test_exact_required_fields(self):
        for key in budget():
            with self.subTest(key=key):
                data = budget()
                del data[key]
                with self.assertRaises(ValueError):
                    validate_budget(data)
        for value in (None, [], {**budget(), "extra": 1}):
            with self.assertRaises(ValueError):
                validate_budget(value)

    def test_bad_bounds_types_and_choices(self):
        invalid = {"epochs": [0, 51, 1., True],
                   "patience": [0, 51, True, 1.],
                   "batch_size": [0, 17, True, 1.],
                   "lr": [0, -.01, .011, True, "0.01", float("nan"), float("inf")],
                   "weight_decay": [-.1, 1.1, False, float("nan")],
                   "grad_clip": [0, -1, True, float("inf")],
                   "seeds": [[], [43], [42, 43], [42, 44, 43], [42, 42, 44],
                             [42., 43, 44], (42,), [True]],
                   "adaptation": ["full", None, []], "precision": ["fp16"],
                   "optimizer": ["SGD"], "schedule": ["cosine"],
                   "selection": ["test_accuracy"], "test_scored": [True, 0, None]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    data = budget()
                    data[key] = value
                    with self.assertRaises(ValueError):
                        validate_budget(data)
        with self.assertRaises(ValueError):
            validate_budget({**budget(), "epochs": 2, "patience": 3})


class ProbabilityTests(unittest.TestCase):
    def test_perfect_full49(self):
        result = probability_metrics(np.eye(49), np.arange(49))
        self.assertEqual(result["rows"], 49)
        self.assertEqual(result["macro_f1_full"], 1.)
        self.assertEqual(result["accuracy"], 1.)
        self.assertEqual(result["support"], [1] * 49)
        self.assertEqual(result["nll"], 0.)
        self.assertFalse(result["nll_is_infinite"])
        self.assertEqual(result["multiclass_brier_sum"], 0.)

    def test_absent_classes_stay_in_denominator(self):
        result = probability_metrics(np.eye(49)[[0, 0, 1]], [0, 1, 1])
        self.assertAlmostEqual(result["macro_f1_full"], (4 / 3) / 49)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["support"][:3], [1, 2, 0])
        self.assertEqual(result["confusion"][1][:3], [1, 1, 0])
        self.assertTrue(result["nll_is_infinite"])
        self.assertIsNone(result["nll"])
        self.assertAlmostEqual(result["multiclass_brier_sum"], 2 / 3)
        json.dumps(result, allow_nan=False)

    def test_uniform_and_argmax_tie(self):
        p = np.full((2, 49), 1 / 49, dtype=np.float32)
        before = p.copy()
        result = probability_metrics(p, [0, 48])
        np.testing.assert_array_equal(p, before)
        self.assertEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["nll"], np.log(49), places=6)
        self.assertAlmostEqual(result["multiclass_brier_sum"], 48 / 49, places=6)

    def test_subnormal_probability_not_clipped(self):
        p = np.zeros((1, 49))
        p[0, :2] = [np.nextafter(0., 1.), 1.]
        result = probability_metrics(p, [0])
        self.assertFalse(result["nll_is_infinite"])
        self.assertGreater(result["nll"], 700)

    def test_invalid_probabilities(self):
        valid = np.full((1, 49), 1 / 49)
        bad = [[], np.zeros((0, 49)), np.ones((1, 48)), valid[0],
               np.zeros((1, 49)), valid * .9, valid.astype(str),
               valid.astype(complex), valid.astype(bool), [[1], [1, 2]]]
        for value in (float("nan"), float("inf"), -.1, 1.1):
            p = valid.copy()
            p[0, 0] = value
            bad.append(p)
        for p in bad:
            with self.subTest(shape=np.shape(p) if not isinstance(p, list) else "list"):
                with self.assertRaises(ValueError):
                    probability_metrics(p, [0])

    def test_invalid_targets(self):
        p = np.eye(49)[[0]]
        for y in ([], [49], [-1], [0.], [True], ["0"], [[0]], [0, 1], [float("nan")]):
            with self.subTest(y=y):
                with self.assertRaises(ValueError):
                    probability_metrics(p, y)


class PairedTests(unittest.TestCase):
    def test_single_pair_has_no_std(self):
        result = paired_summary([report(42, "pretrained", .3), report(42, "random", .2)])
        self.assertEqual(result["n_seeds"], 1)
        self.assertAlmostEqual(result["delta_mean"], .1)
        self.assertIsNone(result["delta_sample_std"])
        json.dumps(result, allow_nan=False)

    def test_three_pairs_sorted_sample_std(self):
        reports = []
        for seed, score in ((44, .6), (42, .2), (43, .4)):
            reports.extend([report(seed, "pretrained", score), report(seed, "random", .2)])
        original = copy.deepcopy(reports)
        result = paired_summary(reports)
        self.assertEqual(reports, original)
        self.assertEqual([row["seed"] for row in result["per_seed"]], [42, 43, 44])
        self.assertAlmostEqual(result["delta_mean"], .2)
        self.assertAlmostEqual(result["delta_sample_std"], .2)
        self.assertIn("no confidence interval", result["scope"])

    def test_bad_reports(self):
        a, b = report(42, "random", .2), report(42, "pretrained", .3)
        bad = [[], None, [None], [a], [a, b, a], [a, b, report(43, "random", .2)],
               [report(43, "random", .2), report(43, "pretrained", .2)],
               [report(True, "random", .2)], [report(42., "random", .2)],
               [report(42, "other", .2)], [{"seed": 42, "arm": "random"}],
               [{"seed": 42, "arm": "random", "metrics": None}]]
        for score in (float("nan"), float("inf"), -.1, 1.1, True, ".3", None):
            bad.append([a, report(42, "pretrained", score)])
        for reports in bad:
            with self.subTest(reports=reports):
                with self.assertRaises(ValueError):
                    paired_summary(reports)


if __name__ == "__main__":
    unittest.main()
