"""The fixed diagnostic rule inventory must never depend on Cal/Val/Test targets."""
import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("diagnose_p22_rules", ROOT / "scripts" / "diagnose_p22_rules.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class TestTrainRules(unittest.TestCase):
    def test_heldout_values_do_not_change_rules(self):
        labels = pd.DataFrame({"split": ["train"]*4+["val", "cal", "test"]})
        for column in diagnostic.runner.TARGET_COLUMNS.values():
            labels[column] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        first = {k: v.to_dict() for k, v in diagnostic.train_rules(labels).items()}
        labels.loc[labels.split != "train", list(diagnostic.runner.TARGET_COLUMNS.values())] = 999
        second = {k: v.to_dict() for k, v in diagnostic.train_rules(labels).items()}
        self.assertEqual(first, second)
        self.assertEqual(len(first), 18)
        relief_rules = [v for v in first.values() if "relief_p95_p5" in str(v)]
        self.assertTrue(relief_rules)
        self.assertEqual(first["g10_train_q50"]["threshold"], 0.25)

    def test_missing_train_is_refused(self):
        with self.assertRaises(ValueError):
            diagnostic.train_rules(pd.DataFrame({"split": ["test"]}))


if __name__ == "__main__":
    unittest.main()
