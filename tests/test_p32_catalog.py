"""Catalog-only taxonomy controls; no image, label, Val or language-model assets."""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunarrisk.benchmarks.moonstone import AGE_NAMES, age_target
from lunarrisk.lunarjev.catalog import (
    OFFICIAL_SYMBOLS, SOURCE_HASHES, TIER, _tokens, encode_catalog, load_catalog, permute_encoded,
)

CATALOG = ROOT / "protocols/p32_candidate_catalog.json"


class TestP32Catalog(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog(CATALOG)

    def test_official_source_order_and_all_54_candidates(self):
        source = ROOT / "reports/evidence/p30_20261103/geologic_units.json"
        mapping = json.loads(source.read_text())
        rows = self.catalog["tasks"]["geology"]["candidates"]
        self.assertEqual(len(rows), 49)
        self.assertEqual(tuple(row["symbol"] for row in rows), OFFICIAL_SYMBOLS)
        self.assertEqual([row["class_id"] for row in rows], list(range(49)))
        self.assertEqual({row["symbol"]: row["official_unit_id"] for row in rows}, mapping)
        ages = self.catalog["tasks"]["age"]["candidates"]
        self.assertEqual(tuple(row["symbol"] for row in ages), AGE_NAMES)
        self.assertEqual([row["class_id"] for row in ages], list(range(5)))
        for row in rows:
            self.assertEqual(row["age_id"], age_target(row["symbol"]))
            self.assertIn(AGE_NAMES[row["age_id"]], row["description"])
            self.assertEqual(row["public_name"], row["symbol"])

    def test_source_bytes_and_per_candidate_provenance(self):
        for source_id, source in self.catalog["sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(),
                             SOURCE_HASHES[source_id])
            self.assertTrue(source["url"].startswith("https://"))
            self.assertTrue(source["revision"])
        for task in ("geology", "age"):
            for row in self.catalog["tasks"][task]["candidates"]:
                self.assertEqual(row["task"], task)
                self.assertEqual(row["status"], TIER)
                self.assertEqual(set(row["source_hash"]), set(row["source"]))
                for key in row["source"]:
                    self.assertEqual(row["source_hash"][key], SOURCE_HASHES[key])

    def test_case_sensitive_prefix_priority_and_unknown_handling(self):
        self.assertEqual([age_target(value) for value in ("pNc", "Nc", "EIp", "Ic", "Ec", "Cc")],
                         [0, 1, 2, 2, 3, 4])
        self.assertEqual([age_target(value) for value in ("pnc", "nc", "eIp", "ic", "cc", "?", " Cc")],
                         [-1] * 7)
        # A prefix match alone cannot admit a non-catalog geology symbol.
        for symbol in ("ic", "IC", "Cunknown", "?"):
            invalid = deepcopy(self.catalog)
            invalid["tasks"]["geology"]["candidates"][12]["symbol"] = symbol
            with self.subTest(symbol=symbol), self.assertRaises(ValueError):
                encode_catalog(invalid)

    def test_literal_symbol_case_survives_lowercase_word_tokenization(self):
        left, right = _tokens("Ic", "geology", "Ic"), _tokens("IC", "geology", "IC")
        self.assertEqual(left[0], right[0])
        self.assertNotEqual(left[-1], right[-1])
        self.assertEqual(left[-1], "literal:geology:Ic")
        encoded = encode_catalog(self.catalog)
        for task in ("geology", "age"):
            table = self.catalog["tasks"][task]["candidates"]
            matrix = encoded["candidate_features"][task]
            columns = [encoded["metadata"]["vocab"].index(f"literal:{task}:{row['symbol']}")
                       for row in table]
            selected = matrix[:, columns]
            self.assertTrue((np.diag(selected) > 0).all())
            self.assertEqual(np.count_nonzero(selected), len(table))
            self.assertEqual(np.unique(matrix, axis=0).shape[0], len(table))
        self.assertTrue(any("identity-like" in item for item in encoded["metadata"]["limitations"]))

    def test_finite_reproducible_normalized_shared_dimension(self):
        original = deepcopy(self.catalog)
        first, second = encode_catalog(self.catalog), encode_catalog(self.catalog)
        self.assertEqual(self.catalog, original)
        dimension = len(first["metadata"]["vocab"])
        self.assertEqual(first["metadata"], second["metadata"])
        self.assertEqual(first["metadata"]["document_count"], 56)
        self.assertTrue(np.isfinite(first["metadata"]["idf"]).all())
        for task, count in (("geology", 49), ("age", 5)):
            features = first["candidate_features"][task]
            vector = first["task_features"][task]
            self.assertEqual(features.shape, (count, dimension))
            self.assertEqual(vector.shape, (dimension,))
            self.assertEqual(features.dtype, np.float32)
            self.assertEqual(vector.dtype, np.float32)
            self.assertTrue(np.isfinite(features).all())
            self.assertTrue(np.isfinite(vector).all())
            np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=6)
            np.testing.assert_array_equal(features, second["candidate_features"][task])
        self.assertFalse(np.array_equal(first["task_features"]["age"], first["task_features"]["geology"]))
        self.assertEqual(first["geology_to_age"], [age_target(symbol) for symbol in OFFICIAL_SYMBOLS])
        json.dumps(first["metadata"], allow_nan=False)

    def test_idf_counts_only_public_documents(self):
        encoded = encode_catalog(self.catalog)
        vocab, idf = encoded["metadata"]["vocab"], encoded["metadata"]["idf"]
        # Each literal occurs in exactly one of 56 documents.
        expected = np.log(57 / 2) + 1
        self.assertAlmostEqual(idf[vocab.index("literal:geology:Cc")], expected)
        # All 54 candidates and both task definitions use Moonstone.
        self.assertEqual(idf[vocab.index("word:moonstone")], 1.0)

    def test_no_sample_labels_or_validation_inputs(self):
        self.assertEqual(list(inspect.signature(encode_catalog).parameters), ["catalog"])
        for key in ("sample_labels", "sample_id", "val_errors", "targets", "split"):
            invalid = deepcopy(self.catalog)
            invalid[key] = ["Cc", "pNc"]
            with self.subTest(key=key), self.assertRaises(ValueError):
                encode_catalog(invalid)
        invalid = deepcopy(self.catalog)
        invalid["tasks"]["geology"]["candidates"][0]["description"] += " Sample truth: Cc."
        with self.assertRaises(ValueError):
            encode_catalog(invalid)
        # Encoding an already loaded table must not read any files.
        with patch.object(Path, "open", side_effect=AssertionError("unexpected asset/label I/O")):
            encode_catalog(self.catalog)

    def test_catalog_validation_rejects_reordered_missing_or_relabelled_rows(self):
        changes = [
            lambda c: c["tasks"]["geology"]["candidates"].reverse(),
            lambda c: c["tasks"]["age"]["candidates"].pop(),
            lambda c: c["tasks"]["geology"]["candidates"][0].update(age_id=0),
            lambda c: c["tasks"]["age"]["candidates"][0].update(class_id=1),
            lambda c: c.update(shared_for_all_samples=False),
            lambda c: c["sources"]["geologic_units"].update(sha256="0" * 64),
            lambda c: c["descriptor_control"].update(prompt_search=True),
            lambda c: c["representation"].update(lowercase_words=False),
        ]
        for change in changes:
            invalid = deepcopy(self.catalog)
            change(invalid)
            with self.assertRaises(ValueError):
                encode_catalog(invalid)

    def test_load_checks_audited_source_hash(self):
        with patch("lunarrisk.lunarjev.catalog.SOURCE_HASHES", {**SOURCE_HASHES, "age_mapping": "0" * 64}):
            invalid = deepcopy(self.catalog)
            invalid["sources"]["age_mapping"]["sha256"] = "0" * 64
            for task in ("geology", "age"):
                for row in invalid["tasks"][task]["candidates"]:
                    row["source_hash"]["age_mapping"] = "0" * 64
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "catalog.json"
                path.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, "audited source SHA256 mismatch"):
                    load_catalog(path)

    def test_permutation_moves_all_descriptor_information_together(self):
        aligned = encode_catalog(self.catalog)
        control = permute_encoded(aligned)
        repeated = permute_encoded(aligned)
        self.assertEqual(control["metadata"]["permutation_seed"], 20261104)
        self.assertEqual(control["metadata"]["descriptor_alignment"], "fixed_permutation")
        moved_rows = 0
        for task, count in (("geology", 49), ("age", 5)):
            permutation = control["metadata"]["permutation"][task]
            self.assertEqual(sorted(permutation), list(range(count)))
            self.assertNotEqual(permutation, list(range(count)))
            self.assertEqual(permutation, repeated["metadata"]["permutation"][task])
            for destination, source in enumerate(permutation):
                np.testing.assert_array_equal(control["candidate_features"][task][destination],
                                              aligned["candidate_features"][task][source])
                self.assertEqual(control["metadata"]["descriptors"][task][destination],
                                 aligned["metadata"]["descriptors"][task][source])
                moved_rows += 1
            np.testing.assert_array_equal(control["task_features"][task], aligned["task_features"][task])
            self.assertFalse(np.shares_memory(control["candidate_features"][task], aligned["candidate_features"][task]))
        self.assertEqual(moved_rows, 54)
        self.assertEqual(control["geology_to_age"], aligned["geology_to_age"])
        for key in ("idf", "vocab", "source_hashes", "catalog_sha256"):
            self.assertEqual(control["metadata"][key], aligned["metadata"][key])
        self.assertEqual(aligned["metadata"]["descriptor_alignment"], "aligned")
        json.dumps(control["metadata"], allow_nan=False)

    def test_permutation_is_deterministic_does_not_touch_global_rng_or_input(self):
        aligned = encode_catalog(self.catalog)
        np.random.seed(123)
        expected = np.random.random(3)
        np.random.seed(123)
        permuted = permute_encoded(aligned)
        np.testing.assert_array_equal(np.random.random(3), expected)
        # Deep-copy metadata and task vectors too.
        permuted["metadata"]["descriptors"]["age"][0]["description"] = "changed"
        permuted["task_features"]["age"][:] = 0
        self.assertEqual(aligned["metadata"], encode_catalog(self.catalog)["metadata"])
        self.assertTrue(np.linalg.norm(aligned["task_features"]["age"]) > 0)
        different = permute_encoded(aligned, seed=1)
        self.assertNotEqual(different["metadata"]["permutation"], permute_encoded(aligned)["metadata"]["permutation"])
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                permute_encoded(aligned, seed)
        with self.assertRaises(ValueError):
            permute_encoded(permute_encoded(aligned))

    def test_assets_and_scientific_claims_are_explicit(self):
        inventory = self.catalog["asset_inventory"]
        self.assertEqual(inventory["local_text_encoder"]["status"], "LOCAL_PRESENT_NOT_VERIFIED_NOT_USED")
        self.assertFalse(inventory["local_text_encoder"]["weights_full_hash_verified"])
        self.assertFalse(self.catalog["representation"]["pretrained_language_model"])
        text = CATALOG.read_text()
        self.assertNotIn("/home/", text)
        self.assertNotIn("/data/", text)
        self.assertTrue(any("not complete geology definitions" in item for item in self.catalog["limitations"]))


if __name__ == "__main__":
    unittest.main()
