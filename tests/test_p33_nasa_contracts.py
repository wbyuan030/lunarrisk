"""Offline synthetic contract checks; no downloaded NASA model or real pixels."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunarrisk.nasa_transfer.contracts import (
    audit_inference_state, prepare_terrain, unwrap_state_dict, validate_domain,
)


def fixture():
    domain = {"num_channels": 1, "pretokenized": False, "input_size": 4,
              "patch_size": 2, "pre_resize": None, "scaler": "std",
              "stats": {"mean": [10.0], "std": [2.0]}}
    provenance = {"input_space": "raw_physical", "units": {"elevation": "m", "slope": "degree"},
                  "spatial_transform": "none", "domain_shift": "MOONSTONE_TERRAIN_NOT_NASA_NATIVE",
                  "meters_per_pixel": 237.0, "elevation_source": "synthetic test only",
                  "slope_source": "synthetic test only", "units_evidence": "synthetic fixture",
                  "nodata_evidence": "synthetic fixture"}
    raw = {key: np.zeros((2, 1, 4, 4), dtype=np.float32) for key in ("elevation", "slope")}
    valid = {key: np.ones_like(value, dtype=bool) for key, value in raw.items()}
    return raw, valid, {"dtm": copy.deepcopy(domain), "slope": copy.deepcopy(domain)}, provenance


def target_state():
    return {"encoder.0.weight": np.zeros((4, 4)), "encoder_norm.weight": np.ones(4),
            "encoder_embeddings.dtm.proj.weight": np.zeros((4, 4)),
            "encoder_embeddings.slope.proj.weight": np.zeros((4, 4)),
            "encoder_embeddings.dtm.pos_emb": np.zeros((1, 4, 4)),
            "register_tokens": np.zeros((1, 2, 4))}


class TestNasaTerrainContract(unittest.TestCase):
    def test_zero_is_observed_and_inputs_are_not_mutated(self):
        raw, valid, domains, provenance = fixture()
        out = prepare_terrain(raw, valid, domains, provenance)
        self.assertEqual(set(out), {"dtm", "slope"})
        np.testing.assert_array_equal(out["dtm"], -5)
        np.testing.assert_array_equal(raw["elevation"], 0)

    def test_scalers_and_per_sample_local_center(self):
        for scaler in ("std", "local_mean_std", "minmax", None):
            with self.subTest(scaler=scaler):
                raw, valid, domains, provenance = fixture()
                raw["elevation"][0] = 4
                raw["elevation"][1] = 12
                domains["dtm"]["scaler"] = scaler
                domains["dtm"]["stats"].update(min=[0], max=[20])
                out = prepare_terrain(raw, valid, domains, provenance)["dtm"]
                expected = {"std": (-3, 1), "local_mean_std": (0, 0), "minmax": (.2, .6), None: (4, 12)}[scaler]
                for row, value in enumerate(expected):
                    np.testing.assert_allclose(out[row], value)

    def test_partial_and_whole_missingness_fail_closed(self):
        for missing_all in (False, True):
            raw, valid, domains, provenance = fixture()
            valid["elevation"].reshape(-1)[:(32 if missing_all else 1)] = False
            with self.assertRaisesRegex(ValueError, "missing/nonfinite"):
                prepare_terrain(raw, valid, domains, provenance)

    def test_nonfinite_or_nondegree_slope_rejected(self):
        for value in (np.nan, np.inf, -1, 91):
            raw, valid, domains, provenance = fixture()
            raw["slope"].reshape(-1)[0] = value
            with self.assertRaises(ValueError):
                prepare_terrain(raw, valid, domains, provenance)

    def test_extra_channels_labels_or_geometry_rejected(self):
        for key in ("vis", "wac_hapke_643nm", "metadata", "dominant_unit_name", "target"):
            raw, valid, domains, provenance = fixture()
            raw[key] = np.zeros((2, 1, 4, 4))
            with self.assertRaises(ValueError):
                prepare_terrain(raw, valid, domains, provenance)

    def test_provenance_cannot_be_omitted_or_zscored(self):
        for key in fixture()[3]:
            raw, valid, domains, provenance = fixture()
            del provenance[key]
            with self.assertRaises(ValueError):
                prepare_terrain(raw, valid, domains, provenance)
        raw, valid, domains, provenance = fixture()
        provenance["input_space"] = "moonstone_zscore"
        with self.assertRaises(ValueError):
            prepare_terrain(raw, valid, domains, provenance)

    def test_shape_resize_and_mask_dtype_rejected(self):
        for kind in ("shape", "batch", "integer", "mask"):
            raw, valid, domains, provenance = fixture()
            if kind == "shape":
                raw["elevation"] = raw["elevation"][:, :, :2]
            elif kind == "batch":
                raw["elevation"] = raw["elevation"][:1]
                valid["elevation"] = valid["elevation"][:1]
            elif kind == "integer":
                raw["elevation"] = raw["elevation"].astype(np.int32)
            else:
                valid["elevation"] = valid["elevation"].astype(np.float32)
            with self.assertRaises(ValueError):
                prepare_terrain(raw, valid, domains, provenance)

    def test_invalid_domain_never_uses_guessed_defaults(self):
        for key, value in (("scaler", "unknown"), ("num_channels", 5), ("pretokenized", True),
                           ("patch_size", 3), ("pre_resize", 8), ("input_size", True)):
            domain = fixture()[2]["dtm"]
            domain[key] = value
            with self.assertRaises(ValueError):
                validate_domain(domain)
        domain = fixture()[2]["dtm"]
        del domain["scaler"]
        with self.assertRaises(ValueError):
            validate_domain(domain)
        for std in (0, -1, np.nan, np.inf):
            domain = fixture()[2]["dtm"]
            domain["stats"]["std"] = [std]
            with self.assertRaises(ValueError):
                validate_domain(domain)


class TestNasaCheckpointAudit(unittest.TestCase):
    def test_complete_selected_inference_coverage(self):
        target = target_state()
        source = dict(target, **{"decoder.0.weight": np.zeros((4, 4)),
                                 "encoder_embeddings.vis.proj.weight": np.zeros((5, 4))})
        for checkpoint in (source, {"model": source}, {"state_dict": source},
                           {"module." + key: value for key, value in source.items()}):
            report = audit_inference_state(target, checkpoint)
            self.assertEqual(report["loaded_inference_fraction"], 1.0)
            self.assertEqual(report["inference_key_count"], len(target))
            self.assertEqual(len(report["excluded_keys"]), 2)

    def test_missing_shape_mismatch_and_unexpected_encoder_are_fatal(self):
        for kind in ("missing", "shape", "extra", "unknown"):
            target = target_state()
            source = dict(target)
            if kind == "missing":
                del source["encoder_embeddings.dtm.pos_emb"]
            elif kind == "shape":
                source["encoder.0.weight"] = np.zeros((5, 4))
            elif kind == "extra":
                source["encoder.1.weight"] = np.zeros((4, 4))
            else:
                source["unrecognized.weight"] = np.zeros(1)
            with self.assertRaises(ValueError):
                audit_inference_state(target, source)

    def test_ambiguous_envelope_and_prefix_collision_are_fatal(self):
        with self.assertRaises(ValueError):
            unwrap_state_dict({"model": target_state(), "state_dict": target_state()})
        with self.assertRaises(ValueError):
            unwrap_state_dict({"a": np.zeros(1), "module.a": np.zeros(1)})
        for state in ({}, {"a": "not a tensor"}, {"state_dict": {}}):
            with self.assertRaises(ValueError):
                unwrap_state_dict(state)

    def test_incomplete_target_is_not_valid_coverage(self):
        target = target_state()
        del target["encoder_norm.weight"]
        with self.assertRaises(ValueError):
            audit_inference_state(target, target)


if __name__ == "__main__":
    unittest.main()
