#!/usr/bin/env python3
"""Offline regression tests for safe hydration and label merging (P18-01 / P18-02).

Every test uses synthetic manifests/labels and stub download or reader functions; no
network access and no writes to the real ``outputs/labels_hydrated.csv``.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hydrate_tiles", ROOT / "scripts" / "hydrate_tiles.py")
ht = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ht
SPEC.loader.exec_module(ht)

SPLITS = ("train", "cal", "val", "test")
MANIFEST_COLUMNS = ["sample_id", "split_final", "split_official", "group_id", "NAC_EDR_ID",
                    "LTM_CODE", "block_5km", "has_slope3m", "coarse_slope_path", "image_path"]


def make_manifest(per_split: dict[str, int], groups_per_split: int = 3, seed: int = 0) -> pd.DataFrame:
    """Synthetic manifest with round-robin friendly structure."""
    rng = np.random.default_rng(seed)
    rows = []
    for split, count in per_split.items():
        groups = [f"G_{split}_{i}" for i in range(groups_per_split)]
        for index in range(count):
            group = groups[index % groups_per_split]
            sample_id = f"{split}_{index:03d}"
            rows.append({
                "sample_id": sample_id,
                "split_final": split,
                "split_official": split,
                "group_id": group,
                "NAC_EDR_ID": f"EDR_{split}_{index % groups_per_split}",
                "LTM_CODE": f"LTM_{index % 4}",
                "block_5km": f"blk_{index % 5}",
                "has_slope3m": True,
                "coarse_slope_path": f"slope_60m/{sample_id}.nc",
                "image_path": f"nac/{sample_id}.nc",
            })
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=rng.integers(0, 1 << 30)).reset_index(drop=True)


def fake_record(row: pd.Series, data_dir: Path) -> dict:
    """Minimal stand-in for hydrate_one's return value."""
    return {
        "sample_id": str(row["sample_id"]),
        "split": str(row["split_final"]),
        "official_split": str(row["split_official"]),
        "group_id": str(row["group_id"]),
        "NAC_EDR_ID": str(row["NAC_EDR_ID"]),
        "LTM_CODE": str(row["LTM_CODE"]),
        "block_5km": str(row["block_5km"]),
        "new_metric": 0.5,  # deliberately a column the old header does not have
    }


class HydrationCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest_path = self.root / "manifest.parquet"
        self.labels_path = self.root / "labels.csv"
        self.data_dir = self.root / "hydrated"

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self, frame: pd.DataFrame):
        frame.to_parquet(self.manifest_path, index=False)
        return frame

    def args(self, **overrides) -> argparse.Namespace:
        values = dict(manifest=self.manifest_path, data_dir=self.data_dir, labels=self.labels_path,
                      train=6, cal=3, val=3, test=3, workers=2, seed=42,
                      units_source="test-units")
        values.update(overrides)
        return argparse.Namespace(**values)

    def write_labels(self, frame: pd.DataFrame):
        frame.to_csv(self.labels_path, index=False)

    def touch_sources(self, sample_ids, files=("nac.nc", "slope_3m.nc")):
        for sample_id in sample_ids:
            directory = self.data_dir / str(sample_id)
            directory.mkdir(parents=True, exist_ok=True)
            for name in files:
                (directory / name).write_bytes(b"stub")


class TestSelectionStability(HydrationCase):
    def test_train_quota_growth_is_prefix_stable(self):
        frame = self.write_manifest(make_manifest({"train": 90, "cal": 6, "val": 6, "test": 6}))
        small = ht.select_tiles(frame, {"train": 12, "cal": 3, "val": 3, "test": 3}, seed=42)
        large = ht.select_tiles(frame, {"train": 60, "cal": 3, "val": 3, "test": 3}, seed=42)
        train_small = set(small[small["split_final"] == "train"]["sample_id"])
        train_large = set(large[large["split_final"] == "train"]["sample_id"])
        self.assertTrue(train_small.issubset(train_large))
        self.assertGreater(len(train_large), len(train_small))
        for split in ("cal", "val", "test"):
            self.assertEqual(
                list(small[small["split_final"] == split]["sample_id"]),
                list(large[large["split_final"] == split]["sample_id"]),
                f"{split} selection must not move when only the train quota grows")

    def test_manifest_without_coarse_slope_path_is_refused(self):
        frame = make_manifest({"train": 6, "cal": 3, "val": 3, "test": 3}).drop(columns=["coarse_slope_path"])
        self.write_manifest(frame)
        with self.assertRaises(ht.HydrationError):
            ht.build_selection(self.manifest_path, {"train": 3, "cal": 2, "val": 2, "test": 2}, 42)


class TestNoOpResume(HydrationCase):
    def _prepare_complete(self):
        frame = self.write_manifest(make_manifest({"train": 6, "cal": 3, "val": 3, "test": 3}))
        args = self.args()
        selected = ht.build_selection(self.manifest_path, {"train": 6, "cal": 3, "val": 3, "test": 3}, 42)
        labels = pd.DataFrame({
            "sample_id": selected["sample_id"].astype(str),
            "split": selected["split_final"].astype(str),
            "official_split": selected["split_official"].astype(str),
            "group_id": selected["group_id"].astype(str),
            "NAC_EDR_ID": selected["NAC_EDR_ID"].astype(str),
            "LTM_CODE": selected["LTM_CODE"].astype(str),
            "block_5km": selected["block_5km"].astype(str),
            "slope3m_g10": 0.1,
            "slope3m_valid_fraction": 1.0,
        })
        self.write_labels(labels)
        self.touch_sources(labels["sample_id"])
        return frame, args, labels

    def test_nothing_to_do_exits_zero_and_never_touches_labels(self):
        _frame, args, labels = self._prepare_complete()
        before = self.labels_path.read_bytes()

        def forbidden_download(*_args, **_kwargs):
            raise AssertionError("no download may be attempted for a complete cohort")

        def forbidden_hydrate(*_args, **_kwargs):
            raise AssertionError("no hydration may be attempted for a complete cohort")

        code, stats = ht.run_hydration(args, download_fn=forbidden_download, hydrate_fn=forbidden_hydrate)
        self.assertEqual(code, 0)
        self.assertEqual(stats["todo_count"], 0)
        self.assertEqual(stats["download_attempted"], 0)
        self.assertEqual(stats["new_records"], 0)
        self.assertFalse(stats["labels_written"])
        self.assertEqual(self.labels_path.read_bytes(), before)
        self.assertFalse((self.root / "labels.csv.tmp").exists())
        summary = pd.read_json(self.labels_path.with_suffix(".summary.json"), typ="series")
        self.assertEqual(int(summary["tiles"]), len(labels))
        self.assertEqual(int(summary["selected_count"]), len(labels))


class TestAppendOnlyMerge(HydrationCase):
    def _prepare_partial(self, missing_ids):
        frame = self.write_manifest(make_manifest({"train": 6, "cal": 3, "val": 3, "test": 3}))
        args = self.args()
        selected = ht.build_selection(self.manifest_path, {"train": 6, "cal": 3, "val": 3, "test": 3}, 42)
        have = selected[~selected["sample_id"].isin(missing_ids)]
        labels = pd.DataFrame({
            "sample_id": have["sample_id"].astype(str),
            "split": have["split_final"].astype(str),
            "official_split": have["split_official"].astype(str),
            "group_id": have["group_id"].astype(str),
            "NAC_EDR_ID": have["NAC_EDR_ID"].astype(str),
            "LTM_CODE": have["LTM_CODE"].astype(str),
            "block_5km": have["block_5km"].astype(str),
            "slope3m_g10": 0.25,
            "slope3m_valid_fraction": 1.0,
        })
        self.write_labels(labels)
        self.touch_sources(labels["sample_id"])
        return frame, args, labels

    def test_only_the_missing_tile_is_requested(self):
        _frame, args, labels = self._prepare_partial({"train_005"})
        calls = []

        def spy_hydrate(row, data_dir, units_source, download_fn):
            calls.append(str(row["sample_id"]))
            self.touch_sources([str(row["sample_id"])])
            return fake_record(row, data_dir)

        code, stats = ht.run_hydration(args, hydrate_fn=spy_hydrate)
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["train_005"])
        self.assertEqual(stats["download_attempted"], 1)
        self.assertEqual(stats["new_records"], 1)
        merged = pd.read_csv(self.labels_path)
        self.assertEqual(len(merged), len(labels) + 1)
        self.assertEqual(list(merged["sample_id"][:len(labels)]), list(labels["sample_id"]))
        self.assertTrue(np.allclose(merged["slope3m_g10"][:len(labels)], 0.25))
        self.assertIn("new_metric", merged.columns)
        self.assertTrue(merged["new_metric"][:len(labels)].isna().all())

    def test_all_pending_failures_exit_nonzero_without_writing(self):
        _frame, args, _labels = self._prepare_partial({"train_003", "train_004"})
        before = self.labels_path.read_bytes()

        def failing_hydrate(*_args, **_kwargs):
            raise RuntimeError("simulated S3 failure")

        code, stats = ht.run_hydration(args, hydrate_fn=failing_hydrate)
        self.assertEqual(code, 2)
        self.assertEqual(stats["new_records"], 0)
        self.assertEqual(len(stats["failures"]), 2)
        self.assertEqual({f["sample_id"] for f in stats["failures"]}, {"train_003", "train_004"})
        self.assertFalse(stats["labels_written"])
        self.assertEqual(self.labels_path.read_bytes(), before)

    def test_partial_failure_keeps_successes_and_reports_nonzero(self):
        _frame, args, labels = self._prepare_partial({"train_001", "train_002", "train_003"})

        def flaky_hydrate(row, data_dir, units_source, download_fn):
            if str(row["sample_id"]) == "train_002":
                raise RuntimeError("simulated S3 failure")
            self.touch_sources([str(row["sample_id"])])
            return fake_record(row, data_dir)

        code, stats = ht.run_hydration(args, hydrate_fn=flaky_hydrate)
        self.assertEqual(code, 2)
        self.assertEqual(stats["new_records"], 2)
        self.assertEqual([f["sample_id"] for f in stats["failures"]], ["train_002"])
        merged = pd.read_csv(self.labels_path)
        self.assertEqual(set(merged["sample_id"]) - set(labels["sample_id"]), {"train_001", "train_003"})

    def test_record_outside_selection_is_refused(self):
        _frame, args, _labels = self._prepare_partial({"train_005", "train_004"})

        def stray_hydrate(row, data_dir, units_source, download_fn):
            record = fake_record(row, data_dir)
            record["sample_id"] = "not_in_selection"
            return record

        with self.assertRaises(ht.HydrationError):
            ht.run_hydration(args, hydrate_fn=stray_hydrate)
        self.assertFalse((self.root / "labels.csv.tmp").exists())


class TestConsistencyGuards(HydrationCase):
    def _labels_from_selection(self, selected: pd.DataFrame, **overrides) -> pd.DataFrame:
        labels = pd.DataFrame({
            "sample_id": selected["sample_id"].astype(str),
            "split": selected["split_final"].astype(str),
            "official_split": selected["split_official"].astype(str),
            "group_id": selected["group_id"].astype(str),
            "NAC_EDR_ID": selected["NAC_EDR_ID"].astype(str),
            "LTM_CODE": selected["LTM_CODE"].astype(str),
            "block_5km": selected["block_5km"].astype(str),
            "slope3m_g10": 0.1,
            "slope3m_valid_fraction": 1.0,
        })
        for key, value in overrides.items():
            labels.loc[0, key] = value
        return labels

    def test_split_drift_is_refused(self):
        self.write_manifest(make_manifest({"train": 6, "cal": 3, "val": 3, "test": 3}))
        args = self.args()
        selected = ht.build_selection(self.manifest_path, {"train": 6, "cal": 3, "val": 3, "test": 3}, 42)
        labels = self._labels_from_selection(selected, split="val")
        self.write_labels(labels)
        self.touch_sources(labels["sample_id"])
        with self.assertRaises(ht.HydrationError):
            ht.run_hydration(args, hydrate_fn=lambda *a, **k: fake_record(a[0], a[1]))

    def test_product_drift_is_refused(self):
        self.write_manifest(make_manifest({"train": 6, "cal": 3, "val": 3, "test": 3}))
        args = self.args()
        selected = ht.build_selection(self.manifest_path, {"train": 6, "cal": 3, "val": 3, "test": 3}, 42)
        labels = self._labels_from_selection(selected, NAC_EDR_ID="SOMETHING_ELSE")
        self.write_labels(labels)
        self.touch_sources(labels["sample_id"])
        with self.assertRaises(ht.HydrationError):
            ht.run_hydration(args, hydrate_fn=lambda *a, **k: fake_record(a[0], a[1]))

    def test_existing_row_outside_selection_is_refused(self):
        self.write_manifest(make_manifest({"train": 6, "cal": 3, "val": 3, "test": 3}))
        args = self.args()
        selected = ht.build_selection(self.manifest_path, {"train": 6, "cal": 3, "val": 3, "test": 3}, 42)
        labels = self._labels_from_selection(selected)
        extra = labels.iloc[[0]].copy()
        extra["sample_id"] = "ghost_tile"
        labels = pd.concat([labels, extra], ignore_index=True)
        self.write_labels(labels)
        self.touch_sources(labels["sample_id"])
        with self.assertRaises(ht.HydrationError):
            ht.run_hydration(args, hydrate_fn=lambda *a, **k: fake_record(a[0], a[1]))

    def test_missing_local_source_is_refused(self):
        self.write_manifest(make_manifest({"train": 6, "cal": 3, "val": 3, "test": 3}))
        args = self.args()
        selected = ht.build_selection(self.manifest_path, {"train": 6, "cal": 3, "val": 3, "test": 3}, 42)
        labels = self._labels_from_selection(selected)
        self.write_labels(labels)
        self.touch_sources(labels["sample_id"])
        (self.data_dir / str(labels["sample_id"].iloc[0]) / "slope_3m.nc").unlink()
        with self.assertRaises(ht.HydrationError):
            ht.run_hydration(args, hydrate_fn=lambda *a, **k: fake_record(a[0], a[1]))

    def test_duplicate_sample_id_in_labels_is_refused(self):
        self.write_manifest(make_manifest({"train": 6, "cal": 3, "val": 3, "test": 3}))
        args = self.args()
        selected = ht.build_selection(self.manifest_path, {"train": 6, "cal": 3, "val": 3, "test": 3}, 42)
        labels = self._labels_from_selection(selected)
        labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
        self.write_labels(labels)
        self.touch_sources(labels["sample_id"])
        with self.assertRaises(ht.HydrationError):
            ht.run_hydration(args, hydrate_fn=lambda *a, **k: fake_record(a[0], a[1]))


class TestMergeSchema(unittest.TestCase):
    def test_new_columns_are_appended_without_shifting_old_values(self):
        existing = pd.DataFrame({"sample_id": ["a", "b"], "split": ["train", "val"]})
        records = [{"sample_id": "c", "split": "test", "extra": 1.5}]
        merged = ht.merge_records(existing, records)
        self.assertEqual(list(merged.columns), ["sample_id", "split", "extra"])
        self.assertEqual(list(merged["split"]), ["train", "val", "test"])
        self.assertTrue(np.isnan(merged["extra"].iloc[0]))
        self.assertEqual(float(merged["extra"].iloc[2]), 1.5)

    def test_duplicate_ids_are_refused(self):
        existing = pd.DataFrame({"sample_id": ["a"], "split": ["train"]})
        with self.assertRaises(ht.HydrationError):
            ht.merge_records(existing, [{"sample_id": "a", "split": "train"}])

    def test_records_without_sample_id_are_refused(self):
        with self.assertRaises(ht.HydrationError):
            ht.merge_records(pd.DataFrame({"sample_id": ["a"]}), [{"split": "train"}])

    def test_empty_records_returns_existing_unchanged(self):
        existing = pd.DataFrame({"sample_id": ["a"], "split": ["train"]})
        merged = ht.merge_records(existing, [])
        pd.testing.assert_frame_equal(merged, existing)


class TestAtomicWriteAndEnrich(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.labels_path = self.root / "labels.csv"
        self.manifest_path = self.root / "manifest.parquet"

    def tearDown(self):
        self.tmp.cleanup()

    def test_atomic_write_leaves_no_temp_file(self):
        frame = pd.DataFrame({"sample_id": ["a"], "split": ["train"]})
        ht.atomic_write_csv(self.labels_path, frame)
        self.assertTrue(self.labels_path.exists())
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def _write_inputs(self, sample_ids=("a", "b")):
        labels = pd.DataFrame({"sample_id": list(sample_ids), "split": ["train", "val"],
                               "slope3m_g10": [0.1, 0.2]})
        labels.to_csv(self.labels_path, index=False)
        manifest = pd.DataFrame({
            "sample_id": list(sample_ids),
            "incidence_angle_deg": [30.0, 40.0],
            "emission_angle_deg": [5.0, 6.0],
            "phase_angle_deg": [25.0, 34.0],
            "sub_solar_ground_azimuth_deg": [10.0, 20.0],
        })
        manifest.to_parquet(self.manifest_path, index=False)
        return labels

    def test_enrich_is_one_to_one_and_keeps_rows(self):
        labels = self._write_inputs()
        for sample_id in labels["sample_id"]:
            directory = self.root / "hydrated" / sample_id
            directory.mkdir(parents=True)
            (directory / "dtm_3m.nc").write_bytes(b"stub")
        code = ht.enrich_existing(self.labels_path, self.manifest_path, self.root / "hydrated",
                                  read_elevation_fn=lambda path: np.arange(16.0).reshape(4, 4))
        self.assertEqual(code, 0)
        enriched = pd.read_csv(self.labels_path)
        self.assertEqual(list(enriched["sample_id"]), list(labels["sample_id"]))
        self.assertTrue(np.allclose(enriched["slope3m_g10"], labels["slope3m_g10"]))
        for column in ("incidence_angle_deg", "dtm3m_roughness_9m_m", "dtm3m_relief_p95_p5_m"):
            self.assertIn(column, enriched.columns)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_enrich_failure_leaves_labels_untouched(self):
        self._write_inputs()
        before = self.labels_path.read_bytes()

        def exploding_reader(_path):
            raise RuntimeError("simulated corrupt DTM")

        for sample_id in ("a", "b"):
            directory = self.root / "hydrated" / sample_id
            directory.mkdir(parents=True)
            (directory / "dtm_3m.nc").write_bytes(b"stub")
        with self.assertRaises(RuntimeError):
            ht.enrich_existing(self.labels_path, self.manifest_path, self.root / "hydrated",
                               read_elevation_fn=exploding_reader)
        self.assertEqual(self.labels_path.read_bytes(), before)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_enrich_rejects_duplicate_manifest_ids(self):
        self._write_inputs()
        manifest = pd.DataFrame({
            "sample_id": ["a", "a"],
            "incidence_angle_deg": [30.0, 31.0],
            "emission_angle_deg": [5.0, 6.0],
            "phase_angle_deg": [25.0, 26.0],
            "sub_solar_ground_azimuth_deg": [10.0, 11.0],
        })
        manifest.to_parquet(self.manifest_path, index=False)
        with self.assertRaises(ht.HydrationError):
            ht.enrich_existing(self.labels_path, self.manifest_path, self.root / "hydrated",
                               read_elevation_fn=lambda path: np.zeros((4, 4)))


if __name__ == "__main__":
    unittest.main()
