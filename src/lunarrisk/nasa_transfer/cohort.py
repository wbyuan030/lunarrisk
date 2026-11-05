"""Authenticate Train membership against the already frozen P31/P32 cohort.

The trust roots are repository-pinned historical artifacts, not hashes supplied
alongside an arbitrary batch. This checks IDs/targets, not raw-pixel authenticity.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

TRAIN_PLAN_SHA256 = "a5f57da8ded63e8ba1aa43dd883564947ac3f2c70efe800601500d228d7933c4"
METADATA_ARCHIVE_SHA256 = "1c971c5d6c8ea245ba34dfdf5b929917c77a366713645bb9c2a3e53ae74efbf0"
TRAIN_PLAN_PATH = "protocols/p31_local_prefix_eval_plan.json"
METADATA_ARCHIVE_PATH = "reports/evidence/p30_20261103/metadata_rows.csv.gz"


def _read_pinned(path, expected_sha256, name):
    body = Path(path).read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError(f"authoritative {name} SHA256 mismatch")
    return body


def load_authoritative_train(plan_path, metadata_path):
    """Return the fixed Train ID→Geology target mapping and provenance receipt."""
    plan = json.loads(_read_pinned(plan_path, TRAIN_PLAN_SHA256, "Train plan"))
    body = _read_pinned(metadata_path, METADATA_ARCHIVE_SHA256, "metadata archive")
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
        records = list(csv.DictReader(io.TextIOWrapper(compressed, encoding="utf-8")))
    if plan["protocol"] != "P31_LOCAL_PREFIX_STORED_DERIVED_V1" or plan["status"] != "LOCKED_BEFORE_MODEL_SCORING":
        raise ValueError("unexpected frozen Train protocol")
    train = plan["splits"]["train"]
    indices = train["indices"]
    if len(indices) != 5135 or train["rows"] != len(indices) or len(set(indices)) != len(indices):
        raise ValueError("frozen Train cardinality mismatch")
    complete = {entry["index"]: entry["sample_id"] for entry in plan["complete_chunks"]}
    targets = {}
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
            raise ValueError("invalid Train metadata index")
        row = records[index]
        sample_id = row["sample_id"]
        target = int(row["unit_id"]) - 1
        if (row["stored_split"] != "0" or sample_id in targets or not sample_id
                or not 0 <= target < 49 or complete.get(index) != sample_id):
            raise ValueError("wrong split, duplicate ID, invalid target or absent chunk in Train manifest")
        targets[sample_id] = target
    ordered_hash = hashlib.sha256("\n".join(targets).encode()).hexdigest()
    if ordered_hash != train["ordered_ids_sha256"]:
        raise ValueError("Train ordered IDs differ from frozen plan")
    if np.bincount(list(targets.values()), minlength=49).tolist() != train["geology_class_counts"]:
        raise ValueError("Train targets differ from frozen Geology counts")
    return targets, {"train_plan_sha256": TRAIN_PLAN_SHA256,
                     "metadata_archive_sha256": METADATA_ARCHIVE_SHA256,
                     "train_rows": len(targets), "train_ordered_ids_sha256": ordered_hash}


def validate_train_batch(sample_ids, targets, authoritative_targets):
    """Reject unknown, Val/Test and mislabeled IDs before any model execution."""
    if len(sample_ids) != len(targets):
        raise ValueError("sample IDs/targets differ in length")
    for sample_id, target in zip(sample_ids, targets):
        if sample_id not in authoritative_targets:
            raise ValueError(f"sample ID is not in the authoritative Train manifest: {sample_id}")
        if int(target) != authoritative_targets[sample_id]:
            raise ValueError(f"Geology target differs from authoritative Train manifest: {sample_id}")
