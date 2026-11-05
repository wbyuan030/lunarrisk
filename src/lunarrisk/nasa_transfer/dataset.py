"""Authenticated, raw-only streaming of the unchanged historical Train/Val cohort.

No downloads, pixel exports, filtering, interpolation or Test access. A channel
slice materializes only terrain; HDF5 may decompress its complete 28-channel
physical chunk internally. Call scan_datasets before freezing a scoring run.
"""
from __future__ import annotations

import copy
import json
import operator
from pathlib import Path

import numpy as np

from ..benchmarks.moonstone import CHANNELS
from ..benchmarks.prefix import PrefixClassifierDataset, _signature
from .cohort import load_authoritative_cohorts
from .contracts import file_sha256, fingerprint, prepare_terrain, validate_domain, validate_provenance, spatial_policy_receipt, STRICT_SPATIAL_POLICY

NODATA_POLICY = "finite_only_no_finite_sentinels"


class TerrainDataset:
    """Internal wrapper over an authenticated PrefixClassifierDataset.

    Public construction alone does not authenticate assets. Production callers
    must use open_datasets, which checks historical hashes and exact membership.
    """

    def __init__(self, source, authoritative_targets, domains, provenance, *,
                 spatial_policy=STRICT_SPATIAL_POLICY, config_sha256=None):
        spatial_policy_receipt(spatial_policy, config_sha256)
        validate_provenance(provenance, spatial_policy=spatial_policy)
        if provenance.get("nodata_policy") != NODATA_POLICY:
            raise ValueError("audited finite-only NoData policy asserting no finite sentinels required")
        for key in ("dtm", "slope"):
            validate_domain(domains[key], spatial_policy=spatial_policy)
        if source.split not in (0, 1) or isinstance(source.split, bool):
            raise ValueError("Train/Val only; Test is forbidden")
        self._spatial_policy = spatial_policy
        self._config_sha256 = config_sha256
        self._source = source
        self._domains = copy.deepcopy(domains)
        self._provenance = copy.deepcopy(provenance)
        self._indices = tuple(map(int, source.indices))
        self._ids = tuple(str(source.ids[index]) for index in self._indices)
        self._targets = tuple(map(int, source.targets))
        if (self._ids != tuple(authoritative_targets)
                or self._targets != tuple(authoritative_targets.values())):
            raise ValueError("dataset differs from full authoritative cohort; filtering forbidden")
        self.split = "train" if source.split == 0 else "val"
        self.ordered_ids_hash = source.ordered_ids_hash

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("boolean sample index forbidden")
        index = operator.index(index)
        source_index = self._indices[index]
        patches = self._source._file()["patches"]
        # Never call PrefixClassifierDataset.__getitem__: it applies Moonstone
        # z-scores and replaces missing pixels. Preserve raw physical values.
        raw = {name: np.asarray(patches[source_index, CHANNELS.index(name):CHANNELS.index(name) + 1,
                                       :, :])[None, ...]
               for name in ("elevation", "slope")}
        valid = {name: np.isfinite(values) for name, values in raw.items()}
        prepared = prepare_terrain(raw, valid, self._domains, self._provenance,
                                   spatial_policy=self._spatial_policy, config_sha256=self._config_sha256)
        return {"dtm": prepared["dtm"][0], "slope": prepared["slope"][0],
                "target": np.int64(self._targets[index]), "sample_id": self._ids[index]}

    def close(self):
        self._source.close()


def open_datasets(*, data_path, stats_path, class_map_path, metadata_path,
                  range_report_path, plan_path, metadata_archive_path, domains, provenance,
                  spatial_policy=STRICT_SPATIAL_POLICY, config_sha256=None):
    """Verify assets once, return ({'train': ds, 'val': ds}, identity_receipt).

    `domains` must come from the caller's authenticated actual NASA config, and
    `provenance` must be audited, including nodata_policy == NODATA_POLICY.
    Both are deep copied and fingerprinted. Prefix, stats, map, metadata and
    cached-range report are authenticated against the repository-pinned plan.
    The existing reader independently verifies all referenced range checksums
    and physical chunk coordinates on first access. Close both returned datasets.
    """
    cohorts, receipt = load_authoritative_cohorts(plan_path, metadata_archive_path)
    # Re-read plan only after its pinned authentication above. Guard against
    # replacement between authentication and parsing by checking these bytes.
    from .cohort import TRAIN_PLAN_SHA256, _read_pinned
    plan = json.loads(_read_pinned(plan_path, TRAIN_PLAN_SHA256, "Train plan"))
    policy_receipt = spatial_policy_receipt(spatial_policy, config_sha256)
    validate_provenance(provenance, spatial_policy=spatial_policy)
    if provenance.get("nodata_policy") != NODATA_POLICY:
        raise ValueError("audited finite-only NoData policy asserting no finite sentinels required")
    for key in ("dtm", "slope"):
        validate_domain(domains[key], spatial_policy=spatial_policy)
    paths = {"sha256": data_path, "stats_sha256": stats_path,
             "class_map_sha256": class_map_path, "metadata_sha256": metadata_path,
             "range_report_sha256": range_report_path}
    signatures = {}
    hashes = {}
    for key, path in paths.items():
        path = Path(path)
        before = _signature(path.stat())
        actual = file_sha256(path)
        if actual != plan["data"][key]:
            raise ValueError(f"historical asset {key} SHA256 mismatch")
        if _signature(path.stat()) != before:
            raise ValueError(f"asset mutated during authentication: {key}")
        signatures[key], hashes[key] = before, actual
    if signatures["sha256"][2] != plan["data"]["bytes"]:
        raise ValueError("retained prefix byte length differs from frozen plan")
    stats = json.loads(Path(stats_path).read_text())
    mapping = json.loads(Path(class_map_path).read_text())
    datasets = {}
    try:
        for split, code in (("train", 0), ("val", 1)):
            source = PrefixClassifierDataset(data_path, stats, mapping, metadata_path,
                                             range_report_path, plan, task="geology", split=code)
            try:
                datasets[split] = TerrainDataset(source, cohorts[split], domains, provenance,
                                                  spatial_policy=spatial_policy, config_sha256=config_sha256)
            except BaseException:
                source.close()
                raise
        for key, path in paths.items():
            if _signature(Path(path).stat()) != signatures[key]:
                raise ValueError(f"asset mutated since authentication: {key}")
    except BaseException:
        for dataset in datasets.values():
            dataset.close()
        raise
    receipt.update(protocol="P33_RAW_TERRAIN_FIXED_P31_COHORT_V1",
                   asset_sha256=hashes, data_bytes=signatures["sha256"][2],
                   domains_sha256=fingerprint(domains), provenance_sha256=fingerprint(provenance),
                   nodata_policy=NODATA_POLICY, pixel_scan="NOT_RUN", spatial_policy=policy_receipt,
                   upstream_full_integrity_verified=False, test_access="FORBIDDEN")
    return datasets, receipt


def scan_datasets(datasets):
    """Read every fixed Train/Val terrain sample without model execution/export.

    Fail the entire scan on one invalid sample; never drop or repair a sample.
    This receipt establishes input eligibility only, not model/science success.
    """
    if set(datasets) != {"train", "val"}:
        raise ValueError("exactly full Train and Val datasets required")
    result = {"pixel_scan": "PASS", "model_execution": "NOT_RUN", "splits": {}}
    for split in ("train", "val"):
        dataset = datasets[split]
        if not isinstance(dataset, TerrainDataset) or dataset.split != split:
            raise ValueError("expected authenticated terrain split")
        for index in range(len(dataset)):
            dataset[index]
        result["splits"][split] = {"rows": len(dataset),
                                   "ordered_ids_sha256": dataset.ordered_ids_hash}
    return result
