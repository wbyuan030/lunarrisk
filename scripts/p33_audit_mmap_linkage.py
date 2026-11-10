#!/usr/bin/env python3
"""Link every fixed Train/Val terrain patch to the published Moonstone mmap product.

Pinned upstream semantics: ``data_preparation/step04_tile.py`` writes HDF5
channel 1/2 (``elevation``, ``slope``) straight from
``aligned/elevation_blended.tif`` and ``derived/slope.tif`` in physical units,
while ``data_preparation/step15_build_mmap.py`` converts the same GeoTIFFs to
headerless float32 ``(23040, 46080)`` arrays, applies the published per-channel
z-score, replaces non-finite pixels with 0 and stores ``packbits(~isfinite)``.

This audit re-derives the z-score from the retained HDF5 physical values and
compares the full 256x256 window against the published mmap window located by
the authenticated ``row_idx``/``col_idx`` mapping. Equality establishes that
the cohort terrain is the shipped product; it never proves an independent
re-execution of the raw LOLA/SLDEM ingestion, model execution, training,
scores or any held-out access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunarrisk.benchmarks.moonstone import CHANNELS
from lunarrisk.benchmarks.prefix import PrefixClassifierDataset, PrefixFile, _signature, cached_blocks
from lunarrisk.nasa_transfer.cohort import (
    METADATA_ARCHIVE_PATH,
    METADATA_ARCHIVE_SHA256,
    TRAIN_PLAN_PATH,
    TRAIN_PLAN_SHA256,
    _read_pinned,
    load_authoritative_cohorts,
)
from lunarrisk.nasa_transfer.contracts import file_sha256

PATCH = 256
GRID = (23040, 46080)
TERRAIN = ("elevation", "slope")
MASK_BITORDER = "big"
MASK_ROW_BYTES = GRID[1] // 8
PLAN_PATH = "protocols/p33_mmap_terrain_linkage_plan.json"
SOURCES = (
    "scripts/p33_audit_mmap_linkage.py",
    "src/lunarrisk/benchmarks/prefix.py",
    "src/lunarrisk/benchmarks/moonstone.py",
    "src/lunarrisk/nasa_transfer/cohort.py",
    "src/lunarrisk/nasa_transfer/contracts.py",
)


def zscore(raw, mean, std):
    """Reproduce the pinned step15 float32 z-score, including NaN -> 0."""
    raw = np.asarray(raw)
    if raw.dtype != np.float32:
        raise ValueError("terrain input must be float32 physical values")
    if not np.isfinite((mean, std)).all() or std <= 0:
        raise ValueError("invalid published normalization statistics")
    result = (raw - np.float32(mean)) / np.float32(std)
    missing = ~np.isfinite(raw)
    if missing.any():
        result = np.where(missing, np.float32(0), result)
    return result.astype(np.float32)


def compare(z_derived, reference, rtol, atol):
    """Count exact/within/beyond matches; NaN never counts as a match."""
    z_derived = np.asarray(z_derived, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if z_derived.shape != reference.shape or z_derived.ndim != 2:
        raise ValueError("paired windows must be equal two-dimensional shapes")
    exact = int((z_derived == reference).sum())
    within = int(np.isclose(z_derived, reference, rtol=rtol, atol=atol).sum())
    diff = np.abs(z_derived.astype(np.float64) - reference.astype(np.float64))
    finite = np.isfinite(diff)
    return {
        "pixels": int(z_derived.size),
        "exact": exact,
        "within": within,
        "beyond": int(z_derived.size - within),
        "derived_nan": int(np.isnan(z_derived).sum()),
        "reference_nan": int(np.isnan(reference).sum()),
        "max_abs_diff": float(diff[finite].max()) if finite.any() else None,
    }


def window(row, col, width=GRID[1]):
    """Return the fixed patch window, rejecting out-of-grid or unaligned tiles."""
    for name, value in (("row", row), ("col", col)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"non-integer {name} index")
        if not 0 <= int(value) < (GRID[0] if name == "row" else width) // PATCH:
            raise ValueError(f"{name} index outside the fixed tile grid")
    return int(row) * PATCH, (int(row) + 1) * PATCH, int(col) * PATCH, (int(col) + 1) * PATCH


def mask_nonzero(packed, row, col, rows=GRID[0], row_bytes=MASK_ROW_BYTES):
    """Count declared-NaN bits inside a tile of the packed global mask."""
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 1 or packed.size != rows * row_bytes:
        raise ValueError("packed mask does not match the fixed grid")
    y0, _, x0, _ = window(row, col, width=row_bytes * 8)
    tile = packed.reshape(rows, row_bytes)[y0:y0 + PATCH, x0 // 8:(x0 + PATCH) // 8]
    return int(np.count_nonzero(tile))


def load_linkage_plan_object(plan):
    """Validate the pre-registered plan fields against the pinned conventions."""
    if plan.get("protocol") != "P33_MMAP_TERRAIN_LINKAGE_V1":
        raise ValueError("unexpected linkage plan")
    grid = plan["grid"]
    if tuple(grid["shape"]) != GRID or grid["patch"] != PATCH or grid["dtype"] != "float32":
        raise ValueError("plan geometry differs from the pinned grid")
    if abs(float(plan["comparison"]["rtol"]) - 1e-5) > 0 or abs(float(plan["comparison"]["atol"]) - 1e-6) > 0:
        raise ValueError("plan tolerance differs from the established probe tolerance")
    if tuple(plan["terrain_channels"]) != TERRAIN:
        raise ValueError("plan must cover exactly elevation and slope")
    return plan


def load_linkage_plan(path):
    """Read and validate the pre-registered plan."""
    return load_linkage_plan_object(json.loads(Path(path).read_text()))


def export_metadata(prefix_path, range_report, plan):
    """Export the fixed HDF5 metadata group through the guarded prefix adapter."""
    import h5py

    logical = int(plan["data"]["upstream_bytes"])
    prefix_bytes = int(plan["data"]["bytes"])
    report_hash = hashlib.sha256(Path(range_report).read_bytes()).hexdigest()
    adapter = PrefixFile(prefix_path, logical, cached_blocks(range_report, report_hash), prefix_bytes)
    try:
        with h5py.File(adapter, "r") as handle:
            names = list(handle.attrs["channel_names"])
            names = [value.decode() if isinstance(value, bytes) else str(value) for value in names]
            if tuple(names) != CHANNELS:
                raise ValueError("HDF5 channel order differs from the pinned code")
            group = handle["metadata"]
            required = ("patch_id", "split", "row_idx", "col_idx", "center_lat", "center_lon")
            arrays = {name: group[name][:].tolist() for name in required if name in group}
            missing = [name for name in required if name not in arrays]
            if missing:
                raise ValueError(f"HDF5 metadata group lacks {missing}")
            geology = group["geology"]
            arrays["geology/dominant_unit_id"] = geology["dominant_unit_id"][:].tolist()
            arrays["geology/dominant_unit_name"] = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in geology["dominant_unit_name"][:]]
    finally:
        adapter.close()
    payload = {"metadata/" + key: value for key, value in arrays.items()}
    return {"attributes": {"channel_names": names}, "metadata": payload}


def verify_metadata(export, archive_rows):
    """Require exact agreement between the HDF5 metadata group and the pinned archive."""
    meta = export["metadata"]
    counts = {key: len(value) for key, value in meta.items()}
    if len(set(counts.values())) != 1 or len(archive_rows) != max(counts.values()):
        raise ValueError("metadata/archive row counts disagree")
    ids = [str(value) for value in meta["metadata/patch_id"]]
    if ids != [row["sample_id"] for row in archive_rows] or len(set(ids)) != len(ids):
        raise ValueError("patch IDs differ from the authenticated archive")
    checks = {}
    checks["split"] = [int(value) for value in meta["metadata/split"]] == [int(row["stored_split"]) for row in archive_rows]
    checks["unit_id"] = [int(value) for value in meta["metadata/geology/dominant_unit_id"]] == \
        [int(row["unit_id"]) for row in archive_rows]
    names = [str(value).strip() for value in meta["metadata/geology/dominant_unit_name"]]
    checks["unit_name"] = names == [row["unit_name"].strip() for row in archive_rows]
    for key, column in (("row_idx", "row_idx"), ("col_idx", "col_idx")):
        checks[key] = [int(value) for value in meta["metadata/" + key]] == [int(row[column]) for row in archive_rows]
    lats = np.asarray(meta["metadata/center_lat"], dtype=np.float64)
    lons = np.asarray(meta["metadata/center_lon"], dtype=np.float64)
    rows = np.asarray([int(row["row_idx"]) for row in archive_rows])
    cols = np.asarray([int(row["col_idx"]) for row in archive_rows])
    expected_lat = 90.0 - (rows * PATCH + PATCH / 2) / 128.0
    expected_lon = -180.0 + (cols * PATCH + PATCH / 2) / 128.0
    checks["center_lat_within_half_pixel"] = bool(np.abs(lats - expected_lat).max() <= 1.0 / 256.0)
    checks["center_lon_within_half_pixel"] = bool(np.abs(lons - expected_lon).max() <= 1.0 / 256.0)
    if not all(checks.values()):
        raise ValueError(f"metadata arrays disagree with the authenticated archive: {checks}")
    return {"rows": len(ids), "checks": checks,
            "archive_sha256": METADATA_ARCHIVE_SHA256,
            "historical_export_sha256_present_locally": False,
            "source": "HDF5 metadata group read through the guarded prefix adapter"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "range-report", "mmap-dir", "outdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / PLAN_PATH)
    parser.add_argument("--stats", type=Path, default=ROOT / "reports/evidence/p30_20261103/hf_channel_stats.json")
    parser.add_argument("--class-map", type=Path, default=ROOT / "reports/evidence/p30_20261103/geologic_units.json")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_hashes = {name: file_sha256(ROOT / name) for name in SOURCES}
    linkage = load_linkage_plan(args.plan)
    cohorts, authority = load_authoritative_cohorts(ROOT / TRAIN_PLAN_PATH, ROOT / METADATA_ARCHIVE_PATH)
    plan = json.loads(_read_pinned(ROOT / TRAIN_PLAN_PATH, TRAIN_PLAN_SHA256, "Train plan"))
    paths = {"sha256": args.data, "range_report_sha256": args.range_report, "stats_sha256": args.stats,
             "class_map_sha256": args.class_map}
    signatures, assets = {}, {}
    for key, path in paths.items():
        signatures[key] = _signature(path.stat())
        assets[key] = file_sha256(path)
        if assets[key] != plan["data"][key] or _signature(path.stat()) != signatures[key]:
            raise ValueError(f"historical asset mismatch/mutation: {key}")
    if args.data.stat().st_size != plan["data"]["bytes"]:
        raise ValueError("prefix length mismatch")
    print("historical assets authenticated", flush=True)
    archive_rows = _archive_rows()
    export = export_metadata(args.data, args.range_report, plan)
    findings = verify_metadata(export, archive_rows)
    export_path = args.outdir / "metadata_export.json"
    export_path.write_text(json.dumps(export, indent=2))
    print("HDF5 metadata group matches the pinned archive", flush=True)
    published, local_by_digest = {}, {}
    for entry in linkage["assets"]:
        if "lfs_sha256" not in entry:
            published[entry["path"]] = entry["git_blob_sha"]
            continue
        local = args.mmap_dir / entry["path"].rsplit("/", 1)[-1]
        if local.is_file() and local.stat().st_size == entry["bytes"]:
            digest = file_sha256(local)
            if digest != entry["lfs_sha256"]:
                raise ValueError(f"published asset SHA256 mismatch: {entry['path']}")
            local_by_digest[digest] = local
        elif entry["lfs_sha256"] not in local_by_digest:
            raise ValueError(f"published asset absent or truncated: {entry['path']}")
        published[entry["path"]] = entry["lfs_sha256"]
    print("published mmap assets authenticated", flush=True)
    stats = json.loads(args.stats.read_text())
    if file_sha256(args.stats) != linkage["normalization"]["stats_sha256"]:
        raise ValueError("normalization statistics differ from the pre-registered file")
    mapping = json.loads(args.class_map.read_text())
    memory, masks, mask_total = {}, {}, {}
    for name in TERRAIN:
        memory[name] = np.memmap(args.mmap_dir / f"{name}.npy", dtype=np.float32, mode="r", shape=GRID)
        if memory[name].shape != GRID:
            raise ValueError("published value array does not match the fixed grid")
        digest = published[f"pretraining/mmap/{name}_mask.npy"]
        masks[name] = np.load(local_by_digest[digest], mmap_mode="r")
        if masks[name].shape != (GRID[0] * MASK_ROW_BYTES,) or masks[name].dtype != np.uint8:
            raise ValueError("published packed mask does not match the fixed grid")
        mask_total[name] = int(np.count_nonzero(masks[name]))
    summaries = {}
    for split, code in (("train", 0), ("val", 1)):
        dataset = PrefixClassifierDataset(args.data, stats, mapping, export_path,
                                          args.range_report, plan, task="geology", split=code)
        try:
            ids = tuple(str(dataset.ids[index]) for index in dataset.indices)
            if ids != tuple(cohorts[split]) or tuple(map(int, dataset.targets)) != tuple(cohorts[split].values()):
                raise ValueError("full authoritative IDs/targets mismatch")
            grid_map = {str(sample): (row, col) for sample, row, col in
                        zip(export["metadata"]["metadata/patch_id"],
                            export["metadata"]["metadata/row_idx"],
                            export["metadata"]["metadata/col_idx"])}
            aggregate = {"rows": 0, "pass_rows": 0, "failed_ids": [], "channels": {},
                         "declared_nan_bits": 0, "ordered_ids_sha256": dataset.ordered_ids_hash}
            digest = hashlib.sha256()
            patches = dataset._file()["patches"]
            start = CHANNELS.index("elevation")
            if CHANNELS[start:start + 2] != TERRAIN:
                raise ValueError("terrain channels must be adjacent")
            with (args.outdir / f"{split}_rows.csv").open("x", newline="") as output:
                writer = None
                for ordinal, index in enumerate(dataset.indices):
                    dataset._file()  # retain prefix mutation guard at every read
                    raw = np.asarray(patches[int(index), start:start + 2, :, :])
                    if raw.shape != (2, PATCH, PATCH) or raw.dtype != np.float32:
                        raise ValueError("unexpected terrain slice schema")
                    sample_id = ids[ordinal]
                    if sample_id not in grid_map:
                        raise ValueError("sample absent from the authenticated grid metadata")
                    row_idx, col_idx = grid_map[sample_id]
                    y0, y1, x0, x1 = window(row_idx, col_idx)
                    digest.update(sample_id.encode() + b"\0" + raw.tobytes(order="C"))
                    record = {"split": split, "ordinal": ordinal, "source_index": int(index),
                              "sample_id": sample_id, "target": int(dataset.targets[ordinal]),
                              "row_idx": row_idx, "col_idx": col_idx}
                    ok = True
                    for band, name in zip(raw, TERRAIN):
                        mean, std = stats[name]
                        reference = np.asarray(memory[name][y0:y1, x0:x1], dtype=np.float32)
                        measures = compare(zscore(band, mean, std), reference,
                                           linkage["comparison"]["rtol"], linkage["comparison"]["atol"])
                        record.update({name + "_" + key: value for key, value in measures.items()})
                        total = aggregate["channels"].setdefault(
                            name, {"pixels": 0, "exact": 0, "within": 0, "beyond": 0,
                                   "derived_nan": 0, "reference_nan": 0, "max_abs_diff": None})
                        for key in ("pixels", "exact", "within", "beyond", "derived_nan", "reference_nan"):
                            total[key] += measures[key]
                        if measures["max_abs_diff"] is not None:
                            current = total["max_abs_diff"]
                            total["max_abs_diff"] = measures["max_abs_diff"] if current is None \
                                else max(current, measures["max_abs_diff"])
                        ok = ok and measures["beyond"] == 0
                    bits = mask_nonzero(masks["elevation"], row_idx, col_idx)
                    if bits != mask_nonzero(masks["slope"], row_idx, col_idx):
                        raise ValueError("elevation/slope masks disagree on the shared file")
                    record["declared_nan_bits"] = bits
                    aggregate["declared_nan_bits"] += bits
                    ok = ok and bits == 0
                    aggregate["rows"] += 1
                    aggregate["pass_rows"] += int(ok)
                    if not ok:
                        aggregate["failed_ids"].append(sample_id)
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=list(record))
                        writer.writeheader()
                    writer.writerow(record)
                    if (ordinal + 1) % 500 == 0:
                        print(f"{split}: {ordinal + 1}/{len(dataset)}", flush=True)
            aggregate["terrain_ordered_sha256"] = digest.hexdigest()
            summaries[split] = aggregate
        finally:
            dataset.close()
    for key, path in paths.items():
        if _signature(path.stat()) != signatures[key]:
            raise ValueError(f"asset mutated during scan: {key}")
    if source_hashes != {name: file_sha256(ROOT / name) for name in SOURCES}:
        raise ValueError("source changed during scan")
    complete = all(summaries[split]["failed_ids"] == [] and summaries[split]["rows"] > 0 for split in summaries) \
        and all(total["beyond"] == 0 for value in summaries.values() for total in value["channels"].values())
    report = {"status": "TERRAIN_MMAP_LINKAGE_PASS" if complete else "TERRAIN_MMAP_LINKAGE_FAILED",
              "protocol": linkage["protocol"], "model_execution": "NOT_RUN", "training": "NOT_RUN",
              "scores": "NOT_RUN", "test_pixels_read": 0, "Test": "HOLD",
              "authority": authority, "asset_sha256": assets, "published_sha256": published,
              "source_sha256": source_hashes, "declared_nan_bytes_global": mask_total,
              "metadata": {"findings": findings, "derived_export_sha256": file_sha256(export_path),
                           "historical_export_sha256": plan["data"]["metadata_sha256"]},
              "splits": summaries,
              "rows_csv_sha256": {split: file_sha256(args.outdir / f"{split}_rows.csv") for split in summaries},
              "upstream_ingestion_re_executed": False,
              "claims_permitted": linkage["claims_permitted"],
              "elapsed_seconds": time.monotonic() - started,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "python": sys.version.split()[0], "numpy": np.__version__,
              "torch_imported": "torch" in sys.modules}
    with (args.outdir / "summary.json").open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"status": report["status"],
                      "rows": {split: value["rows"] for split, value in summaries.items()},
                      "pass_rows": {split: value["pass_rows"] for split, value in summaries.items()}}))
    if not complete:
        raise SystemExit(1)


def _archive_rows():
    """Return authenticated metadata rows (split/target identity checked separately)."""
    import csv as csv_module
    import gzip
    import io
    body = _read_pinned(ROOT / METADATA_ARCHIVE_PATH, METADATA_ARCHIVE_SHA256, "metadata archive")
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
        return list(csv_module.DictReader(io.TextIOWrapper(compressed, encoding="utf-8")))


if __name__ == "__main__":
    main()
