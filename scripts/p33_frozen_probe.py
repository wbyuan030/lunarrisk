#!/usr/bin/env python3
"""Paired frozen-feature Geology probe over the fixed Train/Val cohort.

Both arms share the official NASA base architecture, two raw physical inputs,
the same frozen encoder policy, the same head initialization and the same
optimizer budget. Only the encoder weights differ: verified pretrained versus
random from the same seed. Features are the encoder token sequence mean-pooled
exactly as `TerrainTransfer.forward` does, cached once per arm so that head
training is cheap and reproducible. Test is never constructed or read.
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
from lunarrisk.benchmarks.prefix import PrefixClassifierDataset, _signature
from lunarrisk.nasa_transfer.cohort import (
    METADATA_ARCHIVE_PATH, METADATA_ARCHIVE_SHA256, TRAIN_PLAN_PATH, TRAIN_PLAN_SHA256,
    _read_pinned, load_authoritative_cohorts,
)
from lunarrisk.nasa_transfer.contracts import (
    NASA_CHECKPOINT_BYTES, NASA_CHECKPOINT_SHA256, NATIVE_SPATIAL_POLICY,
    file_sha256, prepare_terrain, validate_provenance,
)
from lunarrisk.nasa_transfer.experiment import probability_metrics, validate_budget

PLAN_PATH = "protocols/p33_frozen_probe_plan.json"
ADAPTATION = "frozen"
# Measured across fresh processes on this ROCm build: random 4.8e-7, pretrained
# up to 1.5e-4 against feature scales of 0.28-3.8.  GPU kernels are stable within a process but not bitwise across
# processes, so reproducibility is judged numerically with a pre-set bound.
FEATURE_TOLERANCE = 1e-3


class ProbeDataset:
    """Raw physical terrain from the authenticated prefix, prepared per config."""

    def __init__(self, source, authoritative_targets, domains, provenance, config_sha256):
        validate_provenance(provenance, spatial_policy=NATIVE_SPATIAL_POLICY)
        if source.split not in (0, 1) or isinstance(source.split, bool):
            raise ValueError("Train/Val only; Test is forbidden")
        self._source = source
        self._domains = domains
        self._provenance = provenance
        self._config_sha256 = config_sha256
        self.split = "train" if source.split == 0 else "val"
        self._indices = tuple(map(int, source.indices))
        self._ids = tuple(str(source.ids[index]) for index in self._indices)
        self._targets = tuple(map(int, source.targets))
        if (self._ids != tuple(authoritative_targets)
                or self._targets != tuple(authoritative_targets.values())):
            raise ValueError("dataset differs from the full authoritative cohort")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, position):
        if isinstance(position, (bool, np.bool_)) or not isinstance(position, (int, np.integer)):
            raise TypeError("integer sample index required")
        if not 0 <= int(position) < len(self):
            raise IndexError("sample index out of range")
        source_index = self._indices[int(position)]
        patches = self._source._file()["patches"]
        raw = {name: np.asarray(patches[source_index,
                                       CHANNELS.index(name):CHANNELS.index(name) + 1, :, :])[None, ...]
               for name in ("elevation", "slope")}
        valid = {name: np.isfinite(values) for name, values in raw.items()}
        prepared = prepare_terrain(raw, valid, self._domains, self._provenance,
                                   spatial_policy=NATIVE_SPATIAL_POLICY,
                                   config_sha256=self._config_sha256)
        return {"dtm": prepared["dtm"][0], "slope": prepared["slope"][0],
                "target": np.int64(self._targets[int(position)]), "sample_id": self._ids[int(position)]}

    def close(self):
        self._source.close()


def load_probe_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("protocol") != "P33_FROZEN_FEATURE_PROBE_V1":
        raise ValueError("unexpected probe plan")
    if plan["design"]["adaptation"] != ADAPTATION or plan["design"]["spatial_policy"] != NATIVE_SPATIAL_POLICY:
        raise ValueError("plan adaptation or spatial policy differs from this implementation")
    if plan["budget"]["seeds"] != [42, 43, 44]:
        raise ValueError("the frozen plan declares three seeds")
    payload = {key: plan["budget"][key] for key in (
        "epochs", "patience", "batch_size", "lr", "weight_decay", "grad_clip", "seeds",
        "optimizer", "schedule", "selection", "test_scored")}
    payload["adaptation"] = plan["design"]["adaptation"]
    payload["precision"] = "fp32"
    budget = validate_budget(payload)
    return plan, budget


def authenticate(args, plan):
    """Verify every pinned input before the model or the prefix is touched."""
    if file_sha256(args.config) != plan["inputs"]["nasa_config_sha256"]:
        raise ValueError("NASA config SHA256 mismatch")
    if args.config_sha256 != plan["inputs"]["nasa_config_sha256"]:
        raise ValueError("--config-sha256 differs from the plan")
    if args.checkpoint.stat().st_size != NASA_CHECKPOINT_BYTES or file_sha256(args.checkpoint) != NASA_CHECKPOINT_SHA256:
        raise ValueError("NASA checkpoint size/hash mismatch")
    if file_sha256(args.metadata) != plan["inputs"]["metadata_export_sha256"]:
        raise ValueError("metadata export SHA256 mismatch")
    if file_sha256(args.range_report) != plan["inputs"]["range_report_sha256"]:
        raise ValueError("range report SHA256 mismatch")
    if file_sha256(args.stats) != plan["inputs"]["hf_channel_stats_sha256"]:
        raise ValueError("channel statistics SHA256 mismatch")
    if file_sha256(args.class_map) != plan["inputs"]["class_map_sha256"]:
        raise ValueError("class map SHA256 mismatch")
    train_plan = json.loads(_read_pinned(ROOT / TRAIN_PLAN_PATH, TRAIN_PLAN_SHA256, "Train plan"))
    if file_sha256(args.data) != train_plan["data"]["sha256"]:
        raise ValueError("training prefix SHA256 mismatch")
    if args.data.stat().st_size != train_plan["data"]["bytes"]:
        raise ValueError("training prefix length mismatch")
    provenance = json.loads(args.provenance.read_text())
    validate_provenance(provenance, spatial_policy=NATIVE_SPATIAL_POLICY)
    return train_plan, provenance


def build_datasets(args, train_plan, provenance):
    cohorts, receipt = load_authoritative_cohorts(ROOT / TRAIN_PLAN_PATH, ROOT / METADATA_ARCHIVE_PATH)
    stats = json.loads(args.stats.read_text())
    mapping = json.loads(args.class_map.read_text())
    import yaml
    config = yaml.safe_load(args.config.read_text())
    datasets = {}
    for split, code in (("train", 0), ("val", 1)):
        source = PrefixClassifierDataset(args.data, stats, mapping, args.metadata,
                                         args.range_report, train_plan, task="geology", split=code)
        datasets[split] = ProbeDataset(source, cohorts[split], config["data"]["domains"],
                                       provenance, args.config_sha256)
    return datasets, receipt


def build_models(args):
    from lunarrisk.nasa_transfer.torch_adapter import (
        build_paired_models, load_verified_checkpoint, official_backbone,
    )
    import torch
    factory = lambda: official_backbone(args.config, args.config_sha256, args.upstream,
                                        spatial_policy=NATIVE_SPATIAL_POLICY)
    models, coverage = build_paired_models(
        factory, lambda backbone: load_verified_checkpoint(backbone, args.checkpoint),
        seed=42, adapt_inputs=False, num_classes=49)
    return models, coverage


def _batch_inputs(items, device):
    import torch
    return {key: torch.stack([torch.as_tensor(item[key]) for item in items]).to(device, dtype=torch.float32)
            for key in ("dtm", "slope")}


def _features(model, inputs):
    """Encoder tokens exactly as TerrainTransfer.forward consumes them."""
    adapted = {key: model.adapters[key](inputs[key]) for key in model.adapters}
    tokens = model.backbone(adapted)[-1]
    if tokens.ndim != 3 or not tokens.shape[1]:
        raise ValueError("unexpected encoder token shape")
    return tokens


def extract(args, plan, models, datasets, cache_dir):
    import torch
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    for model in models.values():
        model.to(device).float().eval()
    cache_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"plan": plan["protocol"], "adaptation": ADAPTATION, "device": str(device),
                "torch": torch.__version__, "splits": {}, "arms": {}}
    for arm, model in models.items():
        manifest["arms"][arm] = {"state_sha256": hashlib.sha256(
            json.dumps({key: float(torch.sum(value.double())) for key, value in
                        model.backbone.state_dict().items()}).encode()).hexdigest()}
    started = time.monotonic()
    for split, dataset in datasets.items():
        n = len(dataset)
        ids, targets = list(dataset._ids), list(dataset._targets)
        shape = None
        buffers = {}
        with torch.no_grad():
            for start in range(0, n, args.extraction_batch):
                items = [dataset[position] for position in range(start, min(start + args.extraction_batch, n))]
                inputs = _batch_inputs(items, device)
                for arm, model in models.items():
                    tokens = _features(model, inputs)
                    # Store exactly the pooled vector that TerrainTransfer.forward feeds the head.
                    pooled = tokens.mean(dim=1).cpu().numpy().astype(np.float32)
                    if shape is None:
                        shape = (n,) + pooled.shape[1:]
                        for name in models:
                            buffers[name] = np.memmap(cache_dir / f"{name}_{split}.f32", dtype=np.float32,
                                                      mode="w+", shape=shape)
                    buffers[arm][start:start + len(items)] = pooled
                if (start // args.extraction_batch) % 50 == 0:
                    print(f"{split}: {min(start + args.extraction_batch, n)}/{n}", flush=True)
        for name, buffer in buffers.items():
            buffer.flush()
        manifest["splits"][split] = {"rows": n, "shape": list(shape),
                                     "ordered_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                                     "ids": ids, "targets": targets}
        print(f"{split}: complete shape={shape}", flush=True)
    del buffers
    manifest["seconds"] = time.monotonic() - started
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _load_split(cache_dir, arm, split, manifest):
    shape = tuple(manifest["splits"][split]["shape"])
    array = np.memmap(cache_dir / f"{arm}_{split}.f32", dtype=np.float32, mode="r", shape=shape)
    targets = np.asarray(manifest["splits"][split]["targets"], dtype=np.int64)
    return array, targets


def _evaluate(head, features, targets):
    import torch
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(np.asarray(features, dtype=np.float32)))
        probability = torch.softmax(logits.double(), dim=1).numpy()
    return probability_metrics(probability, targets), probability


def train_heads(args, plan, budget, cache_dir, outdir):
    import torch
    from torch import nn
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    outdir.mkdir(parents=True, exist_ok=False)
    reports = {}
    for seed in budget["seeds"]:
        import copy
        width = manifest["splits"]["train"]["shape"][-1]
        torch.manual_seed(seed)
        heads = {"random": nn.Linear(width, 49)}
        heads["pretrained"] = copy.deepcopy(heads["random"])
        reference = heads["random"].state_dict()
        left = heads["pretrained"].state_dict()
        if any(not torch.equal(left[key], reference[key]) for key in left):
            raise RuntimeError("paired head initialization differs")
        for arm, head in heads.items():
            torch.manual_seed(seed)
            optimizer = torch.optim.AdamW(head.parameters(), lr=budget["lr"],
                                          weight_decay=budget["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
            train_features, train_targets = _load_split(cache_dir, arm, "train", manifest)
            val_features, val_targets = _load_split(cache_dir, arm, "val", manifest)
            best, best_epoch, stale, best_state, history = -1.0, 0, 0, None, []
            began = time.monotonic()
            for epoch in range(1, budget["epochs"] + 1):
                head.train()
                generator = torch.Generator().manual_seed(seed + epoch)
                permutation = torch.randperm(len(train_features), generator=generator).tolist()
                loss_sum, seen, order = 0.0, 0, []
                for start in range(0, len(permutation), budget["batch_size"]):
                    chosen = permutation[start:start + budget["batch_size"]]
                    optimizer.zero_grad(set_to_none=True)
                    tokens = torch.from_numpy(np.asarray(train_features[chosen], dtype=np.float32))
                    labels = torch.from_numpy(train_targets[chosen])
                    logits = head(tokens)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), budget["grad_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    seen += len(chosen)
                    loss_sum += float(loss.detach()) * len(chosen)
                    order.extend(str(index) for index in chosen)
                if seen != len(train_features) or len(set(order)) != seen:
                    raise ValueError("training dropped or repeated rows")
                val, _ = _evaluate(head, val_features, val_targets)
                metric = val["macro_f1_full"]
                if metric > best:
                    best, best_epoch, stale = metric, epoch, 0
                    best_state = {key: value.clone() for key, value in head.state_dict().items()}
                else:
                    stale += 1
                history.append({"epoch": epoch, "train_rows": seen, "train_loss": loss_sum / seen,
                                "val": val, "training_order_sha256":
                                    hashlib.sha256("\n".join(order).encode()).hexdigest()})
                if stale >= budget["patience"]:
                    break
            head.load_state_dict(best_state)
            head.eval()
            metrics, probability = _evaluate(head, val_features, val_targets)
            tag = f"{arm}_seed{seed}"
            with (outdir / f"{tag}_val_predictions.csv").open("x", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_id", "split", "target"] + [f"p_{i}" for i in range(49)])
                ids = manifest["splits"]["val"]["ids"]
                writer.writerows([ids[i], "val", int(val_targets[i]), *probability[i].tolist()]
                                 for i in range(len(ids)))
            checkpoint = outdir / f"{tag}.pt"
            torch.save({"arm": arm, "seed": seed, "budget": budget,
                        "state": {key: value for key, value in head.state_dict().items()},
                        "best_epoch": best_epoch, "best_metric": best,
                        "feature_manifest_sha256": file_sha256(cache_dir / "manifest.json")}, checkpoint)
            reports[tag] = {"arm": arm, "seed": seed, "budget": budget, "metrics": metrics,
                            "history": history, "best_epoch": best_epoch, "best_metric": best,
                            "checkpoint_sha256": file_sha256(checkpoint),
                            "prediction_sha256": file_sha256(outdir / f"{tag}_val_predictions.csv"),
                            "seconds": time.monotonic() - began,
                            "trainable_parameters": sum(p.numel() for p in head.parameters())}
            print(f"{tag}: macro_f1={metrics['macro_f1_full']:.6f} best_epoch={best_epoch}", flush=True)
    paired = {}
    for seed in budget["seeds"]:
        diff = reports[f"pretrained_seed{seed}"]["metrics"]["macro_f1_full"] - \
            reports[f"random_seed{seed}"]["metrics"]["macro_f1_full"]
        paired[str(seed)] = diff
    values = {"pretrained": [reports[f"pretrained_seed{seed}"]["metrics"]["macro_f1_full"] for seed in budget["seeds"]],
              "random": [reports[f"random_seed{seed}"]["metrics"]["macro_f1_full"] for seed in budget["seeds"]]}
    summary = {"status": "FROZEN_PROBE_COMPLETE", "protocol": plan["protocol"],
               "adaptation": ADAPTATION, "budget": budget, "arms": reports,
               "per_seed_difference": paired,
               "macro_f1": {arm: {"mean": float(np.mean(values[arm])),
                                  "sample_std": float(np.std(values[arm], ddof=1)) if len(values[arm]) > 1 else None,
                                  "values": values[arm]} for arm in values},
               "test_scored": False, "provenance_limitation": plan["provenance_limitation"],
               "cache_manifest_sha256": file_sha256(cache_dir / "manifest.json"),
               "torch_version": __import__("torch").__version__,
               "trainable_parameters": sum(p.numel() for p in heads["random"].parameters()),
               "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def check(args, plan, models, datasets, cache_dir, outdir):
    """Fresh-process reproducibility: cached features and logits must match."""
    import torch
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    device = torch.device(args.device)
    for model in models.values():
        model.to(device).float().eval()
    outdir.mkdir(parents=True, exist_ok=False)
    receipt = {"plan": plan["protocol"], "checks": [], "status": "PASS"}
    for split, dataset in datasets.items():
        ids = manifest["splits"][split]["ids"]
        positions = [0, 1, len(dataset) - 2, len(dataset) - 1]
        items = [dataset[position] for position in positions]
        inputs = _batch_inputs(items, device)
        for arm, model in models.items():
            with torch.no_grad():
                fresh = _features(model, inputs).mean(dim=1).cpu().numpy().astype(np.float32)
                cached = np.asarray(np.memmap(cache_dir / f"{arm}_{split}.f32", dtype=np.float32, mode="r",
                                             shape=tuple(manifest["splits"][split]["shape"]))[positions])
            bitwise = int((fresh.view(np.uint32) == cached.view(np.uint32)).sum())
            max_abs = float(np.abs(fresh.astype(np.float64) - cached.astype(np.float64)).max())
            receipt["checks"].append({"split": split, "arm": arm, "positions": positions,
                                      "values": int(fresh.size), "bitwise": bitwise,
                                      "max_abs_diff": max_abs, "tolerance": FEATURE_TOLERANCE,
                                      "sample_ids": [item["sample_id"] for item in items]})
            if max_abs > FEATURE_TOLERANCE:
                receipt["status"] = "FAILED"
    (outdir / "replay.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["status"] != "PASS":
        raise SystemExit(1)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("extract", "train", "check", "all"), required=True)
    for name in ("data", "metadata", "range-report", "config", "checkpoint", "provenance", "upstream", "outdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / PLAN_PATH)
    parser.add_argument("--stats", type=Path, default=ROOT / "reports/evidence/p30_20261103/hf_channel_stats.json")
    parser.add_argument("--class-map", type=Path, default=ROOT / "reports/evidence/p30_20261103/geologic_units.json")
    parser.add_argument("--extraction-batch", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, help="existing feature cache for check mode")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    plan, budget = load_probe_plan(args.plan)

    def prepare():
        torch_ready = __import__("torch")
        torch_ready.use_deterministic_algorithms(True)
        # TF32 convolution kernels pick different algorithms per process on this
        # ROCm build; disabling them makes cached features bitwise reproducible.
        torch_ready.backends.cudnn.allow_tf32 = False
        torch_ready.backends.cuda.matmul.allow_tf32 = False
        if args.device == "cuda" and not torch_ready.cuda.is_available():
            raise RuntimeError("CUDA/ROCm device requested but unavailable")
        return torch_ready

    if args.mode == "train":
        prepare()
        summary = train_heads(args, plan, budget, args.cache_dir or (args.outdir / "features"), args.outdir)
        print(json.dumps({"status": summary["status"],
                          "macro_f1": {arm: value["mean"] for arm, value in summary["macro_f1"].items()}}))
        return
    if args.mode in ("extract", "all"):
        prepare()
        train_plan, provenance = authenticate(args, plan)
        datasets, receipt = build_datasets(args, train_plan, provenance)
        models, coverage = build_models(args)
        try:
            cache_dir = args.outdir / "features"
            extract(args, plan, models, datasets, cache_dir)
            if args.mode == "all":
                summary = train_heads(args, plan, budget, cache_dir, args.outdir / "heads")
                print(json.dumps({"status": summary["status"],
                                  "macro_f1": {arm: value["mean"] for arm, value in summary["macro_f1"].items()}}))
        finally:
            for dataset in datasets.values():
                dataset.close()
    elif args.mode == "check":
        prepare()
        train_plan, provenance = authenticate(args, plan)
        datasets, receipt = build_datasets(args, train_plan, provenance)
        models, coverage = build_models(args)
        try:
            check(args, plan, models, datasets, args.cache_dir or (args.outdir / "features"), args.outdir)
        finally:
            for dataset in datasets.values():
                dataset.close()


if __name__ == "__main__":
    main()
