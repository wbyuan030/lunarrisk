#!/usr/bin/env python3
"""Real-pixel Age scratch-head smoke on a fixed HDF5-derived Train/Val pilot.

This is not a benchmark accuracy result or a pretrained-encoder experiment. Test pixels are never
read. Architecture follows the public upstream ScratchAgeClassifier (MIT); training is deliberately
limited to the extraction pilot, and normalization follows the separately pinned data statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunarrisk.benchmarks.moonstone import CHANNELS, MoonstoneHDF5Pilot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model():
    from torch import nn
    return nn.Sequential(
        nn.Conv2d(28, 64, 7, stride=4, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 5))


def dataset_batch(dataset, device):
    import numpy as np
    import torch
    rows = [dataset[i] for i in range(len(dataset))]
    if not rows:
        raise ValueError("nonempty Train/Val pixel pilot required")
    x = torch.from_numpy(np.stack([row["pixels"] for row in rows])).to(device)
    y = torch.as_tensor([row["target"] for row in rows], dtype=torch.long, device=device)
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite normalized input")
    return x, y, [row["sample_id"] for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 5:
        raise ValueError("initial smoke permits 1..5 updates, not benchmark training")
    args.outdir.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested ROCm GPU is unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inputs = {"pilot_sha256": digest(args.pilot), "stats_sha256": digest(args.stats),
              "class_map_sha256": digest(args.class_map)}
    stats, mapping = json.loads(args.stats.read_text()), json.loads(args.class_map.read_text())
    train = MoonstoneHDF5Pilot(args.pilot, stats, mapping, split=0)
    val = MoonstoneHDF5Pilot(args.pilot, stats, mapping, split=1)
    if len(train) > 8 or len(val) > 8:
        raise ValueError("this runner is limited to the initial small-pilot smoke")
    network = model().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    history = []
    start = time.monotonic()
    if args.replay:
        payload = torch.load(args.replay, map_location="cpu", weights_only=True)
        if payload["inputs"] != inputs or payload["protocol"] != "HDF5_DERIVED_PILOT":
            raise ValueError("checkpoint input/protocol mismatch")
        network.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    else:
        x, y, _ = dataset_batch(train, device)
        for step in range(args.steps):
            network.train()
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(network(x), y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite smoke loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
                raise ValueError("nonfinite gradients")
            optimizer.step()
            history.append(float(loss.detach()))
        payload = {"format_version": 1, "architecture": "upstream_scratch_age_cnn_28_64_128_256_5",
                   "state_dict": {k: v.detach().cpu().clone() for k, v in network.state_dict().items()},
                   "optimizer_state_dict": optimizer.state_dict(), "completed_steps": args.steps,
                   "seed": args.seed, "code_sha": os.environ.get("LUNARRISK_GIT_SHA", "UNKNOWN_NOT_CLAIMED"),
                   "torch_rng_state": torch.get_rng_state(),
                   "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                   "inputs": inputs, "stats": stats, "class_map": mapping,
                   "channels": list(CHANNELS), "protocol": "HDF5_DERIVED_PILOT",
                   "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                       (Path(__file__).resolve(), ROOT/"src/lunarrisk/benchmarks/moonstone.py")},
                   "train_ids": train.ids[train.indices].astype(str).tolist(),
                   "val_ids": val.ids[val.indices].astype(str).tolist(), "test_evaluated": False}
        torch.save(payload, args.outdir / "checkpoint.pt")
    network.eval()
    predictions = []
    with torch.no_grad():
        for split, dataset in (("train", train), ("val", val)):
            x, y, ids = dataset_batch(dataset, device)
            probability = torch.softmax(network(x), dim=1).cpu().numpy()
            if not np.isfinite(probability).all():
                raise ValueError("nonfinite probabilities")
            for sample, target, row in zip(ids, y.cpu().tolist(), probability):
                predictions.append({"sample_id": sample, "split": split, "target": target,
                                    **{f"p_{i}": float(p) for i, p in enumerate(row)}})
    import csv
    with (args.outdir / "predictions.csv").open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    report = {"status": "REAL_PIXEL_ENGINEERING_SMOKE_ONLY", "protocol": "HDF5_DERIVED_PILOT",
              "not_benchmark_score": True, "pretrained_weights_used": False, "test_evaluated": False,
              "code_sha": payload["code_sha"], "source_sha256": payload["source_sha256"],
              "steps": payload["completed_steps"], "train_loss": history,
              "inputs": inputs, "train_rows": len(train), "val_rows": len(val),
              "train_class_counts": train.class_counts.tolist(), "val_class_counts": val.class_counts.tolist(),
              "parameters": sum(p.numel() for p in network.parameters()),
              "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "torch": str(torch.__version__), "hip": torch.version.hip,
              "wall_seconds": time.monotonic()-start,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
              "predictions_sha256": digest(args.outdir/"predictions.csv"),
              "checkpoint_sha256": digest(args.replay if args.replay else args.outdir/"checkpoint.pt")}
    (args.outdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    train.close()
    val.close()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
