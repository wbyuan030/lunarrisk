#!/usr/bin/env python3
"""E0 hardware smoke test: tiny 4-layer CNN on random tensors (HIP only).

This proves that PyTorch can run forward/backward on the local AMD GPU.
It says nothing about LFM throughput, real lunar data, or scientific quality.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """4 conv layers, 1-channel input, one logit for BCEWithLogitsLoss."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("outputs/rocm_smoke.json"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if torch.version.hip is None:
        print("FAIL: this PyTorch build has no HIP support (torch.version.hip is None).", file=sys.stderr)
        print("Refusing to fall back to CPU: the E0 gate requires a HIP GPU run.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False inside a HIP build.", file=sys.stderr)
        print("Check /dev/kfd, /dev/dri, group permissions and the container device mounts.", file=sys.stderr)
        return 2
    if args.device_index >= torch.cuda.device_count():
        print(f"FAIL: device index {args.device_index} >= device_count {torch.cuda.device_count()}", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device_index}")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    model = TinyCNN().to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    x = torch.randn(args.batch, 1, args.size, args.size, device=device, dtype=dtype)
    target = (torch.rand(args.batch, device=device) < 0.3).to(dtype)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    losses: list[float] = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"step {step + 1}/{args.steps} loss={losses[-1]:.6f}", flush=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    props = torch.cuda.get_device_properties(args.device_index)
    result = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "device_index": args.device_index,
        "device_name": props.name,
        "gcn_arch": getattr(props, "gcnArchName", None),
        "total_vram_gb": round(props.total_memory / 2**30, 4),
        "multi_processor_count": props.multi_processor_count,
        "config": {
            "steps": args.steps,
            "batch": args.batch,
            "size": args.size,
            "dtype": args.dtype,
            "seed": args.seed,
            "lr": args.lr,
            "model": "4x conv+BN+ReLU+pool, linear(128->1), BCEWithLogits",
        },
        "steps_completed": args.steps,
        "elapsed_sec": round(elapsed, 4),
        "samples_per_sec": round(args.steps * args.batch / elapsed, 3),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 3),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "losses": losses,
        "note": "Synthetic tensors only: hardware/pipeline smoke test, not a lunar-data result.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "status", "device_name", "torch", "torch_hip", "steps_completed",
        "elapsed_sec", "samples_per_sec", "peak_vram_mb")}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
