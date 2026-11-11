#!/usr/bin/env python3
"""Concatenate in-domain and cross-domain frozen features; compare Geology and Age.

All cells share one budget. Feature sources are authenticated: NASA pooled
tokens from the pinned extraction cache and Moonstone pooled tokens plus Age
targets from the authenticated P3.2 H5 caches, whose ordered IDs must equal the
NASA manifest. Test is never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lunarrisk.nasa_transfer.contracts import file_sha256
from lunarrisk.nasa_transfer.experiment import validate_budget

PLAN_PATH = "protocols/p33_feature_fusion_plan.json"


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("protocol") != "P33_FEATURE_FUSION_V1":
        raise ValueError("unexpected fusion plan")
    payload = {key: plan["budget"][key] for key in (
        "epochs", "patience", "batch_size", "lr", "weight_decay", "grad_clip", "seeds",
        "optimizer", "schedule", "selection", "test_scored")}
    payload["adaptation"] = "frozen"
    payload["precision"] = "fp32"
    return plan, validate_budget(payload)


def macro_metrics(probabilities, targets, classes):
    """Macro-F1 over the full class count; absent classes stay in the denominator."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != classes or len(y) != len(p):
        raise ValueError("probability/target shape mismatch")
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("invalid probabilities")
    prediction = p.argmax(axis=1)
    confusion = np.bincount(y * classes + prediction, minlength=classes ** 2).reshape(classes, classes)
    support = confusion.sum(axis=1)
    denominator = support + confusion.sum(axis=0)
    f1 = np.divide(2.0 * np.diag(confusion), denominator, out=np.zeros(classes), where=denominator > 0)
    return {"macro_f1_full": float(f1.mean()), "accuracy": float(np.mean(prediction == y)),
            "rows": int(len(p)), "per_class_f1": f1.tolist(), "support": support.tolist()}


def train_cell(head_name, features, targets, seed, budget, classes, torch, nn):
    torch.manual_seed(seed)
    width = features[("x", "train")].shape[1]
    head = nn.Linear(width, classes)
    weight = None
    if head_name == "linear_balanced":
        counts = np.bincount(targets["train"], minlength=classes).astype(np.float64)
        weights = np.zeros(classes)
        np.divide(1.0, np.sqrt(counts), out=weights, where=counts > 0)
        weights = weights / weights[counts > 0].mean()
        weight = torch.tensor(weights, dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=budget["lr"], weight_decay=budget["weight_decay"],
                                  betas=(0.9, 0.999), eps=1e-8)
    best, best_epoch, stale, history = -1.0, 0, 0, []
    started = time.monotonic()
    for epoch in range(1, budget["epochs"] + 1):
        head.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(len(features[("x", "train")]), generator=generator).tolist()
        loss_sum, seen = 0.0, 0
        for start in range(0, len(permutation), budget["batch_size"]):
            chosen = permutation[start:start + budget["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            tokens = torch.from_numpy(np.asarray(features[("x", "train")][chosen], dtype=np.float32))
            labels = torch.from_numpy(targets["train"][chosen])
            loss = torch.nn.functional.cross_entropy(head(tokens), labels, weight=weight)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), budget["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            seen += len(chosen)
            loss_sum += float(loss.detach()) * len(chosen)
        if seen != len(features[("x", "train")]):
            raise ValueError("training dropped rows")
        head.eval()
        with torch.no_grad():
            val_p = torch.softmax(head(torch.from_numpy(np.asarray(features[("x", "val")], dtype=np.float32))
                                      ).double(), dim=1).numpy()
        metrics = macro_metrics(val_p, targets["val"], classes)
        if metrics["macro_f1_full"] > best:
            best, best_epoch, stale = metrics["macro_f1_full"], epoch, 0
            best_state = {key: value.clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "val_macro_f1": metrics["macro_f1_full"]})
        if stale >= budget["patience"]:
            break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_p = torch.softmax(head(torch.from_numpy(np.asarray(features[("x", "val")], dtype=np.float32))
                                  ).double(), dim=1).numpy()
    return {"metrics": macro_metrics(val_p, targets["val"], classes), "best_epoch": best_epoch,
            "history": history, "seconds": time.monotonic() - started,
            "parameters": width * classes + classes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nasa-cache", type=Path, required=True)
    parser.add_argument("--moonstone-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / PLAN_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    plan, budget = load_plan(args.plan)
    import h5py
    import torch
    from torch import nn
    torch.use_deterministic_algorithms(True)
    args.outdir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.nasa_cache / "manifest.json").read_text())
    ids = {split: manifest["splits"][split]["ids"] for split in ("train", "val")}
    geology = {split: np.asarray(manifest["splits"][split]["targets"], dtype=np.int64)
               for split in ("train", "val")}
    nasa = {split: np.asarray(np.memmap(args.nasa_cache / f"pretrained_{split}.f32", dtype=np.float32,
                                       mode="r", shape=tuple(manifest["splits"][split]["shape"])))
            for split in ("train", "val")}
    moon_pooled, moon_age = {}, {}
    for seed in budget["seeds"]:
        with h5py.File(args.moonstone_cache / f"context_seed{seed}.h5", "r") as handle:
            pooled = {split: np.asarray(handle[split]["pooled"][:], dtype=np.float32) for split in ("train", "val")}
            age = {split: np.asarray(handle[split]["age"][:], dtype=np.int64) for split in ("train", "val")}
            cache_ids = {split: [value.decode() if isinstance(value, bytes) else str(value)
                                 for value in handle[split]["sample_ids"][:]] for split in ("train", "val")}
        if any(cache_ids[split] != ids[split] for split in ("train", "val")):
            raise ValueError(f"Moonstone seed{seed} ordered IDs differ from the NASA manifest")
        moon_pooled[seed], moon_age[seed] = pooled, age
    feature_sets = {
        "moonstone_256": lambda seed: {("x", s): moon_pooled[seed][s] for s in ("train", "val")},
        "nasa_768": lambda seed: {("x", s): nasa[s] for s in ("train", "val")},
        "fusion_1024": lambda seed: {("x", s): np.concatenate([moon_pooled[seed][s], nasa[s]], axis=1)
                                     for s in ("train", "val")},
    }
    task_targets = {"geology": geology, "age": {split: moon_age[42][split] for split in ("train", "val")}}
    for seed in budget["seeds"]:
        for split in ("train", "val"):
            if not np.array_equal(moon_age[seed][split], moon_age[42][split]):
                raise ValueError("Age targets differ across Moonstone caches")
    classes = {"geology": 49, "age": 5}
    cells, started = {}, time.monotonic()
    for feature_name, build in feature_sets.items():
        for task, targets in task_targets.items():
            for head_name in ("linear", "linear_balanced"):
                runs = []
                for seed in budget["seeds"]:
                    features = build(seed)
                    runs.append(train_cell(head_name, features, targets, seed, budget, classes[task], torch, nn))
                key = f"{feature_name}/{task}/{head_name}"
                cells[key] = {"values": [run["metrics"]["macro_f1_full"] for run in runs],
                              "accuracy": [run["metrics"]["accuracy"] for run in runs],
                              "best_epochs": [run["best_epoch"] for run in runs],
                              "seconds": [round(run["seconds"], 3) for run in runs],
                              "parameters": runs[0]["parameters"]}
                print(f"{key}: {[round(v, 6) for v in cells[key]['values']]}", flush=True)
    summary = {"status": "FEATURE_FUSION_COMPLETE", "protocol": plan["protocol"], "budget": budget,
               "age_support": {split: np.bincount(moon_age[42][split], minlength=5).tolist()
                               for split in ("train", "val")},
               "cells": {key: {**value, "mean": float(np.mean(value["values"])),
                               "sample_std": float(np.std(value["values"], ddof=1))}
                         for key, value in cells.items()},
               "nasa_manifest_sha256": file_sha256(args.nasa_cache / "manifest.json"),
               "moonstone_cache_sha256": {seed: file_sha256(args.moonstone_cache / f"context_seed{seed}.h5")
                                          for seed in budget["seeds"]},
               "elapsed_seconds": time.monotonic() - started, "test_scored": False}
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for key, value in summary["cells"].items():
        print(f"{key}: mean={value['mean']:.6f} std={value['sample_std']:.6f}")


if __name__ == "__main__":
    main()
