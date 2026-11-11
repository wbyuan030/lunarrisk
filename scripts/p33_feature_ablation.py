#!/usr/bin/env python3
"""Compare frozen feature sources and head designs under one fixed budget.

Feature sets: verified NASA pretrained tokens, the identical-architecture random
arm, and the authenticated P3.2 Moonstone pooled tokens. Heads: linear, MLP and
linear with class-balanced cross-entropy. Every cell shares the pre-registered
budget, permutation rule and Val selection. Test is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lunarrisk.nasa_transfer.experiment import probability_metrics, validate_budget
from lunarrisk.nasa_transfer.contracts import file_sha256

PLAN_PATH = "protocols/p33_feature_ablation_plan.json"


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("protocol") != "P33_FEATURE_ABLATION_V1":
        raise ValueError("unexpected ablation plan")
    payload = {key: plan["budget"][key] for key in (
        "epochs", "patience", "batch_size", "lr", "weight_decay", "grad_clip", "seeds",
        "optimizer", "schedule", "selection", "test_scored")}
    payload["adaptation"] = "frozen"
    payload["precision"] = "fp32"
    budget = validate_budget(payload)
    return plan, budget


def load_nasa(cache_dir):
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    data = {}
    for arm in ("pretrained", "random"):
        for split in ("train", "val"):
            shape = tuple(manifest["splits"][split]["shape"])
            data[(arm, split)] = np.asarray(np.memmap(cache_dir / f"{arm}_{split}.f32",
                                                      dtype=np.float32, mode="r", shape=shape))
    ids = {split: manifest["splits"][split]["ids"] for split in ("train", "val")}
    targets = {split: np.asarray(manifest["splits"][split]["targets"], dtype=np.int64)
               for split in ("train", "val")}
    return data, ids, targets, manifest


def load_moonstone(cache_dir, seed):
    import h5py
    path = cache_dir / f"context_seed{seed}.h5"
    data, ids, targets = {}, {}, {}
    with h5py.File(path, "r") as handle:
        for split in ("train", "val"):
            data[("x", split)] = np.asarray(handle[split]["pooled"][:], dtype=np.float32)
            ids[split] = [value.decode() if isinstance(value, bytes) else str(value)
                          for value in handle[split]["sample_ids"][:]]
            targets[split] = np.asarray(handle[split]["geology"][:], dtype=np.int64)
    return data, ids, targets


def make_head(name, width, torch, nn):
    if name == "linear":
        return nn.Linear(width, 49)
    if name == "mlp":
        return nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 49))
    if name == "linear_balanced":
        return nn.Linear(width, 49)
    raise ValueError("unknown head")


def class_weights(targets, torch):
    counts = np.bincount(targets, minlength=49).astype(np.float64)
    weights = np.zeros(49, dtype=np.float64)
    np.divide(1.0, np.sqrt(counts), out=weights, where=counts > 0)
    weights = weights / weights[counts > 0].mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(head, features, targets, torch, device="cpu"):
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device))
        probability = torch.softmax(logits.double(), dim=1).cpu().numpy()
    return probability_metrics(probability, targets)


def train_cell(name, features, targets, seed, budget, torch, nn, device="cpu"):
    torch.manual_seed(seed)
    head = make_head(name, features[("x", "train")].shape[1], torch, nn).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(targets["train"], torch)
                                    if name == "linear_balanced" else None)
    optimizer = torch.optim.AdamW(head.parameters(), lr=budget["lr"],
                                  weight_decay=budget["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
    best, best_epoch, stale, history = -1.0, 0, 0, []
    started = time.monotonic()
    for epoch in range(1, budget["epochs"] + 1):
        head.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(len(features[("x", "train")]), generator=generator).tolist()
        loss_sum, seen, order = 0.0, 0, []
        for start in range(0, len(permutation), budget["batch_size"]):
            chosen = permutation[start:start + budget["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            tokens = torch.from_numpy(np.asarray(features[("x", "train")][chosen], dtype=np.float32)).to(device)
            labels = torch.from_numpy(targets["train"][chosen]).to(device)
            loss = criterion(head(tokens), labels) if criterion is not None else \
                torch.nn.functional.cross_entropy(head(tokens), labels)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), budget["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            seen += len(chosen)
            loss_sum += float(loss.detach()) * len(chosen)
            order.extend(str(index) for index in chosen)
        if seen != len(features[("x", "train")]) or len(set(order)) != seen:
            raise ValueError("training dropped or repeated rows")
        metrics = evaluate(head, features[("x", "val")], targets["val"], torch, device)
        if metrics["macro_f1_full"] > best:
            best, best_epoch, stale = metrics["macro_f1_full"], epoch, 0
            best_state = {key: value.clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": loss_sum / seen,
                        "val_macro_f1": metrics["macro_f1_full"], "val_accuracy": metrics["accuracy"]})
        if stale >= budget["patience"]:
            break
    head.load_state_dict(best_state)
    final = evaluate(head, features[("x", "val")], targets["val"], torch, device)
    return {"metrics": final, "best_epoch": best_epoch, "history": history,
            "seconds": time.monotonic() - started,
            "parameters": int(sum(p.numel() for p in head.parameters()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nasa-cache", type=Path, required=True)
    parser.add_argument("--moonstone-cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / PLAN_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    plan, budget = load_plan(args.plan)
    import torch
    from torch import nn
    torch.use_deterministic_algorithms(True)
    args.outdir.mkdir(parents=True, exist_ok=False)
    nasa, nasa_ids, nasa_targets, nasa_manifest = load_nasa(args.nasa_cache)
    cells = {}
    started = time.monotonic()
    for feature_name in ("nasa_pretrained", "nasa_random"):
        key = "pretrained" if feature_name == "nasa_pretrained" else "random"
        features = {("x", split): nasa[(key, split)] for split in ("train", "val")}
        ids, targets = nasa_ids, nasa_targets
        for head_name in ("linear", "mlp", "linear_balanced"):
            runs = [train_cell(head_name, features, targets, seed, budget, torch, nn, args.device)
                    for seed in budget["seeds"]]
            cells[f"{feature_name}/{head_name}"] = {
                "values": [run["metrics"]["macro_f1_full"] for run in runs],
                "accuracy": [run["metrics"]["accuracy"] for run in runs],
                "best_epochs": [run["best_epoch"] for run in runs],
                "seconds": [round(run["seconds"], 3) for run in runs],
                "parameters": runs[0]["parameters"],
                "history": [run["history"] for run in runs]}
            print(f"{feature_name}/{head_name}: {[round(v, 6) for v in cells[f'{feature_name}/{head_name}']['values']]}", flush=True)
    for seed in budget["seeds"]:
        moon, moon_ids, moon_targets = load_moonstone(args.moonstone_cache, seed)
        if moon_ids["train"] != nasa_ids["train"] or moon_ids["val"] != nasa_ids["val"] \
                or not np.array_equal(moon_targets["train"], nasa_targets["train"]) \
                or not np.array_equal(moon_targets["val"], nasa_targets["val"]):
            raise ValueError(f"Moonstone cache seed{seed} cohort differs from the NASA manifest")
        for head_name in ("linear", "mlp", "linear_balanced"):
            run = train_cell(head_name, moon, moon_targets, seed, budget, torch, nn, args.device)
            cell = cells.setdefault(f"moonstone/{head_name}", {"values": [], "accuracy": [],
                                                               "best_epochs": [], "seconds": [],
                                                               "parameters": run["parameters"], "history": []})
            cell["values"].append(run["metrics"]["macro_f1_full"])
            cell["accuracy"].append(run["metrics"]["accuracy"])
            cell["best_epochs"].append(run["best_epoch"])
            cell["seconds"].append(round(run["seconds"], 3))
            cell["history"].append(run["history"])
        print(f"moonstone seed{seed}: {[round(v, 6) for v in cells['moonstone/linear']['values']]}", flush=True)
    summary = {"status": "FEATURE_ABLATION_COMPLETE", "protocol": plan["protocol"], "budget": budget,
               "cells": {key: {"values": value["values"],
                               "mean": float(np.mean(value["values"])),
                               "sample_std": float(np.std(value["values"], ddof=1)) if len(value["values"]) > 1 else None,
                               "accuracy_mean": float(np.mean(value["accuracy"])),
                               "best_epochs": value["best_epochs"], "seconds": value["seconds"],
                               "parameters": value["parameters"], "history": value["history"]}
                         for key, value in cells.items()},
               "nasa_manifest_sha256": file_sha256(args.nasa_cache / "manifest.json"),
               "moonstone_cache_sha256": {seed: file_sha256(args.moonstone_cache / f"context_seed{seed}.h5")
                                          for seed in budget["seeds"]},
               "elapsed_seconds": time.monotonic() - started,
               "test_scored": False}
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for key, value in summary["cells"].items():
        print(f"{key}: mean={value['mean']:.6f} std={value['sample_std']:.6f} params={value['parameters']}")


if __name__ == "__main__":
    main()
