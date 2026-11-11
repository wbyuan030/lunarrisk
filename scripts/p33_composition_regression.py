#!/usr/bin/env python3
"""Regress per-patch FeO and TiO2 from frozen features under one fixed budget.

Targets come from the published LPGRS aligned rasters sampled only over the
fixed Train/Val tiles; features come from the authenticated NASA and Moonstone
caches. Test is never read.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunarrisk.nasa_transfer.contracts import file_sha256

PLAN_PATH = "protocols/p33_composition_regression_plan.json"


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("protocol") != "P33_COMPOSITION_REGRESSION_V1":
        raise ValueError("unexpected regression plan")
    budget = plan["budget"]
    if budget["seeds"] != [42, 43, 44] or budget["test_scored"] is not False:
        raise ValueError("plan must declare three seeds and no Test scoring")
    return plan, budget


def regression_metrics(prediction, target, mean, std):
    """Return original-scale RMSE/MAE and R2 for one target."""
    prediction = np.asarray(prediction, dtype=np.float64) * std + mean
    target = np.asarray(target, dtype=np.float64) * std + mean
    residual = prediction - target
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    total = float(np.sum((target - target.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / total) if total > 0 else None
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_cell(features, targets, scales, seed, budget, torch, nn):
    torch.manual_seed(seed)
    width = features[("x", "train")].shape[1]
    head = nn.Linear(width, 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=budget["lr"], weight_decay=budget["weight_decay"],
                                  betas=(0.9, 0.999), eps=1e-8)
    best, best_epoch, stale, history = np.inf, 0, 0, []
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
            loss = torch.nn.functional.mse_loss(head(tokens), labels)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), budget["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            seen += len(chosen)
            loss_sum += float(loss.detach()) * len(chosen)
        head.eval()
        with torch.no_grad():
            prediction = head(torch.from_numpy(np.asarray(features[("x", "val")], dtype=np.float32))).numpy()
        metrics = {"feo": regression_metrics(prediction[:, 0], targets["val"][:, 0], *scales["feo"]),
                   "tio2": regression_metrics(prediction[:, 1], targets["val"][:, 1], *scales["tio2"])}
        score = 0.5 * (metrics["feo"]["rmse"] / scales["feo"][1] + metrics["tio2"]["rmse"] / scales["tio2"][1])
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            best_state = {key: value.clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "val_feo_rmse": metrics["feo"]["rmse"],
                        "val_tio2_rmse": metrics["tio2"]["rmse"]})
        if stale >= budget["patience"]:
            break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        prediction = head(torch.from_numpy(np.asarray(features[("x", "val")], dtype=np.float32))).numpy()
    return {"metrics": {"feo": regression_metrics(prediction[:, 0], targets["val"][:, 0], *scales["feo"]),
                        "tio2": regression_metrics(prediction[:, 1], targets["val"][:, 1], *scales["tio2"])},
            "best_epoch": best_epoch, "history": history, "seconds": time.monotonic() - started,
            "parameters": width * 2 + 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nasa-cache", type=Path, required=True)
    parser.add_argument("--moonstone-cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / PLAN_PATH)
    args = parser.parse_args()
    plan, budget = load_plan(args.plan)
    import h5py
    import torch
    from torch import nn
    torch.use_deterministic_algorithms(True)
    args.outdir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.nasa_cache / "manifest.json").read_text())
    nasa = {split: np.asarray(np.memmap(args.nasa_cache / f"pretrained_{split}.f32", dtype=np.float32,
                                       mode="r", shape=tuple(manifest["splits"][split]["shape"])))
            for split in ("train", "val")}
    labels = {"train": {}, "val": {}}
    with args.labels.open(newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["split"]][row["sample_id"]] = (float(row["feo_mean"]), float(row["tio2_mean"]))
    for split in ("train", "val"):
        ids = manifest["splits"][split]["ids"]
        if [ids[i] for i in range(len(ids))] != ids or any(sample not in labels[split] for sample in ids):
            raise ValueError("label coverage differs from the cohort manifest")
    raw = {split: np.asarray([labels[split][sample] for sample in manifest["splits"][split]["ids"]])
           for split in ("train", "val")}
    scales = {"feo": (float(raw["train"][:, 0].mean()), float(raw["train"][:, 0].std())),
              "tio2": (float(raw["train"][:, 1].mean()), float(raw["train"][:, 1].std()))}
    targets = {split: (raw[split] - np.array([scales["feo"][0], scales["tio2"][0]]))
               / np.array([scales["feo"][1], scales["tio2"][1]]) for split in ("train", "val")}
    targets = {split: targets[split].astype(np.float64) for split in targets}
    moon = {}
    for seed in budget["seeds"]:
        with h5py.File(args.moonstone_cache / f"context_seed{seed}.h5", "r") as handle:
            pooled = {split: np.asarray(handle[split]["pooled"][:], dtype=np.float32) for split in ("train", "val")}
            cache_ids = {split: [v.decode() if isinstance(v, bytes) else str(v)
                                 for v in handle[split]["sample_ids"][:]] for split in ("train", "val")}
        if any(cache_ids[split] != manifest["splits"][split]["ids"] for split in ("train", "val")):
            raise ValueError(f"Moonstone seed{seed} IDs differ from the NASA manifest")
        moon[seed] = pooled
    builders = {
        "moonstone_256": lambda seed: {("x", s): moon[seed][s] for s in ("train", "val")},
        "nasa_768": lambda seed: {("x", s): nasa[s] for s in ("train", "val")},
        "fusion_1024": lambda seed: {("x", s): np.concatenate([moon[seed][s], nasa[s]], axis=1)
                                     for s in ("train", "val")},
    }
    cells, started = {}, time.monotonic()
    for name, build in builders.items():
        runs = [train_cell(build(seed), targets, scales, seed, budget, torch, nn) for seed in budget["seeds"]]
        cells[name] = {"feo_rmse": [run["metrics"]["feo"]["rmse"] for run in runs],
                       "tio2_rmse": [run["metrics"]["tio2"]["rmse"] for run in runs],
                       "feo_r2": [run["metrics"]["feo"]["r2"] for run in runs],
                       "tio2_r2": [run["metrics"]["tio2"]["r2"] for run in runs],
                       "best_epochs": [run["best_epoch"] for run in runs],
                       "parameters": runs[0]["parameters"]}
        print(f"{name}: feo_rmse={[round(v, 5) for v in cells[name]['feo_rmse']]} "
              f"tio2_rmse={[round(v, 5) for v in cells[name]['tio2_rmse']]}", flush=True)
    baselines = {}
    for target, index in (("feo", 0), ("tio2", 1)):
        baseline = np.full(len(raw["val"]), raw["train"][:, index].mean())
        baselines[target] = regression_metrics(baseline, raw["val"][:, index], 0.0, 1.0)
    summary = {"status": "COMPOSITION_REGRESSION_COMPLETE", "protocol": plan["protocol"], "budget": budget,
               "scales": {key: {"train_mean": value[0], "train_std": value[1]} for key, value in scales.items()},
               "train_mean_baseline": baselines,
               "cells": {name: {**value,
                                "feo_rmse_mean": float(np.mean(value["feo_rmse"])),
                                "feo_rmse_std": float(np.std(value["feo_rmse"], ddof=1)),
                                "tio2_rmse_mean": float(np.mean(value["tio2_rmse"])),
                                "tio2_rmse_std": float(np.std(value["tio2_rmse"], ddof=1))}
                         for name, value in cells.items()},
               "label_rows": int(len(raw["train"]) + len(raw["val"])),
               "labels_sha256": file_sha256(args.labels),
               "nasa_manifest_sha256": file_sha256(args.nasa_cache / "manifest.json"),
               "elapsed_seconds": time.monotonic() - started, "test_scored": False}
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: (round(v["feo_rmse_mean"], 5), round(v["tio2_rmse_mean"], 5)) for k, v in summary["cells"].items()}))
    print("baseline", {k: round(v["rmse"], 5) for k, v in baselines.items()})


if __name__ == "__main__":
    main()
