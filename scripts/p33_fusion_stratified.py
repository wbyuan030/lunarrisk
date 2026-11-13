#!/usr/bin/env python3
"""Latitude-stratified fusion check plus a learned two-block scalar gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lunarrisk.nasa_transfer.contracts import file_sha256
from p33_feature_fusion import macro_metrics

PLAN_PATH = "protocols/p33_fusion_stratified_plan.json"
BANDS = (("equatorial", 9.0, 30.0), ("mid", 30.0, 60.0), ("polar", 60.0, 90.0))


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("protocol") != "P33_FUSION_STRATIFIED_V1":
        raise ValueError("unexpected stratified plan")
    if plan["budget"]["seeds"] != [42, 43, 44] or plan["budget"]["test_scored"] is not False:
        raise ValueError("plan must declare three seeds and no Test scoring")
    return plan, plan["budget"]


def load_latitudes(metadata_path):
    export = json.loads(Path(metadata_path).read_text())["metadata"]
    return {str(pid): float(lat) for pid, lat in
            zip(export["metadata/patch_id"], export["metadata/center_lat"])}


def make_model(width, gated, torch, nn):
    class Gated(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.weight = torch.nn.Parameter(torch.ones(2))

        def forward(self, x):
            first = x.shape[1] - 768
            return self.inner(torch.cat([x[:, :first] * self.weight[0], x[:, first:] * self.weight[1]], dim=1))

    head = nn.Linear(width, 49)
    return Gated(head) if gated else head


def train_cell(features, targets, seed, budget, gated, torch, nn):
    torch.manual_seed(seed)
    width = features[("x", "train")].shape[1]
    model = make_model(width, gated, torch, nn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget["lr"], weight_decay=budget["weight_decay"],
                                  betas=(0.9, 0.999), eps=1e-8)
    best, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, budget["epochs"] + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(len(features[("x", "train")]), generator=generator).tolist()
        for start in range(0, len(permutation), budget["batch_size"]):
            chosen = permutation[start:start + budget["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            tokens = torch.from_numpy(np.asarray(features[("x", "train")][chosen], dtype=np.float32))
            labels = torch.from_numpy(targets["train"][chosen])
            loss = torch.nn.functional.cross_entropy(model(tokens), labels)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), budget["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(torch.from_numpy(
                np.asarray(features[("x", "val")], dtype=np.float32))).double(), dim=1).numpy()
        metric = macro_metrics(probability, targets["val"], 49)["macro_f1_full"]
        if metric > best:
            best, best_epoch, stale = metric, epoch, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            best_probability = probability
        else:
            stale += 1
        if stale >= budget["patience"]:
            break
    model.load_state_dict(best_state)
    weights = None
    if gated:
        weights = [float(v) for v in model.weight.detach().cpu().numpy()]
    return {"probability": best_probability, "best_epoch": best_epoch, "weights": weights}


def band_metrics(probability, targets, latitudes):
    rows = {}
    for name, low, high in BANDS:
        members = (latitudes >= low) & (latitudes < high)
        if not members.any():
            rows[name] = None
            continue
        rows[name] = macro_metrics(probability[members], targets[members], 49)
    rows["overall"] = macro_metrics(probability, targets, 49)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nasa-cache", type=Path, required=True)
    parser.add_argument("--moonstone-cache", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
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
    ids = {split: manifest["splits"][split]["ids"] for split in ("train", "val")}
    geology = {split: np.asarray(manifest["splits"][split]["targets"], dtype=np.int64)
               for split in ("train", "val")}
    nasa = {split: np.asarray(np.memmap(args.nasa_cache / f"pretrained_{split}.f32", dtype=np.float32,
                                       mode="r", shape=tuple(manifest["splits"][split]["shape"])))
            for split in ("train", "val")}
    lat_map = load_latitudes(args.metadata)
    latitudes = np.array([lat_map[sample] for sample in ids["val"]])
    moon = {}
    for seed in budget["seeds"]:
        with h5py.File(args.moonstone_cache / f"context_seed{seed}.h5", "r") as handle:
            pooled = {split: np.asarray(handle[split]["pooled"][:], dtype=np.float32) for split in ("train", "val")}
            cache_ids = {split: [v.decode() if isinstance(v, bytes) else str(v)
                                 for v in handle[split]["sample_ids"][:]] for split in ("train", "val")}
        if any(cache_ids[split] != ids[split] for split in ("train", "val")):
            raise ValueError(f"Moonstone seed{seed} IDs differ from the NASA manifest")
        moon[seed] = pooled
    builders = {
        "moonstone_256": lambda seed: {("x", s): moon[seed][s] for s in ("train", "val")},
        "fusion_1024": lambda seed: {("x", s): np.concatenate([moon[seed][s], nasa[s]], axis=1)
                                     for s in ("train", "val")},
    }
    cells, started = {}, time.monotonic()
    for name, build in builders.items():
        for gated in ((False, True) if name == "fusion_1024" else (False,)):
            tag = name + ("/gated" if gated else "")
            runs = []
            for seed in budget["seeds"]:
                features = build(seed)
                runs.append(train_cell(features, geology, seed, budget, gated, torch, nn))
            cells[tag] = {"bands": [band_metrics(run["probability"], geology["val"], latitudes) for run in runs],
                          "best_epochs": [run["best_epoch"] for run in runs],
                          "weights": [run["weights"] for run in runs]}
            print(f"{tag}: overall {[round(bands['overall']['macro_f1_full'], 6) for bands in cells[tag]['bands']]}", flush=True)
    summary = {"status": "FUSION_STRATIFIED_COMPLETE", "protocol": plan["protocol"], "budget": budget,
               "band_sizes": {name: int(((latitudes >= low) & (latitudes < high)).sum())
                              for name, low, high in BANDS},
               "cells": {tag: {"overall_mean": float(np.mean([b["overall"]["macro_f1_full"] for b in value["bands"]])),
                               "bands": {name: {"values": [b[name]["macro_f1_full"] for b in value["bands"]],
                                                "mean": float(np.mean([b[name]["macro_f1_full"] for b in value["bands"]]))}
                                         for name, _, _ in BANDS},
                               "best_epochs": value["best_epochs"], "weights": value["weights"]}
                         for tag, value in cells.items()},
               "nasa_manifest_sha256": file_sha256(args.nasa_cache / "manifest.json"),
               "metadata_sha256": file_sha256(args.metadata),
               "elapsed_seconds": time.monotonic() - started, "test_scored": False}
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for tag, value in summary["cells"].items():
        print(tag, {k: round(v["mean"], 6) for k, v in value["bands"].items()}, "weights", value["weights"])


if __name__ == "__main__":
    main()
