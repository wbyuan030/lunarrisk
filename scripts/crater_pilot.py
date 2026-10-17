#!/usr/bin/env python3
"""Engineering smoke test on the official NAC crater detection dataset.

Target: high annotated-crater count, defined by the *median train count*:
    unsafe_proxy = count(standard annotations) >= median(train counts)

This is a proxy for annotation density, NOT landing safety. Only the
``standard`` annotation protocol is used; ``large`` boxes are excluded.
The official split is used unchanged (it is a box-level split, not a site-OOD
split); the report records site overlap so results are not overstated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SPLITS = ("train", "val", "test")
RISK_COVERAGES = (0.2, 0.4, 0.6, 0.8)


class TinyCNN(nn.Module):
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


def find_root(data: Path) -> Path:
    candidates = [data] + [p for p in data.rglob("*") if p.is_dir()]
    for candidate in candidates:
        if (candidate / "images").is_dir() and any(candidate.glob("annotations*.json")):
            return candidate
    raise FileNotFoundError(
        f"no directory containing images/ and annotations*.json under {data}; "
        "pass --data pointing at the extracted crater dataset root"
    )


def annotation_path(root: Path, split: str) -> Path:
    min5 = root / f"annotations_min5px_{split}.json"
    if min5.exists():
        return min5
    full = root / f"annotations_{split}.json"
    if full.exists():
        return full
    raise FileNotFoundError(f"missing annotations_min5px_{split}.json / annotations_{split}.json in {root}")


def protocol_of(image: dict) -> str:
    label_set = str(image.get("label_set", "")).lower()
    name = Path(image.get("file_name", "")).name.lower()
    if label_set in {"standard", "large"}:
        return label_set
    if "__standard_" in name:
        return "standard"
    if "__large_" in name:
        return "large"
    return "unknown"


def load_split(root: Path, split: str) -> tuple[list[dict], dict]:
    path = annotation_path(root, split)
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(int(a["image_id"]) for a in data.get("annotations", []))
    images = []
    for image in data.get("images", []):
        protocol = protocol_of(image)
        if protocol != "standard":
            continue
        name = Path(image["file_name"]).name
        image_path = root / "images" / name
        if not image_path.exists():
            raise FileNotFoundError(f"image referenced by {path.name} is missing: {image_path}")
        images.append({
            "path": image_path,
            "count": int(counts.get(int(image["id"]), 0)),
            "site": image.get("site") or site_from_name(name),
            "split": split,
            "protocol": protocol,
        })
    if not images:
        raise RuntimeError(f"{path.name}: zero standard-protocol images; refusing an empty run")
    meta = {
        "annotation_file": path.name,
        "images_total": len(data.get("images", [])),
        "standard_images": len(images),
        "annotations_total": len(data.get("annotations", [])),
        "sites": sorted({img["site"] for img in images}),
    }
    return images, meta


def site_from_name(name: str) -> str:
    return name.split("__")[0]


def load_image(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"{path}: expected 2-D image, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8, got {array.dtype}")
    return array.astype(np.float32) / 255.0


def batches(items: list[dict], batch_size: int, rng: np.random.Generator | None = None):
    order = np.arange(len(items))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        x = np.stack([load_image(items[i]["path"]) for i in idx])[:, None, :, :]
        y = np.array([items[i]["label"] for i in idx], dtype=np.float32)
        yield torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def predict(model: nn.Module, items: list[dict], device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    probs = []
    for x, _ in batches(items, batch_size):
        logits = model(x.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    brier = float(np.mean((p - y) ** 2))
    nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    accuracy = float(np.mean((p >= 0.5).astype(np.int64) == y))
    # Rank-based AUC, ties handled by average ranks.
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    positives = int(y.sum())
    negatives = len(y) - positives
    auc = None
    if positives and negatives:
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    return {"brier": brier, "nll": nll, "accuracy_at_0_5": accuracy, "auc": auc}


def risk_coverage(y: np.ndarray, p: np.ndarray) -> list[dict]:
    order = np.argsort(p, kind="mergesort")  # accept lowest predicted risk first
    unsafe_total = int(y.sum())
    n = len(y)
    rows = []
    for coverage in [i / 20 for i in range(1, 21)]:
        k = max(1, min(n, int(round(coverage * n))))
        accepted = order[:k]
        unsafe_accepted = int(y[accepted].sum())
        rows.append({
            "coverage_requested": round(coverage, 2),
            "coverage_actual": round(k / n, 4),
            "accepted_n": k,
            "test_n": n,
            "unsafe_accepted_n": unsafe_accepted,
            "unsafe_total": unsafe_total,
            "UAR": (unsafe_accepted / k) if k else None,
            "HMR": (unsafe_accepted / unsafe_total) if unsafe_total else None,
            "accepted_mean_predicted_risk": float(p[accepted].mean()),
            "accepted_max_predicted_risk": float(p[accepted].max()),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/crater"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("outputs/crater_pilot"))
    args = parser.parse_args(argv)

    root = find_root(args.data)
    split_items, split_meta = {}, {}
    for split in SPLITS:
        split_items[split], split_meta[split] = load_split(root, split)
    train = split_items["train"]
    threshold = float(np.median([item["count"] for item in train]))
    for split in SPLITS:
        for item in split_items[split]:
            item["label"] = int(item["count"] >= threshold)
    if len({item["label"] for item in train}) < 2:
        raise RuntimeError(f"degenerate train labels at median={threshold}; check annotation counts")
    train_positive = float(np.mean([item["label"] for item in train]))
    test_y = np.array([item["label"] for item in split_items["test"]], dtype=np.float32)

    unique_sites = {split: set(split_meta[split]["sites"]) for split in SPLITS}
    site_overlap = {
        "train_test": sorted(unique_sites["train"] & unique_sites["test"]),
        "train_val": sorted(unique_sites["train"] & unique_sites["val"]),
        "val_test": sorted(unique_sites["val"] & unique_sites["test"]),
    }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in batches(train, args.batch, rng):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_p = predict(model, split_items["val"], device, args.batch)
        val_y = np.array([item["label"] for item in split_items["val"]], dtype=np.float32)
        val = binary_metrics(val_y, val_p)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_brier": val["brier"]})
        print(f"epoch {epoch}/{args.epochs} train_loss={history[-1]['train_loss']:.4f} "
              f"val_brier={val['brier']:.4f} val_auc={val['auc']}", flush=True)

    test_p = predict(model, split_items["test"], device, args.batch)
    test_metrics = binary_metrics(test_y, test_p)
    test_metrics["brier_constant_baseline"] = float(np.mean((train_positive - test_y) ** 2))
    coverage_rows = risk_coverage(test_y, test_p)
    target_rows = [row for row in coverage_rows if any(abs(row["coverage_requested"] - c) < 1e-9 for c in RISK_COVERAGES)]

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out / "predictions.npz",
        paths=np.array([str(item["path"]) for split in SPLITS for item in split_items[split]]),
        split=np.array([split for split in SPLITS for _ in split_items[split]]),
        counts=np.array([item["count"] for split in SPLITS for item in split_items[split]]),
        labels=np.array([item["label"] for split in SPLITS for item in split_items[split]]),
        probs=np.concatenate([
            predict(model, split_items[s], device, args.batch) for s in SPLITS
        ]),
        sites=np.array([item["site"] for split in SPLITS for item in split_items[split]]),
    )
    with (args.out / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    metrics = {
        "status": "PASS",
        "dataset_root": str(root),
        "repo_revision": _dataset_revision(),
        "protocol": "standard only; large excluded",
        "label_rule": "high annotated-crater count: count >= median(train standard counts)",
        "label_threshold": threshold,
        "note": "Engineering proxy, NOT landing safety. Official box-level split, not site-OOD.",
        "seed": args.seed,
        "device": device_name,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "hyperparameters": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "optimizer": "Adam"},
        "splits": split_meta,
        "site_overlap": site_overlap,
        "train_positive_rate": train_positive,
        "history": history,
        "test": test_metrics,
        "risk_coverage_at": target_rows,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_metrics, "threshold": threshold,
                      "split_sizes": {s: len(split_items[s]) for s in SPLITS}}, indent=2))
    print(f"wrote {args.out}")
    return 0


def _dataset_revision() -> str | None:
    audit = Path("outputs/hf_audit_crater.json")
    if not audit.exists():
        return None
    try:
        return json.loads(audit.read_text(encoding="utf-8"))["listed"]["revision"]
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
