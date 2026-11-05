"""Small paired-head training engine; never imports NASA or reads Test itself.

A caller supplies the authenticated Train/Val datasets and official backbone.
Checkpoints contain the head/adapters only; the frozen backbone is reconstructed
from its pinned source, random seed and optional verified checkpoint.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import file_sha256
from .experiment import probability_metrics, validate_budget
from .torch_adapter import TerrainTransfer


def write_new_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(text)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def task_state(model):
    return {name: {key: value.detach().cpu().clone() for key, value in getattr(model, name).state_dict().items()}
            for name in ("head", "adapters")}


def restore_task_state(model, state):
    if set(state) != {"head", "adapters"}:
        raise ValueError("checkpoint must contain exactly task head and input adapters")
    for name in state:
        getattr(model, name).load_state_dict(state[name], strict=True)


def task_digest(model):
    return state_digest({name + "." + key: value for name, state in task_state(model).items()
                         for key, value in state.items()})


def make_arm(factory, load_pretrained, *, seed, arm, adaptation):
    if arm not in ("random", "pretrained") or adaptation not in ("frozen", "input_affine"):
        raise ValueError("unsupported paired arm/adaptation")
    # Both arms consume exactly the same CPU initialization RNG before loading.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TerrainTransfer(factory(), adapt_inputs=adaptation == "input_affine")
    coverage = load_pretrained(model.backbone) if arm == "pretrained" else {"initialization": "random", "pretrained": False}
    return model, coverage


def prepare_device(name, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("only CPU or existing CUDA/ROCm device is supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested existing GPU unavailable")
        if torch.version.cuda and not torch.version.hip:
            if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
                raise RuntimeError("restart with CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
                raise RuntimeError("deterministic NVIDIA GEMM requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
        torch.cuda.init()
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return device


def release_device_memory(device, baseline_allocated=0):
    """Let the fresh replay process use memory reserved by the training process."""
    gc.collect()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        remaining = torch.cuda.memory_allocated(device)
        if remaining > baseline_allocated:
            raise RuntimeError("training GPU tensors remain live before fresh-process replay")
        return {"allocated_after_release": remaining, "baseline_allocated": baseline_allocated}
    return {"allocated_after_release": None, "baseline_allocated": None}


def _loader(dataset, batch_size, indices=None):
    return DataLoader(dataset, batch_size=batch_size, sampler=indices, shuffle=False,
                      num_workers=0, drop_last=False, pin_memory=False)


def _inputs(batch, device):
    return {key: batch[key].to(device, dtype=torch.float32) for key in ("dtm", "slope")}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, dataset, *, batch_size, device, split, output=None):
    if split not in ("train", "val"):
        raise ValueError("Test scoring is forbidden")
    model.eval()
    ids, targets, parts = [], [], []
    for batch in _loader(dataset, batch_size):
        probability = torch.softmax(model(_inputs(batch, device)).double(), dim=1).cpu().numpy()
        ids.extend(batch["sample_id"])
        targets.extend(batch["target"].numpy().tolist())
        parts.append(probability)
    if len(ids) != len(dataset) or not ids or len(set(ids)) != len(ids):
        raise ValueError("evaluation dropped/repeated rows")
    p, y = np.concatenate(parts), np.asarray(targets, dtype=np.int64)
    metrics = probability_metrics(p, y)
    if output is not None:
        with Path(output).open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "split", "target"] + [f"p_{i}" for i in range(49)])
            writer.writerows([sample_id, split, target, *row.tolist()] for sample_id, target, row in zip(ids, targets, p))
    return metrics


def train_arm(model, datasets, *, budget, seed, arm, device, outdir, identity, coverage):
    budget = validate_budget(budget)
    if set(datasets) != {"train", "val"} or arm not in ("random", "pretrained") or seed not in budget["seeds"]:
        raise ValueError("expected a declared Train/Val paired run")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    device = prepare_device(device, seed)
    model.to(device).float()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_task_sha256 = task_digest(model)
    backbone_sha256 = state_digest(model.backbone.state_dict())
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=budget["lr"], weight_decay=budget["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
    best, best_epoch, stale, best_state = -1.0, 0, 0, None
    history, total_visits = [], 0
    began = time.monotonic()
    for epoch in range(1, budget["epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(datasets["train"]), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        seen, loss_sum, order = 0, 0.0, []
        _sync(device)
        epoch_start = time.monotonic()
        for batch in _loader(datasets["train"], budget["batch_size"], permutation):
            optimizer.zero_grad(set_to_none=True)
            labels = batch["target"].to(device)
            loss = torch.nn.functional.cross_entropy(model(_inputs(batch, device)), labels)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise ValueError("missing/nonfinite trainable gradient")
            if any(parameter.grad is not None for parameter in model.backbone.parameters()):
                raise ValueError("frozen backbone unexpectedly received gradients")
            torch.nn.utils.clip_grad_norm_(parameters, budget["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            seen += len(labels)
            loss_sum += float(loss.detach()) * len(labels)
            order.extend(batch["sample_id"])
        _sync(device)
        if seen != len(datasets["train"]) or len(set(order)) != seen:
            raise ValueError("training dropped/repeated rows")
        train_seconds = time.monotonic() - epoch_start
        val = evaluate(model, datasets["val"], batch_size=budget["batch_size"], device=device, split="val")
        metric = val["macro_f1_full"]
        if metric > best:
            best, best_epoch, stale, best_state = metric, epoch, 0, task_state(model)
        else:
            stale += 1
        total_visits += seen
        history.append({"epoch": epoch, "train_rows": seen, "train_loss": loss_sum / seen,
                        "train_seconds": train_seconds, "val": val,
                        "training_order_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest()})
        # Preserve each completed epoch, including failure recovery evidence. No
        # hidden auto-resume/retraining policy; continuation is a separate action.
        torch.save({"identity": identity, "budget": budget, "epoch": epoch,
                    "task_state": task_state(model), "best_state": best_state,
                    "best_epoch": best_epoch, "best_metric": best, "history": history,
                    "backbone_sha256": backbone_sha256}, outdir / f"epoch_{epoch:03d}.pt")
        if stale >= budget["patience"]:
            break
    restore_task_state(model, best_state)
    selected_sha256 = task_digest(model)
    metrics, prediction_hashes = {}, {}
    for split, dataset in datasets.items():
        path = outdir / f"{split}_predictions.csv"
        metrics[split] = evaluate(model, dataset, batch_size=budget["batch_size"], device=device, split=split, output=path)
        prediction_hashes[split] = file_sha256(path)
    if state_digest(model.backbone.state_dict()) != backbone_sha256:
        raise ValueError("frozen backbone state changed during training")
    checkpoint = outdir / "selected.pt"
    torch.save({"identity": identity, "budget": budget, "task_state": task_state(model),
                "selected_task_sha256": selected_sha256, "backbone_sha256": backbone_sha256,
                "best_epoch": best_epoch, "best_metric": best}, checkpoint)
    report = {"status": "TRAINED_REPLAY_PENDING", "seed": seed, "arm": arm, "identity": identity,
              "budget": budget, "metrics": metrics, "history": history, "best_epoch": best_epoch,
              "best_metric": best, "initial_task_sha256": initial_task_sha256,
              "selected_task_sha256": selected_sha256, "backbone_sha256": backbone_sha256,
              "checkpoint_sha256": file_sha256(checkpoint), "prediction_sha256": prediction_hashes,
              "checkpoint_coverage": coverage, "test_scored": False, "training_visits": total_visits,
              "seconds": time.monotonic() - began, "torch_version": torch.__version__,
              "trainable_parameters": sum(parameter.numel() for parameter in parameters),
              "peak_device_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "memory_scope": "process high-water RSS; device peak includes model, batches and training"}
    write_new_json(outdir / "report.json", report)
    return report


def replay_arm(model, datasets, *, budget, seed, device, original, outdir, identity):
    original, outdir = Path(original), Path(outdir)
    report = json.loads((original / "report.json").read_text())
    checkpoint = original / "selected.pt"
    if report["identity"] != identity or report["budget"] != budget or file_sha256(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("replay run identity or checkpoint checksum mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["identity"] != identity or payload["budget"] != budget:
        raise ValueError("replay checkpoint identity differs")
    if state_digest(model.backbone.state_dict()) != payload["backbone_sha256"]:
        raise ValueError("reconstructed frozen backbone differs")
    restore_task_state(model, payload["task_state"])
    if task_digest(model) != payload["selected_task_sha256"]:
        raise ValueError("selected task state differs")
    device = prepare_device(device, seed)
    model.to(device).float()
    outdir.mkdir(parents=True, exist_ok=False)
    metrics = {}
    for split, dataset in datasets.items():
        path = outdir / f"{split}_predictions.csv"
        metrics[split] = evaluate(model, dataset, batch_size=budget["batch_size"], device=device, split=split, output=path)
        if file_sha256(path) != report["prediction_sha256"][split] or metrics[split] != report["metrics"][split]:
            raise ValueError("fresh-process prediction/metric replay mismatch")
    receipt = {"status": "PASS", "identity": identity, "prediction_sha256": report["prediction_sha256"],
               "selected_task_sha256": payload["selected_task_sha256"], "test_scored": False}
    write_new_json(outdir / "replay.json", receipt)
    return receipt
