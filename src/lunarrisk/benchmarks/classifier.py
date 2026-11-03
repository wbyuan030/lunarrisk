"""Bounded-memory classification primitives for the P3.1 derived Moonstone track."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .moonstone import MoonstoneHDF5Pilot, CHANNELS, select_split


def stream_sha256(path, chunk_bytes=4*1024*1024):
    if chunk_bytes < 1:
        raise ValueError("hash chunk must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            block = source.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


class ClassifierDataset(MoonstoneHDF5Pilot):
    def __init__(self, path, stats, mapping, task="age", split=0, track="official_stored"):
        import h5py
        if split not in (0, 1):
            raise ValueError("P31 runner permits Train/Val only; Test scoring is a separate gate")
        with h5py.File(path, "r") as handle:
            n = len(handle["patches"])
            keys = ("patch_id", "row_idx", "col_idx", "split", "center_lat",
                    "geology/dominant_unit_name", "geology/dominant_unit_id")
            if handle["patches"].dtype != np.dtype("float32"):
                raise ValueError("float32 pixels required")
            if any(handle["metadata/"+key].shape != (n,) for key in keys):
                raise ValueError("metadata lengths/shapes disagree with patches")
        super().__init__(path, stats, mapping, task, split, track)
        test = select_split(self.splits, self.latitudes, 2, track)
        if test[self.indices].any() or not len(self):
            raise ValueError("empty split or Test index selected")
        classes = 5 if task == "age" else 49
        if not ((self.targets >= 0) & (self.targets < classes)).all():
            raise ValueError("invalid classifier target")
        self.ordered_ids_hash = ids_sha256(self.ids[self.indices])

    def __getitem__(self, index):
        # Only the selected row reaches the h5py pixel read; metadata may include all split IDs.
        row = super().__getitem__(index)
        return {"pixels": row["pixels"], "target": row["target"], "sample_id": row["sample_id"]}


class MetadataTargets:
    """Train-prior evaluation uses labels/IDs only; it never touches the pixel dataset."""
    def __init__(self, dataset):
        self.targets = dataset.targets
        self.ids = dataset.ids[dataset.indices]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {"pixels": np.float32(0), "target": self.targets[index], "sample_id": str(self.ids[index])}


class ScratchClassifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(28, 64, 7, stride=4, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(256, classes)

    def forward(self, pixels):
        return self.classifier(self.features(pixels))


def make_loader(dataset, batch_size, shuffle, generator, workers=0):
    if batch_size < 1 or workers < 0:
        raise ValueError("invalid DataLoader settings")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=workers, drop_last=False, pin_memory=False)


def classification_metrics(confusion):
    matrix = np.asarray(confusion, dtype=np.int64)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    tp = np.diag(matrix)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(len(tp)), where=(support+predicted)>0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support>0)
    return {"accuracy": float(tp.sum()/max(1, support.sum())), "macro_f1_full": float(f1.mean()),
            "macro_f1_supported": float(f1[support>0].mean()) if (support>0).any() else 0.0,
            "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
            "support": support.tolist(), "predicted_counts": predicted.tolist(),
            "confusion": matrix.tolist(), "absent_class_policy": "full K, absent F1/recall=0; supported macro secondary"}


@torch.no_grad()
def evaluate(model, loader, device, classes, output=None, split="val", precision="fp32", prior=None):
    model.eval()
    confusion = np.zeros((classes, classes), dtype=np.int64)
    n, nll, brier = 0, 0.0, 0.0
    max_batch = 0
    handle = Path(output).open("w", newline="") if output is not None else None
    writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "target"]+[f"p_{i}" for i in range(classes)]) if handle else None
    if writer:
        writer.writeheader()
    try:
        for batch in loader:
            x, y = batch["pixels"].to(device), batch["target"].to(device)
            if prior is None:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision=="bf16"):
                    logits = model(x)
                logp = torch.log_softmax(logits.float(), dim=-1)
                probability = logp.exp()
            else:
                probability = torch.as_tensor(prior, device=device, dtype=torch.float32).expand(len(y), -1)
                logp = probability.log()
            if not torch.isfinite(probability).all():
                raise ValueError("nonfinite probabilities")
            selected = -logp.gather(1, y[:, None]).sum()
            nll += float(selected)
            target = torch.nn.functional.one_hot(y, classes)
            brier += float(((probability-target)**2).sum())
            truth, prediction = y.cpu().numpy(), probability.argmax(-1).cpu().numpy()
            np.add.at(confusion, (truth, prediction), 1)
            max_batch = max(max_batch, len(y))
            n += len(y)
            if writer:
                for sample, label, row in zip(batch["sample_id"], truth.tolist(), probability.cpu().tolist()):
                    writer.writerow({"sample_id": sample, "split": split, "target": label,
                                     **{f"p_{i}": p for i, p in enumerate(row)}})
    finally:
        if handle:
            handle.close()
    result = classification_metrics(confusion)
    result.update(rows=n, nll=nll/n if n and math.isfinite(nll) else None,
                  nll_is_infinite=not math.isfinite(nll), multiclass_brier_sum=brier/max(1,n),
                  largest_observed_batch=max_batch)
    return result


def seed_all(seed):
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    raw = state["numpy"]
    np.random.set_state((raw[0], np.asarray(raw[1], dtype=np.uint32), *raw[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_checkpoint(path, payload):
    path = Path(path)
    partial = path.with_suffix(path.suffix+".partial")
    # The caller owns an exclusively created run directory; never targets a legacy run.
    with partial.open("xb") as output:
        torch.save(payload, output)
    partial.replace(path)


def learning_rate(step, total_steps, batches_per_epoch, base):
    warmup = min(batches_per_epoch*2, total_steps//10)
    if step < warmup:
        return base*step/max(1,warmup)
    progress = (step-warmup)/max(1,total_steps-warmup)
    return base*0.5*(1+math.cos(math.pi*progress))


@torch.no_grad()
def export_features(model, datasets, path, device, batch_size, provenance=None):
    import h5py
    model.eval()
    with h5py.File(path, "x") as output:
        output.attrs["protocol"] = "P31_DERIVED_FROZEN_FEATURES"
        output.attrs["weights_sha256"] = state_fingerprint(cpu_state(model))
        output.attrs["feature_dtype"] = "float32"
        output.attrs["compute_precision"] = "fp32; autocast disabled"
        output.attrs["channel_names"] = json.dumps(CHANNELS)
        output.attrs["provenance"] = json.dumps(provenance or {}, sort_keys=True)
        output.attrs["encoder_frozen_at_export"] = True
        for name, dataset in datasets.items():
            group = output.create_group(name)
            n = len(dataset)
            group.create_dataset("features", (n,256), dtype="float32", chunks=(min(n,batch_size),256))
            group.create_dataset("targets", (n,), dtype="int64")
            group.create_dataset("sample_ids", (n,), dtype=h5py.string_dtype())
            offset = 0
            loader = make_loader(dataset,batch_size,False,torch.Generator().manual_seed(0))
            for batch in loader:
                values = model.features(batch["pixels"].to(device)).float().cpu().numpy()
                end = offset+len(values)
                group["features"][offset:end] = values
                group["targets"][offset:end] = batch["target"].numpy()
                group["sample_ids"][offset:end] = batch["sample_id"]
                offset = end
            group.attrs["ordered_ids_sha256"] = dataset.ordered_ids_hash


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
