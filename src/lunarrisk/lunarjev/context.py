"""Frozen P31 visual context; labels never enter the encoder."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from lunarrisk.benchmarks.classifier import ids_sha256


class FrozenContext(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.encoder = classifier.features[:-2]
        self.encoder.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, pixels):
        features = self.encoder(pixels)
        if features.shape[1:] != (256, 16, 16):
            raise ValueError("expected audited 256x16x16 spatial feature map")
        tokens = torch.nn.functional.adaptive_avg_pool2d(features, (4, 4))
        return tokens.flatten(2).transpose(1, 2).contiguous()


def validate_arrays(tokens, pooled, ids, geology, age, split, frozen):
    if split not in ("train", "val"):
        raise ValueError("Train/Val cache only")
    n = frozen["rows"]
    if (tokens.shape != (n, 16, 256) or pooled.shape != (n, 256)
            or tokens.dtype != np.dtype("float32") or pooled.dtype != np.dtype("float32")):
        raise ValueError("invalid context shape/dtype")
    if not np.isfinite(tokens).all() or not np.isfinite(pooled).all():
        raise ValueError("nonfinite context")
    if not np.array_equal(tokens.mean(axis=1), pooled):
        raise ValueError("pooled control must be the mean of the same tokens")
    if len(ids) != n or len(set(map(str, ids))) != n or ids_sha256(ids) != frozen["ordered_ids_sha256"]:
        raise ValueError("cache IDs differ from frozen split")
    for task, targets, classes in (("geology", geology, 49), ("age", age, 5)):
        if targets.shape != (n,) or targets.dtype.kind not in "iu" or not ((targets >= 0) & (targets < classes)).all():
            raise ValueError("invalid cache targets")
        if np.bincount(targets, minlength=classes).tolist() != frozen[task+"_class_counts"]:
            raise ValueError("cache support differs from frozen split")
