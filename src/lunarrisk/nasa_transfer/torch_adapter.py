"""Optional PyTorch integration. No automatic downloads or environment changes.

The official pinned constructor is used, not a NASA-like replacement model.
Tests with a tiny stand-in validate only our wrapper, never pretrained transfer.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from .contracts import (NASA_CODE_REVISION, NASA_CHECKPOINT_BYTES, NASA_CHECKPOINT_SHA256,
                        audit_inference_state, file_sha256, unwrap_state_dict, validate_domain, spatial_policy_receipt, STRICT_SPATIAL_POLICY)


def _verify_import_origins(root):
    for name, module in tuple(sys.modules.items()):
        if name == "ni_lfm" or name.startswith("ni_lfm.") or name == "terratorch_integration" or name.startswith("terratorch_integration."):
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError(f"unpinned NASA module already loaded: {name}; use a fresh process")


def official_backbone(config_path, config_sha256, upstream_path, *, spatial_policy=STRICT_SPATIAL_POLICY):
    """Construct the audited base variant with unchanged canonical patch geometry.

    Requires the local agent's compatible, isolated NASA runtime. We intentionally
    do not run pip or mutate sys.path until checkout and config integrity pass.
    """
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("NASA requires Python >=3.11,<3.13; preserve the working ROCm runtime")
    root = Path(upstream_path).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True)
    if revision != NASA_CODE_REVISION or dirty.strip():
        raise ValueError("NASA source must be a clean checkout of the audited commit")
    if file_sha256(config_path) != config_sha256:
        raise ValueError("NASA config SHA256 mismatch")
    spatial_policy_receipt(spatial_policy, config_sha256)
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    domains = OmegaConf.to_container(config.data.domains, resolve=True)
    for key in ("dtm", "slope"):
        validate_domain(domains[key], spatial_policy=spatial_policy)
    _verify_import_origins(root)
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("terratorch_integration.lunar_backbone")
    finally:
        sys.path.remove(str(root))
    _verify_import_origins(root)
    if Path(module.__file__).resolve() != root / "terratorch_integration/lunar_backbone.py":
        raise RuntimeError("a different NASA package is already imported; use a fresh process")
    backbone = module.LunarBackbone(
        variant="base", modalities=["dtm", "slope"], cfg=config,
        checkpoint_path=None, merge_method=None, remove_register_tokens=True,
    )
    # The official forward uses only the encoder. Remove unused generative
    # decoder state before checkpoint loading/device transfer, never serialize it.
    for name in ("decoder", "decoder_embeddings", "decoder_norm", "decoder_proj_context", "mask_token"):
        if hasattr(backbone.model, name):
            delattr(backbone.model, name)
    return backbone


def load_verified_checkpoint(backbone, path, *, expected_sha256=NASA_CHECKPOINT_SHA256,
                             expected_bytes=NASA_CHECKPOINT_BYTES):
    path = Path(path)
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_sha256:
        raise ValueError("NASA checkpoint byte count/SHA256 mismatch before deserialization")
    # weights_only forbids the ordinary arbitrary-pickle fallback. mmap avoids
    # retaining the complete generative checkpoint in resident memory at once.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    report = audit_inference_state(backbone.model.state_dict(), checkpoint)
    state = unwrap_state_dict(checkpoint)
    selected = {key: state[key] for key in report["loaded_keys"]}
    for key, value in selected.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite checkpoint tensor: {key}")
    # The pruned target must now consist entirely of selected inference state.
    backbone.model.load_state_dict(selected, strict=True)
    report.update(checkpoint_sha256=expected_sha256, checkpoint_bytes=expected_bytes)
    return report


class TerrainTransfer(nn.Module):
    """Frozen NASA encoder plus ordinary 49-way head and optional input adapters.

    Frozen encoder parameters and BN/dropout stay frozen. Autograd through the
    encoder is deliberately retained when training the input adapters. This is a
    two-parameter-per-modality affine adapter, NOT LoRA or full fine-tuning.
    """
    def __init__(self, backbone, *, num_classes=49, adapt_inputs=False):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        if getattr(backbone, "merge_method", None) is not None:
            raise ValueError("use concatenated tokens, not zero-inferred masked_group_mean")
        if not getattr(backbone, "remove_register_tokens", False):
            raise ValueError("remove register tokens exactly once before pooled classification")
        self.expected_shapes = {}
        for key in ("dtm", "slope"):
            embedding = backbone.model.encoder_embeddings[key]
            size = getattr(embedding, "image_size", None)
            if not isinstance(size, (tuple, list)) or len(size) != 2 or any(not isinstance(n, int) or n <= 0 for n in size):
                raise ValueError("each terrain embedding must declare its canonical image_size")
            self.expected_shapes[key] = tuple(size)
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.adapters = nn.ModuleDict({key: nn.Conv2d(1, 1, 1) for key in ("dtm", "slope")})
        for module in self.adapters.values():
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            module.requires_grad_(adapt_inputs)
        self.head = nn.Linear(backbone.out_channels[-1], num_classes)
        self.adapt_inputs = bool(adapt_inputs)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs):
        if set(inputs) != {"dtm", "slope"}:
            raise ValueError("only normalized dtm and slope may enter this model")
        batch = None
        for key, value in inputs.items():
            if value.ndim != 4 or value.shape[1] != 1 or value.shape[0] == 0 or not value.is_floating_point():
                raise ValueError("expected nonempty floating (B,1,H,W) terrain")
            if tuple(value.shape[-2:]) != self.expected_shapes[key]:
                raise ValueError("terrain shape differs from canonical NASA input_size; implicit resize is forbidden")
            if batch is not None and value.shape[0] != batch:
                raise ValueError("modality batch sizes differ")
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite input")
            batch = value.shape[0]
        adapted = {key: self.adapters[key](inputs[key]) for key in self.adapters}
        # Do not surround this forward with no_grad: input-adapter gradients
        # must pass THROUGH the frozen encoder.
        tokens = self.backbone(adapted)[-1]
        if tokens.ndim != 3 or tokens.shape[1] == 0 or not torch.isfinite(tokens).all():
            raise ValueError("invalid NASA encoder token output")
        return self.head(tokens.mean(dim=1))


def build_paired_models(factory, load_pretrained, *, seed=42, adapt_inputs=False, num_classes=49):
    """Build identical random/pretrained architectures with paired head/adapters.

    The factory must create a CPU backbone without loading weights. Loading is
    applied to one arm only, after identical initialization. Caller must keep the
    batch order, inputs, labels, optimizer and selection budget paired as well.
    """
    models = {}
    for name in ("random", "pretrained"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            models[name] = TerrainTransfer(factory(), num_classes=num_classes, adapt_inputs=adapt_inputs)
    report = load_pretrained(models["pretrained"].backbone)
    for component in ("head", "adapters"):
        left = getattr(models["random"], component).state_dict()
        right = getattr(models["pretrained"], component).state_dict()
        if left.keys() != right.keys() or not all(torch.equal(left[key], right[key]) for key in left):
            raise RuntimeError("unpaired task-head/adapter initialization")
    return models, report
