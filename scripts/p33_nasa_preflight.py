#!/usr/bin/env python3
"""Audit a local raw Train batch; optionally check paired NASA fwd/backward.

No downloads, training loop, optimizer step, Test scoring, or guessed config.
An output is written only after the requested checks pass; failures exit nonzero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lunarrisk.nasa_transfer.contracts import (NASA_CODE_REVISION, NASA_MODEL_REVISION,
    file_sha256, fingerprint, prepare_terrain, spatial_policy_receipt, STRICT_SPATIAL_POLICY, SPATIAL_POLICIES)


from lunarrisk.nasa_transfer.cohort import (
    TRAIN_PLAN_PATH, METADATA_ARCHIVE_PATH, load_authoritative_train, validate_train_batch,
)


def read_batch(path, provenance, authoritative_targets):
    if provenance.get("split") != "train":
        raise ValueError("finite-gradient preflight is Train only; Test remains HOLD")
    if provenance.get("batch_sha256") != file_sha256(path):
        raise ValueError("batch SHA256 mismatch")
    with np.load(path, allow_pickle=False) as data:
        required = {"elevation", "slope", "valid_elevation", "valid_slope", "sample_ids", "targets"}
        if set(data.files) != required:
            raise ValueError("unexpected batch fields; do not pass extra modalities or geology names")
        raw = {key: np.array(data[key]) for key in ("elevation", "slope")}
        valid = {key: np.array(data["valid_" + key]) for key in raw}
        ids, targets = np.array(data["sample_ids"]), np.array(data["targets"])
    if ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) not in (1, 2) or len(set(ids.tolist())) != len(ids) or any(not value.strip() for value in ids.tolist()):
        raise ValueError("preflight requires 1 or 2 unique string sample IDs")
    if targets.shape != ids.shape or not np.issubdtype(targets.dtype, np.integer) or not np.isin(targets, np.arange(49)).all():
        raise ValueError("targets must be paired Geology IDs 0..48")
    if any(value.shape[0] != len(ids) for value in raw.values()):
        raise ValueError("sample IDs/targets and arrays differ in length")
    validate_train_batch(ids.tolist(), targets, authoritative_targets)
    return raw, valid, ids, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True, help="raw Train NPZ, <=2 rows, no pickle")
    parser.add_argument("--provenance", type=Path, required=True)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--train-plan", type=Path, default=root / TRAIN_PLAN_PATH,
                        help="frozen repository Train plan; contents must match pinned SHA256")
    parser.add_argument("--train-metadata", type=Path, default=root / METADATA_ARCHIVE_PATH,
                        help="frozen metadata archive; contents must match pinned SHA256")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--spatial-policy", choices=SPATIAL_POLICIES, default=STRICT_SPATIAL_POLICY)
    parser.add_argument("--output", type=Path, required=True, help="new JSON path; never overwrite")
    parser.add_argument("--forward-backward", action="store_true")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--adapt-inputs", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite existing preflight evidence")
    if args.forward_backward and (args.upstream is None or args.checkpoint is None):
        parser.error("--forward-backward requires --upstream and --checkpoint")
    if file_sha256(args.config) != args.config_sha256:
        raise ValueError("NASA config SHA256 mismatch")
    policy_receipt = spatial_policy_receipt(args.spatial_policy, args.config_sha256)
    # Parsing data-only YAML is separate from importing/running upstream code.
    import yaml
    config = yaml.safe_load(args.config.read_text())
    provenance = json.loads(args.provenance.read_text())
    authoritative_targets, cohort_receipt = load_authoritative_train(args.train_plan, args.train_metadata)
    raw, valid, ids, targets = read_batch(args.batch, provenance, authoritative_targets)
    prepared = prepare_terrain(raw, valid, config["data"]["domains"], provenance,
                               spatial_policy=args.spatial_policy, config_sha256=args.config_sha256)
    report = {"kind": "ENGINEERING_PREFLIGHT_NOT_TRANSFER_SCORE", "input_contract": "PASS",
              "authoritative_train_membership": cohort_receipt, "spatial_policy": policy_receipt,
              "nasa_code_revision": NASA_CODE_REVISION, "nasa_model_revision": NASA_MODEL_REVISION,
              "config_sha256": args.config_sha256, "batch_sha256": file_sha256(args.batch),
              "provenance_sha256": file_sha256(args.provenance), "sample_ids": ids.tolist(),
              "prepared_sha256": fingerprint({key: {"shape": list(value.shape),
                  "bytes_sha256": __import__('hashlib').sha256(value.tobytes()).hexdigest()}
                  for key, value in prepared.items()}),
              "official_model_execution": "NOT_RUN", "science_claim": "NOT_ESTABLISHED"}
    if args.forward_backward:
        import torch
        from lunarrisk.nasa_transfer.torch_adapter import (official_backbone, build_paired_models,
                                                          load_verified_checkpoint)
        start = time.monotonic()
        factory = lambda: official_backbone(args.config, args.config_sha256, args.upstream, spatial_policy=args.spatial_policy)
        models, coverage = build_paired_models(factory, lambda backbone: load_verified_checkpoint(backbone, args.checkpoint),
                                               adapt_inputs=args.adapt_inputs)
        report["checkpoint_coverage"] = coverage
        report["arms"] = {}
        device = torch.device(args.device)
        batch = {key: torch.from_numpy(value).to(device) for key, value in prepared.items()}
        labels = torch.from_numpy(targets.astype(np.int64)).to(device)
        for name, model in models.items():
            model.to(device).train()
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            arm_start = time.monotonic()
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            trainable = [(key, value) for key, value in model.named_parameters() if value.requires_grad]
            if not torch.isfinite(loss) or any(value.grad is None or not torch.isfinite(value.grad).all() for _, value in trainable):
                raise RuntimeError(f"{name}: missing/nonfinite trainable gradients")
            if any(value.grad is not None for value in model.backbone.parameters()):
                raise RuntimeError("frozen encoder received gradients")
            if args.adapt_inputs and any(value.grad.abs().sum() == 0 for value in model.adapters.parameters()):
                raise RuntimeError("input adapter received a zero gradient; inspect before training")
            if device.type == "cuda":
                torch.cuda.synchronize()
            report["arms"][name] = {"forward_backward": "PASS", "logit_shape": list(logits.shape),
                 "trainable_parameters": sum(value.numel() for _, value in trainable),
                 "seconds": time.monotonic() - arm_start,
                 "peak_device_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
            model.zero_grad(set_to_none=True)
            model.to("cpu")
            del logits, loss
        report.update(official_model_execution="PASS", seconds=time.monotonic() - start, device=str(device),
                      adaptation="input_affine" if args.adapt_inputs else "frozen_probe", torch_version=torch.__version__)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(serialized)
    print(json.dumps({"input_contract": "PASS", "official_model_execution": report["official_model_execution"]}))


if __name__ == "__main__":
    main()
