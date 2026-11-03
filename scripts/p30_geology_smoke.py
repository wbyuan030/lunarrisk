#!/usr/bin/env python3
"""49-output same-feature engineering smoke; no benchmark score or pretrained encoder claim."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "encoder", "stats", "class-map", "outdir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    args = parser.parse_args(argv)
    import torch
    from lunarrisk.benchmarks.candidate_heads import OrdinaryHead, CandidateIDHead, masked_softmax
    from lunarrisk.benchmarks.moonstone import MoonstoneHDF5Pilot
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("existing ROCm GPU required")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    args.outdir.mkdir(parents=True, exist_ok=False)
    inputs = {name: digest(path) for name, path in (("pilot", args.pilot), ("encoder", args.encoder),
              ("stats", args.stats), ("class_map", args.class_map))}
    source_paths = (Path(__file__).resolve(), ROOT/"src/lunarrisk/benchmarks/candidate_heads.py",
                    ROOT/"scripts/p30_age_smoke.py", ROOT/"src/lunarrisk/benchmarks/moonstone.py")
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    start = time.monotonic()
    if args.replay:
        if args.feature_cache is None:
            raise ValueError("fresh-process replay requires the original feature-cache file")
        payload = torch.load(args.replay, map_location="cpu", weights_only=True)
        if payload["inputs"] != inputs or payload["source_sha256"] != sources:
            raise ValueError("checkpoint input/source mismatch")
        cache_path = args.feature_cache
        if digest(cache_path) != payload["feature_cache_sha256"]:
            raise ValueError("feature cache checksum mismatch")
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        stats, mapping = json.loads(args.stats.read_text()), json.loads(args.class_map.read_text())
        spec = importlib.util.spec_from_file_location("p30_age_runner", ROOT/"scripts/p30_age_smoke.py")
        age = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(age)
        saved = torch.load(args.encoder, map_location="cpu", weights_only=True)
        if saved["inputs"]["pilot_sha256"] != inputs["pilot"] or saved["test_evaluated"]:
            raise ValueError("encoder must be the Train-only Age smoke on this exact pilot")
        if saved["inputs"]["stats_sha256"] != inputs["stats"] or saved["inputs"]["class_map_sha256"] != inputs["class_map"]:
            raise ValueError("encoder preprocessing mismatch")
        network = age.model()
        network.load_state_dict(saved["state_dict"])
        encoder = network[:-1].to(device).eval().requires_grad_(False)
        cache = {"inputs": inputs, "encoder_origin": "two-update scratch CNN supervised on five Train Age labels",
                 "pretrained": False, "feature_dim": 256, "splits": {}}
        with torch.no_grad():
            for split, code in (("train", 0), ("val", 1)):
                dataset = MoonstoneHDF5Pilot(args.pilot, stats, mapping, task="geology", split=code)
                if len(dataset) != 5:
                    raise ValueError("locked engineering pilot requires exactly five rows per split")
                x, y, ids = age.dataset_batch(dataset, device)
                features = encoder(x).detach().cpu()
                if not torch.isfinite(features).all():
                    raise ValueError("nonfinite shared features")
                cache["splits"][split] = {"features": features, "targets": y.cpu(), "ids": ids}
                dataset.close()
        cache_path = args.outdir/"feature_cache.pt"
        torch.save(cache, cache_path)
    if cache["inputs"] != inputs:
        raise ValueError("cache input mismatch")
    candidates = torch.arange(49, device=device)
    states, histories, optimizers, parameters = {}, {}, {}, {}
    networks = {}
    for name, constructor in (("ordinary", OrdinaryHead), ("candidate_id", CandidateIDHead)):
        torch.manual_seed(42)
        head = constructor().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
        history = []
        if args.replay:
            head.load_state_dict(payload["heads"][name])
        else:
            train = cache["splits"]["train"]
            x, y = train["features"].to(device), train["targets"].to(device)
            for _ in range(2):
                head.train()
                optimizer.zero_grad()
                logits = head(x) if name == "ordinary" else head(x, candidates)
                loss = torch.nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite head loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise ValueError("nonfinite head gradient")
                optimizer.step()
                history.append(float(loss.detach()))
        head.eval()
        networks[name] = head
        states[name] = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        histories[name], optimizers[name] = history, optimizer.state_dict()
        parameters[name] = sum(p.numel() for p in head.parameters())
    if not args.replay:
        payload = {"protocol": "GEOLOGY_SAME_FEATURE_ENGINEERING_ONLY", "code_sha": os.environ.get("LUNARRISK_GIT_SHA", "UNKNOWN"),
                   "source_sha256": sources, "inputs": inputs, "feature_cache_sha256": digest(cache_path),
                   "heads": states, "optimizers": optimizers, "updates_each": 2, "seed": 42,
                   "test_evaluated": False, "torch_rng_state": torch.get_rng_state(),
                   "cuda_rng_state": torch.cuda.get_rng_state_all()}
        torch.save(payload, args.outdir/"checkpoint.pt")
    rows, permutation_delta, mask_delta = [], 0.0, 0.0
    with torch.no_grad():
        for split, data in cache["splits"].items():
            x = data["features"].to(device)
            for name, head in networks.items():
                logits = head(x) if name == "ordinary" else head(x, candidates)
                probability = torch.softmax(logits, dim=-1)
                if not torch.isfinite(probability).all():
                    raise ValueError("nonfinite probability")
                if name == "candidate_id":
                    reversed_p = torch.softmax(head(x, candidates.flip(0)), dim=-1).flip(-1)
                    permutation_delta = max(permutation_delta, float((reversed_p-probability).abs().max()))
                else:
                    control = masked_softmax(logits, torch.ones_like(logits, dtype=torch.bool))
                    mask_delta = max(mask_delta, float((control-probability).abs().max()))
                for sample, target, p in zip(data["ids"], data["targets"].tolist(), probability.cpu().tolist()):
                    rows.append({"method": name, "sample_id": sample, "split": split, "target": target,
                                 **{f"p_{i}": value for i, value in enumerate(p)}})
    if permutation_delta > 1e-6 or mask_delta > 1e-6:
        raise ValueError("candidate ordering or all-class mask control failed fixed tolerance")
    with (args.outdir/"predictions.csv").open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "GEOLOGY_SAME_FEATURE_ENGINEERING_ONLY", "formal_benchmark_score": False,
              "protocol": "HDF5_DERIVED_PILOT", "test_evaluated": False, "pretrained_encoder": False,
              "encoder_origin": cache["encoder_origin"], "feature_cache_sha256": digest(cache_path),
              "code_sha": payload["code_sha"], "source_sha256": sources, "inputs": inputs,
              "classes": 49, "parameters": parameters, "updates_each": 2, "seed": 42,
              "train_loss": histories, "candidate_permutation_max_delta": permutation_delta,
              "ordinary_all_class_mask_max_delta": mask_delta,
              "splits": {k: {"ids": v["ids"], "targets": v["targets"].tolist()} for k, v in cache["splits"].items()},
              "wall_seconds": time.monotonic()-start, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "torch": str(torch.__version__),
              "hip": torch.version.hip, "predictions_sha256": digest(args.outdir/"predictions.csv"),
              "checkpoint_sha256": digest(args.replay if args.replay else args.outdir/"checkpoint.pt")}
    (args.outdir/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
