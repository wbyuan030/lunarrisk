"""Replay saved P2.2 posteriors and independently repeat point/joint seed 42.

This audits existing development runs without changing models or selecting rules. All comparisons
use the saved rule manifest, calibration temperatures, split order and noise seed. Old run folders
are never overwritten. Independent repetitions execute in new Python processes with full logs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

import train_posterior_p21 as runner
from lunarrisk.probabilistic import export_state
from lunarrisk.rules import from_dict


def replay_baselines(run, cache_path, device):
    import compare_p22 as comparison

    labels = pd.read_csv(run / "labels_p22.csv")
    _, cache, _ = runner.load_dataset(run / "labels_p22.csv", cache_path, runner.load_baseline_module())
    metadata = comparison.load_cache_meta(cache_path)
    records = {}
    for checkpoint in sorted((run / "ckpt").glob("*.pt")):
        if checkpoint.stem.split("_seed")[0] not in ("coarse_ridge", "coarse_logistic", "direct_event"):
            continue
        replay, _ = comparison.predict_baseline_checkpoint(checkpoint, labels, cache, metadata, device)
        original = pd.read_csv(run / "pred" / f"{checkpoint.stem}.csv")
        original, replay = comparison.aligned_frames(original, replay)
        columns = [column for column in original if column.startswith(("p_", "mean_"))]
        maximum = max(float(np.max(np.abs(original[column] - replay[column]))) for column in columns)
        records[checkpoint.stem] = {"max_abs_delta": maximum, "atol": 1e-7,
                                    "passed": maximum <= 1e-7,
                                    "checkpoint_sha256": runner.file_sha256(checkpoint)}
    if not records:
        raise ValueError("no baseline checkpoints found")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    torch.use_deterministic_algorithms(True)
    if args.baseline_worker:
        records = replay_baselines(args.run, args.cache, torch.device(args.device))
        (args.run / "artifact_audit" / "baseline_replay.json").write_text(json.dumps(records, indent=2) + "\n")
        return 0 if all(record["passed"] for record in records.values()) else 2
    if args.repetitions < 1:
        raise ValueError("at least one independent repetition required")
    destination = args.run / "artifact_audit"
    destination.mkdir(exist_ok=False)
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    labels_path = args.run / "labels_p22.csv"
    labels, cache, scalars = runner.load_dataset(labels_path, args.cache, runner.load_baseline_module())
    frames = {}
    audit = {"environment": {"python": sys.version, "torch": torch.__version__,
                             "hip": torch.version.hip,
                             "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
             "source_sha256": runner.source_fingerprints(), "replay": {}, "repetitions": []}
    for checkpoint in sorted((args.run / "ckpt").glob("*.pt")):
        if checkpoint.stem.split("_seed")[0] not in ("point", "diagonal", "joint"):
            continue
        model, payload = runner.load_checkpoint(checkpoint)
        if runner.file_sha256(args.cache) != payload["cache_sha256"]:
            raise ValueError("cache fingerprint changed")
        if runner.file_sha256(labels_path) != payload["labels_sha256"]:
            raise ValueError("labels fingerprint changed")
        if export_state(model)["weights_sha256"] != payload["weights_sha256"]:
            raise ValueError("weights fingerprint changed")
        frame = pd.read_csv(args.run / "pred" / f"{checkpoint.stem}.csv")
        frames[checkpoint.stem] = frame
        report = json.loads((args.run / "report" / f"{checkpoint.stem}.json").read_text())
        rules = {name: from_dict(spec) for name, spec in payload["rules"].items()}
        branches = [("", "splits", "full", "full")]
        if payload["variant"] == "joint":
            branches += [("__u0", "splits_u0", "u0_shared", "full"),
                         ("__u0_refit", "splits_u0_refit", "u0_refit", "full"),
                         ("__matched", "splits_matched_marginals", "full", "matched_marginals")]
        deltas, metric_deltas = [], []
        for split, ids in payload["sample_ids_by_split"].items():
            original = frame.set_index("sample_id").loc[ids]
            if original["split"].tolist() != [split] * len(ids):
                raise ValueError("saved split membership mismatch")
            for suffix, metric_key, ablation, mode in branches:
                _, probabilities, _ = runner.predict_from_checkpoint(
                    checkpoint, ids, cache, scalars, rules, device, draws=payload["draws"],
                    seed=payload["sampling_seed"], covariance_mode=mode, ablation=ablation)
                for name in rules:
                    deltas.append(float(np.max(np.abs(probabilities[name] - original[f"p_{name}{suffix}"]))))
                    truth = original[f"event_{name}"].to_numpy()
                    observed = float(np.mean((original[f"p_{name}{suffix}"].to_numpy()-truth)**2))
                    metric_deltas.append(abs(observed - report[metric_key][split]["events"][name]["brier"]))
        audit["replay"][checkpoint.stem] = {
            "checkpoint_file_sha256": runner.file_sha256(checkpoint),
            "weights_sha256": payload["weights_sha256"],
            "max_probability_delta": max(deltas), "max_brier_delta": max(metric_deltas),
            "passed": max(deltas) == 0 and max(metric_deltas) < 1e-12}

    command = [sys.executable, str(Path(__file__).resolve()), "--run", str(args.run),
               "--cache", str(args.cache), "--device", args.device, "--baseline-worker"]
    with (destination / "baseline_replay.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    audit["baseline_replay"] = json.loads((destination / "baseline_replay.json").read_text())
    config = json.loads((args.run / "compare.json").read_text())["manifest"]["config"]
    for variant in ("point", "joint"):
        stem = f"{variant}_seed42"
        model, payload = runner.load_checkpoint(args.run / "ckpt" / f"{stem}.pt")
        for repetition in range(1, args.repetitions + 1):
            out = destination / f"{stem}_repeat{repetition}"
            out.mkdir()
            command = [sys.executable, str(Path(runner.__file__)), "--labels", str(labels_path),
                       "--cache", str(args.cache), "--variant", variant, "--seed", "42",
                       "--epochs", str(config["epochs"]), "--calibrate", "--deterministic",
                       "--batch-size", str(config["batch_size"]), "--lr", str(config["lr"]),
                       "--draws", str(payload["draws"]), "--sampling-seed", str(payload["sampling_seed"]),
                       "--device", args.device, "--out", str(out / "report.json"),
                       "--save-checkpoint", str(out / "model.pt"),
                       "--predictions-out", str(out / "pred.csv")]
            if variant == "joint":
                command += ["--ablate-u", "--ablate-u-calibration-mode", "both", "--matched-marginals"]
            start = time.monotonic()
            with (out / "run.log").open("w") as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ).returncode
            record = {"variant": variant, "seed": 42, "repetition": repetition, "command": command,
                      "exit_code": status, "wall_seconds": time.monotonic()-start}
            if status == 0:
                _, repeated = runner.load_checkpoint(out / "model.pt")
                original = frames[stem].set_index("sample_id").sort_index()
                replay = pd.read_csv(out / "pred.csv").set_index("sample_id").loc[original.index]
                columns = [column for column in original if column.startswith("p_")]
                record.update({"weights_equal": payload["weights_sha256"] == repeated["weights_sha256"],
                               "temperature_equal": payload["temperature"] == repeated["temperature"],
                               "u0_temperatures_equal": payload["temperatures_u0"] == repeated["temperatures_u0"],
                               "max_probability_delta": float(np.max(np.abs(
                                   original[columns].to_numpy()-replay[columns].to_numpy())))})
            audit["repetitions"].append(record)
    audit["passed"] = all(record["passed"] for record in audit["baseline_replay"].values()) and all(
        record["passed"] for record in audit["replay"].values()) and all(
        record.get("weights_equal") and record.get("temperature_equal") and
        record.get("u0_temperatures_equal") and record.get("max_probability_delta") == 0
        for record in audit["repetitions"])
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
