#!/usr/bin/env python3
"""Lock, run and replay a bounded NASA random/pretrained Train/Val pair.

All assets remain local. No downloads, Test scoring or budget defaults. Run mode
requires the exact pre-score manifest and a successful real-model preflight.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lunarrisk.nasa_transfer.contracts import (NASA_CODE_REVISION, NASA_MODEL_REVISION,
    NASA_CHECKPOINT_BYTES, NASA_CHECKPOINT_SHA256, file_sha256, fingerprint, spatial_policy_receipt, STRICT_SPATIAL_POLICY, SPATIAL_POLICIES)
from lunarrisk.nasa_transfer.cohort import (TRAIN_PLAN_PATH, METADATA_ARCHIVE_PATH,
    TRAIN_PLAN_SHA256, METADATA_ARCHIVE_SHA256)
from lunarrisk.nasa_transfer.experiment import validate_budget, paired_summary

SOURCES = (
    "scripts/p33_train_pair.py", "scripts/p33_nasa_preflight.py", "src/lunarrisk/nasa_transfer/training.py",
    "src/lunarrisk/nasa_transfer/experiment.py", "src/lunarrisk/nasa_transfer/dataset.py",
    "src/lunarrisk/nasa_transfer/cohort.py", "src/lunarrisk/nasa_transfer/contracts.py",
    "src/lunarrisk/nasa_transfer/torch_adapter.py", "src/lunarrisk/benchmarks/prefix.py",
    "src/lunarrisk/benchmarks/moonstone.py", "protocols/p33_native256_spatial_policy.json",
)


def sources():
    return {path: file_sha256(ROOT / path) for path in SOURCES}


def write_new(path, payload):
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as handle:
        handle.write(text)


def validate_preflight(report, config_sha256, adaptation, device, spatial_policy=STRICT_SPATIAL_POLICY):
    if (report.get("kind") != "ENGINEERING_PREFLIGHT_NOT_TRANSFER_SCORE"
            or report.get("official_model_execution") != "PASS" or report.get("input_contract") != "PASS"
            or report.get("config_sha256") != config_sha256 or report.get("device") != device
            or report.get("nasa_code_revision") != NASA_CODE_REVISION
            or report.get("nasa_model_revision") != NASA_MODEL_REVISION
            or report.get("adaptation") != ("input_affine" if adaptation == "input_affine" else "frozen_probe")):
        raise ValueError("requires successful matching real-model preflight, not an input-only or synthetic check")
    if report.get("spatial_policy") != spatial_policy_receipt(spatial_policy, config_sha256):
        raise ValueError("preflight spatial policy missing or different; rerun under the declared protocol")
    coverage = report.get("checkpoint_coverage", {})
    if (coverage.get("checkpoint_sha256") != NASA_CHECKPOINT_SHA256
            or coverage.get("checkpoint_bytes") != NASA_CHECKPOINT_BYTES
            or coverage.get("loaded_inference_fraction") != 1.0):
        raise ValueError("preflight lacks complete verified NASA encoder loading")
    if set(report.get("arms", {})) != {"random", "pretrained"}:
        raise ValueError("preflight must cover both arms")
    if any(row.get("forward_backward") != "PASS" for row in report["arms"].values()):
        raise ValueError("both preflight forwards/backwards must pass")
    cohort = report.get("authoritative_train_membership", {})
    if (cohort.get("train_rows") != 5135 or cohort.get("train_plan_sha256") != TRAIN_PLAN_SHA256
            or cohort.get("metadata_archive_sha256") != METADATA_ARCHIVE_SHA256):
        raise ValueError("preflight lacks authenticated Train membership")


def runtime_identity():
    import torch
    return {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__),
            "torch_hip": torch.version.hip, "torch_cuda": torch.version.cuda}


def load_data(args):
    import yaml
    from lunarrisk.nasa_transfer.dataset import open_datasets
    if file_sha256(args.nasa_config) != args.nasa_config_sha256:
        raise ValueError("NASA config changed")
    config = yaml.safe_load(args.nasa_config.read_text())
    provenance = json.loads(args.provenance.read_text())
    datasets, receipt = open_datasets(
        data_path=args.data, stats_path=args.stats, class_map_path=args.class_map,
        metadata_path=args.metadata, range_report_path=args.range_report,
        plan_path=ROOT / TRAIN_PLAN_PATH, metadata_archive_path=ROOT / METADATA_ARCHIVE_PATH,
        domains=config["data"]["domains"], provenance=provenance,
        spatial_policy=args.spatial_policy, config_sha256=args.nasa_config_sha256,
    )
    return datasets, receipt


def close_data(datasets):
    for dataset in datasets.values():
        dataset.close()


def asset_identity(args):
    if args.nasa_checkpoint.stat().st_size != NASA_CHECKPOINT_BYTES or file_sha256(args.nasa_checkpoint) != NASA_CHECKPOINT_SHA256:
        raise ValueError("NASA checkpoint size/hash differs from fixed source")
    return {"nasa_checkpoint_sha256": NASA_CHECKPOINT_SHA256, "nasa_checkpoint_bytes": NASA_CHECKPOINT_BYTES,
            "nasa_config_sha256": args.nasa_config_sha256, "provenance_sha256": file_sha256(args.provenance),
            "preflight_sha256": file_sha256(args.preflight)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("lock", "run", "replay"), required=True)
    for name in ("data", "metadata", "range-report", "nasa-config", "nasa-checkpoint", "provenance", "preflight", "manifest"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--nasa-config-sha256", required=True)
    result.add_argument("--spatial-policy", choices=SPATIAL_POLICIES, default=STRICT_SPATIAL_POLICY)
    result.add_argument("--upstream", type=Path, required=True)
    result.add_argument("--stats", type=Path, default=ROOT / "reports/evidence/p30_20261103/hf_channel_stats.json")
    result.add_argument("--class-map", type=Path, default=ROOT / "reports/evidence/p30_20261103/geologic_units.json")
    result.add_argument("--manifest-sha256", help="exact lock-produced digest, required for run/replay")
    result.add_argument("--budget", type=Path, help="explicit budget JSON, lock mode only")
    result.add_argument("--outdir", type=Path)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--run-dir", type=Path, help="original selected run, replay only")
    result.add_argument("--seed", type=int, help="replay only")
    result.add_argument("--arm", choices=("random", "pretrained"), help="replay only")
    return result


def replay_command(args, directory, output, seed, arm):
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", "replay"]
    for name in ("data", "metadata", "range_report", "nasa_config", "nasa_checkpoint", "provenance",
                 "preflight", "manifest", "upstream", "stats", "class_map"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name).resolve())]
    command += ["--nasa-config-sha256", args.nasa_config_sha256, "--manifest-sha256", args.manifest_sha256,
                "--device", args.device, "--spatial-policy", args.spatial_policy, "--run-dir", str(directory.resolve()), "--outdir", str(output.resolve()),
                "--seed", str(seed), "--arm", arm]
    return command


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.mode == "lock":
        if args.budget is None or any(value is not None for value in (args.outdir, args.run_dir, args.seed, args.arm, args.manifest_sha256)):
            raise ValueError("lock requires --budget and no run/output selection flags")
        if args.manifest.exists():
            raise FileExistsError("never overwrite a locked manifest")
        budget = validate_budget(json.loads(args.budget.read_text()))
        validate_preflight(json.loads(args.preflight.read_text()), args.nasa_config_sha256, budget["adaptation"], args.device, args.spatial_policy)
        runtime = runtime_identity()
        if json.loads(args.preflight.read_text()).get("torch_version") != runtime["torch"]:
            raise ValueError("preflight torch runtime differs from the locking process")
        before = sources()
        assets = asset_identity(args)
        datasets, receipt = load_data(args)
        try:
            from lunarrisk.nasa_transfer.dataset import scan_datasets
            scan = scan_datasets(datasets)
        finally:
            close_data(datasets)
        if sources() != before:
            raise ValueError("source changed while scanning inputs")
        manifest = {"protocol": "P33_NASA_TERRAIN_PAIRED_V2", "status": "LOCKED_BEFORE_PAIR_SCORES",
                    "budget": budget, "runtime": runtime, "spatial_policy": spatial_policy_receipt(args.spatial_policy, args.nasa_config_sha256), "source_sha256": before, "assets": assets, "data_identity": receipt,
                    "input_scan": scan, "nasa_code_revision": NASA_CODE_REVISION,
                    "nasa_model_revision": NASA_MODEL_REVISION, "device": args.device,
                    "arms": ["random", "pretrained"], "test_scored": False,
                    "scope": "Exploratory reused northern Train/Val; no independent Test or global claim",
                    "early_stop": "same maximum budget/selection rule; realized epoch counts may differ",
                    "resume": "not implemented; preserve interrupted epoch files, no automatic retraining"}
        write_new(args.manifest, manifest)
        print(json.dumps({"status": "LOCKED", "manifest_sha256": file_sha256(args.manifest), "model_scored": False}))
        return 0
    if args.budget is not None or args.outdir is None:
        raise ValueError("run/replay require --outdir and forbid budget overrides")
    if args.mode == "run" and any(value is not None for value in (args.run_dir, args.seed, args.arm)):
        raise ValueError("run executes the complete declared matrix, not selected favorable arms/seeds")
    if args.mode == "replay" and any(value is None for value in (args.run_dir, args.seed, args.arm)):
        raise ValueError("replay requires --run-dir, --seed and --arm")
    if args.outdir.exists():
        raise FileExistsError("output exists; preserve completed/interrupted evidence")
    if args.manifest_sha256 is None or file_sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("run/replay require the unchanged lock-produced manifest SHA256")
    plan = json.loads(args.manifest.read_text())
    budget = validate_budget(plan["budget"])
    if (plan["protocol"] != "P33_NASA_TERRAIN_PAIRED_V2"
            or plan.get("spatial_policy") != spatial_policy_receipt(args.spatial_policy, args.nasa_config_sha256) or plan["status"] != "LOCKED_BEFORE_PAIR_SCORES"
            or plan["source_sha256"] != sources() or plan["test_scored"] is not False
            or plan["arms"] != ["random", "pretrained"] or plan["device"] != args.device):
        raise ValueError("locked source, protocol, device or Test boundary changed")
    validate_preflight(json.loads(args.preflight.read_text()), args.nasa_config_sha256, budget["adaptation"], args.device, args.spatial_policy)
    if runtime_identity() != plan["runtime"]:
        raise ValueError("locked Python/NumPy/torch runtime changed")
    if asset_identity(args) != plan["assets"]:
        raise ValueError("locked assets changed")
    datasets, receipt = load_data(args)
    try:
        if receipt != plan["data_identity"]:
            raise ValueError("locked input identity changed")
        from lunarrisk.nasa_transfer.training import make_arm, train_arm, replay_arm, task_digest, prepare_device, release_device_memory
        from lunarrisk.nasa_transfer.torch_adapter import official_backbone, load_verified_checkpoint
        factory = lambda: official_backbone(args.nasa_config, args.nasa_config_sha256, args.upstream, spatial_policy=args.spatial_policy)
        loader = lambda backbone: load_verified_checkpoint(backbone, args.nasa_checkpoint)
        manifest_sha256 = file_sha256(args.manifest)
        identity = lambda seed, arm: {"manifest_sha256": manifest_sha256, "seed": seed, "arm": arm}
        if args.mode == "replay":
            if args.seed not in budget["seeds"]:
                raise ValueError("undeclared replay seed")
            model, _ = make_arm(factory, loader, seed=args.seed, arm=args.arm, adaptation=budget["adaptation"])
            replay_arm(model, datasets, budget=budget, seed=args.seed, device=args.device, original=args.run_dir,
                       outdir=args.outdir, identity=identity(args.seed, args.arm))
            return 0
        args.outdir.mkdir(parents=True, exist_ok=False)
        write_new(args.outdir / "locked_manifest.json", plan)
        reports = []
        for seed in budget["seeds"]:
            pair = []
            for arm in plan["arms"]:
                if sources() != plan["source_sha256"] or file_sha256(args.manifest) != manifest_sha256:
                    raise ValueError("source/manifest changed during paired sequence")
                name = f"{arm}_seed{seed}"
                directory = args.outdir / name
                import torch
                device = prepare_device(args.device, seed)
                baseline_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                model, coverage = make_arm(factory, loader, seed=seed, arm=arm, adaptation=budget["adaptation"])
                if pair and task_digest(model) != pair[0]["initial_task_sha256"]:
                    raise ValueError("paired task-head/adapter initialization differs before scoring")
                report = train_arm(model, datasets, budget=budget, seed=seed, arm=arm, device=args.device,
                                   outdir=directory, identity=identity(seed, arm), coverage=coverage)
                del model
                memory_release = release_device_memory(args.device, baseline_allocated)
                command = replay_command(args, directory, args.outdir / (name + "_replay"), seed, arm)
                with (args.outdir / (name + "_replay.log")).open("x") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
                replay_receipt = json.loads((args.outdir / (name + "_replay") / "replay.json").read_text())
                if replay_receipt["status"] != "PASS" or replay_receipt["identity"] != identity(seed, arm):
                    raise ValueError("fresh process replay receipt mismatch")
                write_new(directory / "completion.json", {"status": "PASS", "identity": identity(seed, arm),
                          "report_sha256": file_sha256(directory / "report.json"), "replay": replay_receipt, "memory_release": memory_release})
                reports.append(report)
                pair.append(report)
                print("VERIFIED " + name, flush=True)
            if pair[0]["initial_task_sha256"] != pair[1]["initial_task_sha256"]:
                raise ValueError("paired task-head/adapter initialization differs")
            for first, second in zip(pair[0]["history"], pair[1]["history"]):
                if first["training_order_sha256"] != second["training_order_sha256"]:
                    raise ValueError("paired minibatch orders differ")
        if sources() != plan["source_sha256"] or file_sha256(args.manifest) != manifest_sha256:
            raise ValueError("source/manifest changed before final aggregation")
        summary = paired_summary(reports)
        summary.update(status="COMPLETE", manifest_sha256=manifest_sha256, test_scored=False,
                       actual_runs=len(reports), official_model_execution="PASS")
        write_new(args.outdir / "summary.json", summary)
        print(json.dumps({"status": "COMPLETE", "runs": len(reports), "test_scored": False}))
    finally:
        close_data(datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
