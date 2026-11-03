#!/usr/bin/env python3
"""Serial locked P31 Train/Val baselines; preview by default, --apply executes.

Incomplete or incompatible run directories are deliberately never resumed or
replaced. Investigate them and choose a new output namespace explicitly. This
launcher neither acquires data nor scores Test nor changes the locked budget.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = (("age", 42), ("geology", 42), ("age", 43), ("age", 44),
            ("geology", 43), ("geology", 44))
METRICS = ("accuracy", "macro_f1_full", "macro_f1_supported")
SOURCES = ("scripts/p31_train_classifier.py", "src/lunarrisk/benchmarks/classifier.py",
           "src/lunarrisk/benchmarks/moonstone.py")
BOOTSTRAP = ("import runpy,sys; runtime=sys.argv.pop(1); script=sys.argv.pop(1); "
             "sys.path.insert(0,runtime); sys.argv[0]=script; "
             "runpy.run_path(script,run_name='__main__')")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, minimum=0):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= minimum, "invalid/nonfinite " + label)


def validate_plan(plan):
    require(plan["protocol"] == "P31_03_STORED_DERIVED_SCRATCH_BASELINES_V1",
            "unsupported baseline protocol")
    fixed = {"epochs": 50, "batch_size": 16, "seed": [42, 43, 44], "lr": 0.001,
             "patience": 5, "precision": "fp32", "workers": 0, "optimizer": "AdamW",
             "weight_decay": 0.01, "grad_clip_norm": 1.0, "drop_last": False,
             "configuration_trials_per_task": 1, "additional_age_supervision_for_geology": False}
    require(all(plan["training"][k] == v for k, v in fixed.items()),
            "unsupported change to locked baseline settings")
    require(plan["data"]["track"] == "official_stored"
            and plan["tasks"]["age"]["outputs"] == 5
            and plan["tasks"]["geology"]["outputs"] == 49
            and plan["test"]["runner_allows_test"] is False, "unsupported task/split/Test plan")


def expected_config(plan, task, seed):
    train = plan["training"]
    return {"task": task, "track": plan["data"]["track"], "planned_epochs": train["epochs"],
            "batch_size": train["batch_size"], "workers": train["workers"], "seed": seed,
            "lr": train["lr"], "weight_decay": train["weight_decay"],
            "grad_clip": train["grad_clip_norm"], "patience": train["patience"],
            "precision": train["precision"], "purpose": "baseline",
            "schedule": "upstream cosine with min(2 epochs, total_steps//10) warmup",
            "selection": "strictly higher Val full-K macro-F1; earliest epoch on ties"}


def runtime_directory(explicit=None):
    if explicit is not None:
        runtime = explicit.resolve()
    else:
        spec = importlib.util.find_spec("torch")
        require(spec is not None and spec.origin is not None,
                "torch is unavailable; use the existing trusted --runtime-directory")
        runtime = Path(spec.origin).resolve().parents[1]
    require((runtime / "torch" / "__init__.py").is_file(),
            "runtime directory must contain the existing trusted torch package")
    return runtime


def command_for(args, plan, runtime, task, seed):
    train = plan["training"]
    command = [sys.executable, "-I", "-c", BOOTSTRAP, str(runtime),
               str(ROOT / "scripts/p31_train_classifier.py"),
               "--data", str(args.data), "--stats", str(args.stats),
               "--class-map", str(args.class_map), "--outdir", str(args.outdir / run_name(task, seed)),
               "--purpose", "baseline", "--plan", str(args.plan), "--mode", "train",
               "--task", task, "--track", plan["data"]["track"], "--device", "cuda",
               "--expected-data-sha256", plan["data"]["sha256"]]
    for flag in ("epochs", "batch_size", "workers", "lr", "patience", "precision"):
        command += ["--" + flag.replace("_", "-"), str(train[flag])]
    return command + ["--seed", str(seed)]


def run_name(task, seed):
    return f"{task}_seed{seed}"


def identity_for(args, plan, task, seed):
    return {"config": expected_config(plan, task, seed),
            "inputs": {"data_sha256": plan["data"]["sha256"], "data_bytes": plan["data"]["bytes"],
                       **{k: plan["data"][k] for k in ("stats_sha256", "class_map_sha256")}},
            "source_sha256": {path: sha256(ROOT / path) for path in SOURCES},
            "plan_sha256": sha256(args.plan), "launcher_sha256": sha256(__file__)}


def inspect_checkpoint(path, runtime):
    # A checkpoint produced by our child, loaded without arbitrary pickle globals.
    # Reuse precisely the trusted package directory explicitly passed to the child.
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from lunarrisk.benchmarks.classifier import state_fingerprint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["format_version"] == 1 and
            payload["architecture"] == "P31_scratch_CNN_28_64_128_256",
            "unsupported checkpoint format/architecture")
    for key in ("state_dict", "best_state_dict"):
        require(bool(payload[key]), "empty checkpoint weights")
        require(all(bool(torch.isfinite(value).all()) for value in payload[key].values()),
                "nonfinite checkpoint weights")
    return {**{key: payload[key] for key in
               ("config", "inputs", "source_sha256", "plan_sha256", "splits", "test_evaluated",
                "completed_epochs", "best_epoch", "best_metric", "no_improve", "history")},
            "last_weights_sha256": state_fingerprint(payload["state_dict"]),
            "best_weights_sha256": state_fingerprint(payload["best_state_dict"])}


def check_metrics(metrics, counts, classes, label, prior=False):
    require(metrics["rows"] == sum(counts) and metrics["support"] == counts,
            label + " metric rows/support mismatch")
    matrix = metrics["confusion"]
    require(len(matrix) == classes and all(len(row) == classes for row in matrix)
            and all(type(n) is int and n >= 0 for row in matrix for n in row),
            label + " invalid confusion matrix")
    require([sum(row) for row in matrix] == counts, label + " confusion/support mismatch")
    predicted = [sum(row[c] for row in matrix) for c in range(classes)]
    f1 = [2 * matrix[c][c] / (counts[c] + predicted[c]) if counts[c] + predicted[c] else 0.
          for c in range(classes)]
    actual = {"accuracy": sum(matrix[c][c] for c in range(classes)) / sum(counts),
              "macro_f1_full": sum(f1) / classes,
              "macro_f1_supported": sum(f for f, n in zip(f1, counts) if n) / sum(n > 0 for n in counts)}
    require(metrics["predicted_counts"] == predicted, label + " predicted counts mismatch")
    for key, value in actual.items():
        finite(metrics[key], label + " " + key)
        require(math.isclose(metrics[key], value, abs_tol=1e-12), label + " " + key + " mismatch")
    finite(metrics["multiclass_brier_sum"], label + " Brier")
    if prior and metrics["nll_is_infinite"] is True:
        require(metrics["nll"] is None, label + " infinite prior NLL must be explicit")
    else:
        require(metrics["nll_is_infinite"] is False, label + " nonfinite model NLL")
        finite(metrics["nll"], label + " NLL")


def check_predictions(path, split, split_info, metrics, classes, prior=None):
    ids = hashlib.sha256()
    seen = set()
    counts = [0] * classes
    matrix = [[0] * classes for _ in range(classes)]
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        require(reader.fieldnames == ["sample_id", "split", "target"] + [f"p_{i}" for i in range(classes)],
                "unexpected prediction columns: " + str(path))
        for row in reader:
            sample = row["sample_id"]
            require(sample and sample not in seen and row["split"] == split,
                    "duplicate/invalid prediction ID or split")
            if seen:
                ids.update(b"\n")
            ids.update(sample.encode())
            seen.add(sample)
            target = int(row["target"])
            require(0 <= target < classes, "invalid prediction target")
            probabilities = [float(row[f"p_{i}"]) for i in range(classes)]
            require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                    and math.isclose(sum(probabilities), 1., abs_tol=2e-5),
                    "nonfinite/unnormalized probabilities")
            if prior is not None:
                require(all(math.isclose(p, q, abs_tol=1e-7) for p, q in zip(probabilities, prior)),
                        "prior predictions differ from Train frequencies")
            counts[target] += 1
            matrix[target][max(range(classes), key=probabilities.__getitem__)] += 1
    require(len(seen) == split_info["rows"] and ids.hexdigest() == split_info["ordered_ids_sha256"],
            "prediction rows/ordered IDs mismatch")
    require(counts == split_info["class_counts"] and matrix == metrics["confusion"],
            "predictions do not reproduce report confusion")
    return seen


def verify_run(directory, log, identity, plan, runtime):
    report = read_json(directory / "report.json")
    checkpoint = inspect_checkpoint(directory / "checkpoint.pt", runtime)
    asset = read_json(directory / "asset_manifest.json")
    require(report["status"] == "P31_BASELINE_TRAINVAL" and report["purpose"] == "baseline"
            and report["mode"] == "train", "not a formal training report")
    for evidence in (report, checkpoint):
        require(evidence["test_evaluated"] is False, "Test scoring forbidden")
        for key in ("config", "inputs", "source_sha256"):
            require(evidence[key] == identity[key], "run " + key + " mismatch")
        require(set(evidence["splits"]) == {"train", "val"}, "unexpected split")
    require(checkpoint["plan_sha256"] == identity["plan_sha256"], "checkpoint plan mismatch")
    require(checkpoint["splits"] == report["splits"], "checkpoint/report splits mismatch")
    require(all(asset[key] == value for key, value in identity["inputs"].items())
            and asset["full_content_hash_verified_this_invocation"] is True
            and asset["data_revision"] == plan["data"]["revision"], "asset manifest mismatch")
    require(set(report["metrics"]) == set(report["train_prior_metrics"]) ==
            set(report["prediction_sha256"]) == {"train", "val"}, "unexpected metric/prediction split")
    task = identity["config"]["task"]
    classes = plan["tasks"][task]["outputs"]
    prediction_ids = {}
    train_counts = report["splits"]["train"]["class_counts"]
    require(len(train_counts) == classes and sum(train_counts) == plan["data"]["train_rows"],
            "invalid Train class counts")
    prior = [n / sum(train_counts) for n in train_counts]
    require(report["train_prior_probabilities"] == prior, "prior not fitted to Train counts")
    for split in ("train", "val"):
        info = report["splits"][split]
        require(info["rows"] == plan["data"][split + "_rows"]
                and info["ordered_ids_sha256"] == plan["data"][split + "_ordered_ids_sha256"]
                and len(info["class_counts"]) == classes
                and all(type(n) is int and n >= 0 for n in info["class_counts"])
                and sum(info["class_counts"]) == info["rows"], "official split counts/IDs mismatch")
        for kind, metrics in (("", report["metrics"][split]), ("_prior", report["train_prior_metrics"][split])):
            check_metrics(metrics, info["class_counts"], classes, split + kind, prior=bool(kind))
            finite(metrics["largest_observed_batch"], "largest batch", minimum=1)
            require(metrics["largest_observed_batch"] <= identity["config"]["batch_size"], "batch budget exceeded")
            path = directory / (split + kind + "_predictions.csv")
            current_ids = check_predictions(path, split, info, metrics, classes, prior if kind else None)
            if not kind:
                require(sha256(path) == report["prediction_sha256"][split], "prediction SHA256 mismatch")
                prediction_ids[split] = current_ids
    require(not prediction_ids["train"] & prediction_ids["val"], "Train/Val ID overlap")
    completed = checkpoint["completed_epochs"]
    history = checkpoint["history"]
    require(type(completed) is int and 1 <= completed <= plan["training"]["epochs"]
            and len(history) == completed, "incomplete/invalid epoch history")
    best, best_epoch, stale = -1., 0, 0
    for epoch, row in enumerate(history, 1):
        require(row["epoch"] == epoch and row["train_rows"] == plan["data"]["train_rows"]
                and row["largest_train_batch"] == min(identity["config"]["batch_size"], row["train_rows"]),
                "invalid training history counts/batch")
        finite(row["train_loss"], "training loss")
        finite(row["train_seconds"], "training seconds", minimum=1e-12)
        finite(row["train_samples_per_second"], "training throughput", minimum=1e-12)
        check_metrics(row["val"], report["splits"]["val"]["class_counts"], classes, "epoch Val")
        require(stale < plan["training"]["patience"], "training continued after locked early stop")
        score = row["val"]["macro_f1_full"]
        if score > best:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
    require(completed == plan["training"]["epochs"] or stale == plan["training"]["patience"],
            "run stopped before locked budget/early stopping")
    require(checkpoint["best_epoch"] == best_epoch and checkpoint["best_metric"] == best
            and checkpoint["no_improve"] == stale, "incorrect Val selection/early stopping")
    for key in ("completed_epochs", "best_epoch", "last_weights_sha256", "best_weights_sha256"):
        require(report[key] == checkpoint[key], "checkpoint/report " + key + " mismatch")
    require(report["metrics"]["val"] == history[best_epoch - 1]["val"], "selected Val metrics mismatch")
    require(report["train_examples_this_process"] == completed * plan["data"]["train_rows"],
            "training example count mismatch")
    for key in ("peak_process_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes",
                "train_window_seconds_this_process", "train_samples_per_second_this_process"):
        finite(report[key], key, minimum=1e-12)
    require(report["peak_gpu_reserved_bytes"] >= report["peak_gpu_allocated_bytes"], "invalid GPU peaks")
    files = ["report.json", "checkpoint.pt", "asset_manifest.json"] + [
        split + suffix + "_predictions.csv" for split in ("train", "val") for suffix in ("", "_prior")]
    hashes = {name: sha256(directory / name) for name in files}
    hashes["stdout"] = sha256(log)
    return report, {"identity": identity, "artifact_sha256": hashes, "verification": "PASS"}


def aggregate(completed):
    tasks = {}
    for task in ("age", "geology"):
        runs = [(seed, report) for run_task, seed, report in completed if run_task == task]
        groups = {}
        for baseline, field in (("independent_task_scratch_cnn", "metrics"),
                                ("train_empirical_prior", "train_prior_metrics")):
            groups[baseline] = {}
            for split in ("train", "val"):
                groups[baseline][split] = {}
                for metric in METRICS:
                    values = {str(seed): report[field][split][metric] for seed, report in runs}
                    numbers = list(values.values())
                    groups[baseline][split][metric] = {
                        "per_seed": values, "n": len(numbers),
                        "mean": statistics.mean(numbers) if numbers else None,
                        "sample_std_ddof1": statistics.stdev(numbers) if len(numbers) > 1 else None}
        tasks[task] = {"completed_seeds": [seed for seed, _ in runs],
                       "missing_seeds": [s for s in (42, 43, 44) if s not in [seed for seed, _ in runs]],
                       "baselines": groups}
    return {"status": "COMPLETE" if len(completed) == len(SEQUENCE) else "PARTIAL",
            "test_evaluated": False, "std_definition": "sample standard deviation, ddof=1; null for n<2",
            "prior_note": "Train-only deterministic prior repeated per seed; repetitions are not independent prior fits",
            "tasks": tasks}


def save_aggregate(outdir, completed):
    # Immutable snapshots preserve the last successful partial result on failure.
    value = aggregate(completed)
    path = outdir / f"aggregate_{len(completed):02d}.json"
    if path.exists():
        require(read_json(path) == value, "existing aggregate differs from verified runs")
    else:
        write_new(path, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=ROOT / "reports/evidence/p30_20261103/hf_channel_stats.json")
    parser.add_argument("--class-map", type=Path, default=ROOT / "reports/evidence/p30_20261103/geologic_units.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "protocols/p31_baseline_eval_plan.json")
    parser.add_argument("--runtime-directory", type=Path,
                        help="existing trusted package directory containing torch; never a downloaded-data directory")
    parser.add_argument("--max-new-runs", type=int,
                        help="limit newly started runs this invocation; reuse verified runs, keep each locked budget")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="execute fixed serial Train/Val runs")
    mode.add_argument("--dry-run", action="store_true", help="preview only (the default), no outputs or data hashing")
    args = parser.parse_args(argv)
    if args.max_new_runs is not None and args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    for key in ("data", "outdir", "stats", "class_map", "plan"):
        setattr(args, key, getattr(args, key).resolve())
    plan = read_json(args.plan)
    validate_plan(plan)
    runtime = runtime_directory(args.runtime_directory)
    commands = [command_for(args, plan, runtime, task, seed) for task, seed in SEQUENCE]
    if not args.apply:
        print(json.dumps({"mode": "DRY_RUN", "plan_sha256": sha256(args.plan), "commands": commands,
                          "max_new_runs": args.max_new_runs,
                          "note": "No data acquisition or Test scoring; --apply executes fixed runs up to the invocation limit."}, indent=2))
        return 0
    require(args.data.stat().st_size == plan["data"]["bytes"]
            and sha256(args.data) == plan["data"]["sha256"], "data size/SHA256 differs from locked plan")
    for path, key in ((args.stats, "stats_sha256"), (args.class_map, "class_map_sha256")):
        require(sha256(path) == plan["data"][key], key + " differs from locked plan")
    # Freeze one source/plan identity for the whole sequence, including restarts.
    # A concurrent edit must stop the sequence rather than mix configurations.
    base_identity = identity_for(args, plan, "age", 42)
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Advisory flock is released by the OS on interruption and never removes evidence.
    import fcntl
    with (args.outdir / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []
        new_runs = 0
        try:
            for (task, seed), command in zip(SEQUENCE, commands):
                name = run_name(task, seed)
                directory, log = args.outdir / name, args.outdir / (name + ".stdout.log")
                identity = {**base_identity, "config": expected_config(plan, task, seed)}
                require(identity_for(args, plan, task, seed) == identity,
                        "source/plan changed during the serial baseline sequence")
                receipt = directory / "completion.json"
                if directory.exists() or log.exists():
                    require(receipt.is_file() and log.is_file(),
                            "incomplete output; preserve and investigate before using a new namespace: " + name)
                    saved = read_json(receipt)
                    require(saved["identity"] == identity, "completed run identity mismatch: " + name)
                    report, verified = verify_run(directory, log, identity, plan, runtime)
                    require(saved == verified, "completed artifacts changed: " + name)
                    print("REUSED " + name, flush=True)
                else:
                    if args.max_new_runs is not None and new_runs >= args.max_new_runs:
                        break
                    with log.open("x") as stdout:
                        subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
                    report, verified = verify_run(directory, log, identity, plan, runtime)
                    require(identity_for(args, plan, task, seed) == identity,
                            "source/plan changed during the serial baseline sequence")
                    write_new(receipt, verified)
                    new_runs += 1
                    print("VERIFIED " + name, flush=True)
                completed.append((task, seed, report))
                save_aggregate(args.outdir, completed)
        except Exception:
            save_aggregate(args.outdir, completed)
            raise
        summary = save_aggregate(args.outdir, completed)
        print(json.dumps({"status": summary["status"], "new_runs": new_runs,
                          "completed": len(completed),
                          "remaining": [run_name(task, seed) for task, seed in SEQUENCE[len(completed):]],
                          "aggregate": str(args.outdir / f"aggregate_{len(completed):02d}.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
