"""Execute the fixed P2.2 validation sequence with raw logs and explicit exit codes.

Run from an immutable source snapshot inside the existing ROCm container. Output directories must
be new; historical experiments and input labels/cache are read-only. This does not run Gate 4.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

import train_posterior_p21 as runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.run.exists():
        raise ValueError("run directory already exists; preserve it and choose a new path")
    args.logs.mkdir(parents=True, exist_ok=False)
    root = Path(runner.ROOT)
    manifest = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "code_sha": runner.git_sha(), "source_sha256": runner.source_fingerprints(),
                "labels_sha256": runner.file_sha256(args.labels),
                "cache_sha256": runner.file_sha256(args.cache),
                "protocol_sha256": {str(p.relative_to(root)): runner.file_sha256(p)
                                    for p in sorted((root / "protocols").glob("*")) if p.is_file()},
                "environment": {"torch": torch.__version__, "hip": torch.version.hip,
                                "python": sys.version, "kfd": Path("/dev/kfd").exists(),
                                "dri": sorted(p.name for p in Path("/dev/dri").glob("*"))},
                "commands": []}
    if not torch.cuda.is_available():
        raise RuntimeError("local ROCm GPU required for this run")
    manifest["environment"]["device"] = torch.cuda.get_device_name(0)
    tasks = [
        ("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"], 0),
        ("compile", ["-m", "compileall", "-q", "scripts", "src", "tests"], 0),
        ("comparison", ["scripts/compare_p22.py", "--labels", str(args.labels), "--cache", str(args.cache),
                        "--outdir", str(args.run), "--seeds", "42,43,44", "--epochs", "40",
                        "--draws", "4096", "--bootstrap-draws", "2000"], 0),
        ("replay", ["scripts/audit_p22_artifacts.py", "--run", str(args.run), "--cache", str(args.cache),
                    "--repetitions", "2"], 0),
        ("diagnostic", ["scripts/diagnose_p22_rules.py", "--run", str(args.run), "--cache", str(args.cache),
                        "--outdir", str(args.run / "rule_diagnostic")], 0),
        ("collision_v1", ["scripts/check_group_pool_collision.py", "--out", str(args.logs / "collision_v1.json")], 2),
        ("collision_v2", ["scripts/check_group_pool_collision.py", "--pool", "protocols/heldout_v2_group_pool_v2.csv",
                          "--protocol", "protocols/heldout_v2_external_protocol_v2.json",
                          "--out", str(args.logs / "collision_v2.json")], 0),
    ]
    for name, arguments, expected in tasks:
        command = [sys.executable, *arguments]
        start = time.monotonic()
        with (args.logs / f"{name}.log").open("x") as log:
            status = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode
        record = {"task": name, "command": command, "exit_code": status, "expected_exit_code": expected,
                  "wall_seconds": time.monotonic()-start,
                  "log_sha256": runner.file_sha256(args.logs / f"{name}.log")}
        manifest["commands"].append(record)
        manifest["passed"] = all(r["exit_code"] == r["expected_exit_code"] for r in manifest["commands"])
        (args.logs / "execution.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if status != expected:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
