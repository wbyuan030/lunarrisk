#!/usr/bin/env python3
"""Offline NASA28 direct metadata only: exit 0 match, 1 blocked, 2 audit error.

Never imports NASA/torch or approves a compatible runtime. No installs or network.
Output, if requested, must be a new file in an existing directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-candidate", action="store_true", help="also require Python3.12 and exact torch/vision +cpu metadata")
    parser.add_argument("--output", type=Path, help="exclusive-create JSON, never overwrite")
    args = parser.parse_args(argv)
    try:
        from lunarrisk.nasa_transfer.runtime_metadata import audit_repository
        result = audit_repository(ROOT, cpu_candidate=args.cpu_candidate)
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        print(text, end="")
        return 0 if result["status"] == "DIRECT_METADATA_MATCH" else 1
    except (OSError, ValueError, ImportError) as error:
        print(json.dumps({"status": "AUDIT_ERROR", "error": str(error),
                          "compatibility": "NOT_ESTABLISHED", "upstream_execution": "NOT_RUN",
                          "model_execution": "NOT_RUN"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
