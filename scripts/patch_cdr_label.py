#!/usr/bin/env python3
"""Make an ISIS-ingestible label copy of a LROC NAC CDR for the EDR importer.

The official CDR product (DATA_SET_ID LRO-L-LROC-3-CDR-V1.0) is rejected by lronac2isis,
which only accepts the EDR data set id. Copying the file and patching that single label
string lets the LROC NAC importer attach the Instrument/NaifKeywords groups it needs.
Only the label bytes are rewritten; the pixel block is copied verbatim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

OLD = b'"LRO-L-LROC-3-CDR-V1.0"'
NEW = b'"LRO-L-LROC-2-EDR-V1.0"'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.source.read_bytes()
    head = raw[:8192]
    index = head.find(OLD)
    if index < 0:
        raise SystemExit(f"{args.source}: CDR data set id not present in the label head")
    end = min(index + len(NEW), 8192)
    args.out.write_bytes(raw[:index] + NEW + raw[end:])
    print(f"wrote {args.out} (label patched at byte {index}; pixel bytes copied verbatim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
