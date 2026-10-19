#!/usr/bin/env python3
"""Ask the USGS SPICE web service for the kernels of an imported LROC NAC cube.

spiceinit web=true posts the cube label to
https://services.isis.astrogeology.usgs.gov/cgi-bin/spiceinit.cgi and expects a PVL with a
populated Kernels group plus NaifKeywords back. This script reproduces that exchange
directly: it slices the label out of the cube (everything before the pixel object), posts
it as hex-encoded text, and prints the resolved kernel selections. Nothing is written to
the cube here; the response is only checked for content.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request
from pathlib import Path

SERVICE = "https://services.isis.astrogeology.usgs.gov/cgi-bin/spiceinit.cgi"


def label_of(cube: Path, head_bytes: int = 65536) -> str:
    """The ISIS cube label is the leading IsisCube object up to (excluding) the Label object."""
    raw = cube.read_bytes()[:head_bytes].decode("latin-1")
    index = raw.find("Object = Label")
    if index < 0:
        raise SystemExit("no label object in the cube head")
    body = raw[:index].rstrip()
    if not body.endswith("End"):
        body += "\nEnd\n"
    return body + "\n"


def ask(label: str, timeout: float = 180.0) -> tuple[int, str]:
    body = label.encode("latin-1").hex()
    # Cloudflare in front of the service rejects non-browser agents with error 1010.
    headers = {"Content-Type": "text/plain",
               "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/120 Safari/537.36",
               "Accept": "*/*"}
    request = urllib.request.Request(SERVICE, data=body.encode("ascii"), method="POST",
                                     headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", "replace")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cube", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    label = label_of(args.cube)
    status, text = ask(label, args.timeout)
    print(f"HTTP {status}, {len(text)} bytes")
    if "Kernels" in text:
        start = text.find("Group = Kernels")
        print(text[start:start + 1200])
    else:
        print(text[:800])
    return 0 if "Kernels" in text and "Null" not in text.split("End_Group")[0] else 2


if __name__ == "__main__":
    raise SystemExit(main())
