#!/usr/bin/env python3
"""Reproduce the ISIS 8 spiceinit web request against the USGS ALE SPICE service.

The client posts ``application/x-www-form-urlencoded`` to
``<URL>?mission=Lro&instrument=NACL`` with one field, ``name``, whose value is the hex
encoding of an ``<input_label>`` XML document: the hex-encoded ISIS version, a
``<parameters>`` block selecting kernel qualities (reconstructed, not predicted, no
start/end padding, system shape model), and the hex-encoded cube label.

This script sends that exact shape and prints the service response, so the kernel
selections can be checked without running the local kernel-database resolution.
"""
from __future__ import annotations

import argparse
import binascii
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SERVICE = "https://astrogeology.usgs.gov/apis/ale/v0.9.1/spiceserver/"
ISIS_VERSION = "8.3.0"
PARAMETERS = """    <parameters>
      <cksmithed value='No' />
      <ckrecon value='Yes' />
      <ckpredicted value='No' />
      <cknadir value='No' />
      <spksmithed value='No' />
      <spkrecon value='Yes' />
      <spkpredicted value='No' />
      <shape value='system' />
      <startpad time='0.0' />
      <endpad time='0.0' />
    </parameters>"""


def cube_label(cube: Path, head_bytes: int = 65536) -> str:
    raw = cube.read_bytes()[:head_bytes].decode("latin-1")
    index = raw.find("Object = Label")
    if index < 0:
        raise SystemExit("no label object in the cube head")
    label = raw[:index].rstrip()
    if not label.endswith("End"):
        label += "\nEnd\n"
    return label


def input_document(label: str) -> str:
    return ("<input_label>\n  <isis_version>\n"
            + binascii.hexlify(ISIS_VERSION.encode()).decode()
            + "\n  </isis_version>\n" + PARAMETERS
            + "\n  <label>" + binascii.hexlify(label.encode("latin-1")).decode()
            + "</label>\n</input_label>")


def ask(document: str, mission: str, instrument: str, timeout: float = 180.0) -> tuple[int, str]:
    payload = urllib.parse.urlencode({"name": binascii.hexlify(document.encode("ascii")).decode()})
    request = urllib.request.Request(
        f"{SERVICE}?mission={urllib.parse.quote(mission)}&instrument={urllib.parse.quote(instrument)}",
        data=payload.encode("ascii"), method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 Safari/537.36"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cube", type=Path)
    parser.add_argument("--mission", default="Lro")
    parser.add_argument("--instrument", default="NACL")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    document = input_document(cube_label(args.cube))
    status, text = ask(document, args.mission, args.instrument, args.timeout)
    print(f"HTTP {status}, {len(text)} bytes")
    print(text[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
