#!/usr/bin/env python3
"""Prepare a LROC NAC CDR product as an ISIS cube that spiceinit can initialise.

ISIS 8 ships no CDR importer for LROC NAC: ``lronac2isis`` only accepts the EDR data set id,
and ``pds2isis`` (the documented CDR importer) decodes the pixels correctly but leaves the
instrument metadata in the PDS root, where the camera model does not look. The camera needs
the Instrument group that ``lronac2isis`` builds and a Kernels group with the NAIF frame code.

This script runs the exact sequence:

1. ``pds2isis``            CDR IMG -> cube (pixel scaling and NULL handling are correct)
2. ``editlab``             build Instrument (frame id, clock counts, exposure, temperatures,
                           summing) from the PDS label and Kernels (NAIF frame code)
3. ``spiceinit``           with an explicit kernel list, bypassing ISIS kernel databases

Kernel file names come from ``--isisdir`` and are supplied explicitly because the USGS data
area is missing the ``moc_kernels`` database that carries the spacecraft bus attitude.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

LABEL_KEYS = {
    "FrameId": "FRAME_ID",
    "SpacecraftClockPrerollCount": "LRO:SPACECRAFT_CLOCK_PREROLL_COUNT",
    "SpacecraftClockStartCount": "SPACECRAFT_CLOCK_START_COUNT",
    "SpacecraftClockStopCount": "SPACECRAFT_CLOCK_STOP_COUNT",
    "LineExposureDuration": "LINE_EXPOSURE_DURATION",
    "SpatialSumming": "CROSSTRACK_SUMMING",
    "TemperatureSCS": "LRO:TEMPERATURE_SCS",
    "TemperatureFPA": "LRO:TEMPERATURE_FPA",
    "TemperatureFPGA": "LRO:TEMPERATURE_FPGA",
    "TemperatureTelescope": "LRO:TEMPERATURE_TELESCOPE",
    "StartTime": "START_TIME",
    "StopTime": "STOP_TIME",
    "PrerollTime": "LRO:PREROLL_TIME",
    "TemperatureSCSRaw": "LRO:TEMPERATURE_SCS_RAW",
    "TemperatureFPARaw": "LRO:TEMPERATURE_FPA_RAW",
    "TemperatureFPGARaw": "LRO:TEMPERATURE_FPGA_RAW",
    "TemperatureTelescopeRaw": "LRO:TEMPERATURE_TELESCOPE_RAW",
}
FRAME_CODES = {"LEFT": -85600, "RIGHT": -85610}


def label_values(raw: bytes) -> dict:
    head = raw[:8192].decode("latin-1")
    end = head.find("END\r\n")
    head = head[: end if end >= 0 else len(head)]
    head = re.sub(r"/\*.*?\*/", "", head, flags=re.S)
    values = {}
    for isis_key, pds_key in LABEL_KEYS.items():
        match = re.search(rf"^\s*{re.escape(pds_key)}\s*=\s*(.+?)\s*$", head, re.MULTILINE)
        if not match:
            continue
        value = match.group(1).strip().strip('"')
        value = re.sub(r"\s*<[^>]*>", "", value).strip()
        values[isis_key] = value
    return values


def run(*args: str) -> None:
    print("+", " ".join(args))
    subprocess.run(args, check=True)


def normalise_pvl_quotes(cube: Path) -> int:
    """Rewrite `'"…"'` values as plain PVL strings.

    ``editlab`` stores any value containing spaces as a PVL string whose content includes the
    quotes, which breaks ISIS translation lookups (e.g. ``MissionName2DataDir.trn`` reports
    "Unable to find a translation value"). The replacement is space-padded so the file length
    and therefore every absolute StartByte in the label stay valid.
    """
    raw = cube.read_bytes()
    fixed = 0
    for match in re.finditer(rb"'\"([A-Z][A-Z ]*)\"'", raw):
        plain = b'"' + match.group(1) + b'"'
        padded = plain + b" " * (len(match.group(0)) - len(plain))
        raw = raw.replace(match.group(0), padded)
        fixed += 1
    if fixed:
        cube.write_bytes(raw)
    print(f"normalised {fixed} quoted value(s) in {cube}")
    return fixed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="LROC NAC CDR .IMG")
    parser.add_argument("--isisdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output cube")
    parser.add_argument("--ck", nargs="+", required=True,
                        help="attitude kernels (camera CK and spacecraft bus CK)")
    parser.add_argument("--frame-kernel", required=True)
    parser.add_argument("--spk", nargs="+", required=True)
    parser.add_argument("--ik", required=True)
    parser.add_argument("--iak", required=True)
    parser.add_argument("--sclk", required=True)
    parser.add_argument("--lsk", required=True)
    parser.add_argument("--pck", required=True)
    parser.add_argument("--tspk", nargs="+", required=True)
    args = parser.parse_args(argv)

    values = label_values(args.image.read_bytes())
    frame = values.get("FrameId", "LEFT").upper()
    if frame not in FRAME_CODES:
        raise SystemExit(f"unsupported FRAME_ID {frame!r}")
    code = FRAME_CODES[frame]

    run("pds2isis", f"from={args.image}", f"to={args.out}")
    run("editlab", f"from={args.out}", "option=addgrp", "grpname=Instrument")
    instrument = {
        "SpacecraftName": '"LUNAR RECONNAISSANCE ORBITER"',
        "InstrumentId": "NACL" if frame == "LEFT" else "NACR",
        "TargetName": "MOON",
        "CameraVersion": "13",
        **{k: v for k, v in values.items() if k != "FrameId"},
    }
    for key, value in instrument.items():
        run("editlab", f"from={args.out}", "option=addkey", "grpname=Instrument",
            f"keyword={key}", f"value={value}")
    # Keys the camera model reads from the CDR label; pds2isis does not move them to Instrument.
    run("editlab", f"from={args.out}", "option=addkey", "grpname=Instrument",
        "keyword=FrameId", f"value={frame}")
    for key, value in (("SampleFirstPixel", "0"), ("PrerollTime", values.get("PrerollTime", "")),
                       ("MissionPhaseName", "SCIENCE MISSION"),
                       ("InstrumentHostName", '"LUNAR RECONNAISSANCE ORBITER"'),
                       ("InstrumentHostId", "LRO"),
                       ("InstrumentName", "LUNAR RECONNAISSANCE ORBITER NARROW ANGLE CAMERA"),
                       ("TemperatureSCSRaw", values.get("TemperatureSCSRaw", "")),
                       ("TemperatureFPARaw", values.get("TemperatureFPARaw", "")),
                       ("TemperatureFPGARaw", values.get("TemperatureFPGARaw", "")),
                       ("TemperatureTelescopeRaw", values.get("TemperatureTelescopeRaw", ""))):
        if value:
            run("editlab", f"from={args.out}", "option=addkey", "grpname=Instrument",
                f"keyword={key}", f"value={value}")
    run("editlab", f"from={args.out}", "option=addgrp", "grpname=Kernels")
    for key, value in (("NaifIkCode", str(code)), ("NaifFrameCode", str(code))):
        run("editlab", f"from={args.out}", "option=addkey", "grpname=Kernels",
            f"keyword={key}", f"value={value}")

    def join(items):
        return "(" + ",".join(str(item) for item in items) + ")" if len(items) > 1 else str(items[0])

    normalise_pvl_quotes(args.out)
    run("spiceinit", f"from={args.out}", f"LS={args.lsk}", f"SCLK={args.sclk}",
        f"TSPK={join(args.tspk)}", f"PCK={args.pck}", f"IK={args.ik}", f"IAK={args.iak}",
        f"FK={args.frame_kernel}", f"CK={join(args.ck)}", f"SPK={join(args.spk)}")
    print("spiceinit completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
