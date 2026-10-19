#!/usr/bin/env python3
"""Read a PDS3 LROC NAC CDR image pair and report I/O statistics (P1.9 Gate 2 optical probe).

LROC NAC CDR products are PDS3 FIXED_LENGTH files: the first record carries the label and
`^IMAGE = <record>` points at the first image record (1-based), so the pixel array starts at
byte ``(record - 1) * RECORD_BYTES``. Pixels are LSB_INTEGER 16-bit with a linear
``SCALING_FACTOR`` to I/F and a ``NULL`` sentinel. This script verifies that layout from the
label itself, then reports the geometry, the valid fraction and the I/F distribution for one
or more images without touching the DTM or any derived product.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

KEY_RE = r"^\s*{key}\s*=\s*(.+?)\s*$"
# '^IMAGE' is a PDS pointer keyword, not a plain KEY = value assignment.
POINTER_RE = r"^\s*{key}\s*=\s*(\d+)\s*$"


def read_label(raw: bytes) -> dict:
    head = raw[:8192].decode("latin-1")
    end = head.find("END\r\n")
    if end < 0:
        raise ValueError("no PDS END keyword in the first 8 KiB")
    label = {"label_sha256": hashlib.sha256(head[:end + 5].encode("latin-1")).hexdigest()}

    def number(key):
        match = re.search(KEY_RE.format(key=key), head, re.MULTILINE)
        if not match:
            raise KeyError(key)
        return float(match.group(1).split("<")[0].strip())

    def text(key):
        match = re.search(KEY_RE.format(key=key), head, re.MULTILINE)
        return match.group(1).strip().strip('"') if match else None

    label["record_bytes"] = int(number("RECORD_BYTES"))
    pointer = re.search(POINTER_RE.format(key=re.escape("^IMAGE")), head, re.MULTILINE)
    if not pointer:
        raise KeyError("^IMAGE")
    label["image_record"] = int(pointer.group(1))
    label["lines"] = int(number("LINES"))
    label["line_samples"] = int(number("LINE_SAMPLES"))
    label["sample_bits"] = int(number("SAMPLE_BITS"))
    label["scaling_factor"] = number("SCALING_FACTOR")
    # PDS3 prints signed 16-bit NULL values ("-32768"); the raw buffer is unsigned.
    label["null_dn"] = int(number("NULL")) & 0xFFFF
    label["product_id"] = text("PRODUCT_ID")
    label["data_set_id"] = text("DATA_SET_ID")
    label["start_time"] = text("START_TIME")
    label["stop_time"] = text("STOP_TIME")
    label["orbit_number"] = int(number("ORBIT_NUMBER"))
    label["mission_phase_name"] = text("MISSION_PHASE_NAME")
    label["unit"] = text("UNIT")
    for key in ("LRO:CHANNEL_A_OFFSET", "LRO:CHANNEL_B_OFFSET"):
        label[key.lower().replace(":", "_")] = number(key) if re.search(
            KEY_RE.format(key=re.escape(key)), head, re.MULTILINE) else None
    return label


def load_image(path: Path) -> tuple[np.ndarray, dict]:
    raw = path.read_bytes()
    label = read_label(raw)
    start = (label["image_record"] - 1) * label["record_bytes"]
    lines, samples = label["lines"], label["line_samples"]
    if label["sample_bits"] != 16:
        raise ValueError(f"expected 16-bit samples, got {label['sample_bits']}")
    bytes_needed = lines * samples * 2
    available = len(raw) - start
    if available < bytes_needed:
        raise ValueError(f"{path}: image truncated, need {bytes_needed} bytes, found {available}")
    label["pixel_offset"] = start
    label["bytes_after_label"] = available - bytes_needed
    label["file_bytes"] = len(raw)
    pixels = np.frombuffer(raw, dtype="<u2", count=lines * samples, offset=start).reshape(lines, samples)
    return pixels, label


def statistics(pixels: np.ndarray, label: dict) -> dict:
    null = pixels == np.uint16(label["null_dn"])
    valid = pixels[~null].astype(np.float32) * label["scaling_factor"]
    return {
        "shape": [int(label["lines"]), int(label["line_samples"])],
        "valid_fraction": round(float((~null).mean()), 6),
        "saturated_fraction": round(float((valid >= 0.999).mean()), 6),
        "if_percentiles": {
            "p01": round(float(np.percentile(valid, 1)), 5),
            "p50": round(float(np.percentile(valid, 50)), 5),
            "p99": round(float(np.percentile(valid, 99)), 5),
        },
        "if_max": round(float(valid.max()), 5),
        "lines_fully_null": int(null.all(axis=1).sum()),
        "lines_with_any_null": int(null.any(axis=1).sum()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="PDS3 IMG file(s)")
    parser.add_argument("--out", type=Path, default=Path("outputs/p19_nac_cdr_probe.json"))
    args = parser.parse_args(argv)

    report = {"images": []}
    for path in args.images:
        pixels, label = load_image(path)
        entry = {"file": path.name, "label": label, "stats": statistics(pixels, label)}
        report["images"].append(entry)
        print(f"{path.name}: shape={entry['stats']['shape']} valid={entry['stats']['valid_fraction']:.5f} "
              f"If p50={entry['stats']['if_percentiles']['p50']} sat={entry['stats']['saturated_fraction']:.5f} "
              f"pixel_offset={label['pixel_offset']} scale={label['scaling_factor']:.3e} null={label['null_dn']}")
    if len(report["images"]) == 2:
        left, _ = load_image(args.images[0])
        right, _ = load_image(args.images[1])
        # A central window exists in any full-resolution NAC frame; guard small probes.
        rows = min(2000, left.shape[0] // 4)
        cols = min(1000, left.shape[1] // 4)
        centre_r, centre_c = left.shape[0] // 2, left.shape[1] // 2
        window = (slice(centre_r - rows, centre_r + rows), slice(centre_c - cols, centre_c + cols))
        left_scale, right_scale = (report["images"][i]["label"]["scaling_factor"] for i in (0, 1))
        report["stereo_overlap"] = {
            "window": [window[0].start, window[0].stop, window[1].start, window[1].stop],
            "left_window_mean_if": round(float(left[window].mean()) * left_scale, 6),
            "right_window_mean_if": round(float(right[window].mean()) * right_scale, 6),
            "note": "raw frame overlap only; georeferencing requires SPICE/ISIS work",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
