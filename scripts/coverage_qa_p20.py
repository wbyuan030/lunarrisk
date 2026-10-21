"""P20-03 pilot: multi-camera coverage QA per scene on the matched DTM grid.

The right-camera export writes a different NULL sentinel than the left one, so the fill value
is detected per camera as the most frequent value (ISIS NULL maps to one constant in the raw
float export). Both cameras share the DTM grid, so coverage is pure mask arithmetic.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile

PIXEL_M = 2.0


def load(path: Path, lines: int, samples: int):
    raw = np.fromfile(path, dtype="<f4")
    assert raw.size == lines * samples, (path, raw.size)
    grid = raw.reshape(lines, samples).astype(np.float64)
    values, counts = np.unique(grid, return_counts=True)
    fill = values[counts.argmax()]
    if counts.max() / grid.size < 0.01:
        return np.where(np.isfinite(grid), grid, np.nan), None
    return np.where(grid == fill, np.nan, grid), (float(fill), float(counts.max() / grid.size))


def acquisition_unions(items):
    """Group per-camera valid masks by acquisition id (product base, e.g. M136648682).

    items: iterable of (image_name, camera_tag, mask). Returns {acq: {"tags": set, "mask": arr}}.
    """
    acqs = {}
    for name, tag, mask in items:
        match = re.match(r"(M\d+)", name)
        if not match:
            continue
        entry = acqs.setdefault(match.group(1), {"tags": set(), "mask": np.zeros_like(mask)})
        entry["tags"].add(tag)
        entry["mask"] |= mask
    return acqs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--dtm", type=Path, required=True)
    parser.add_argument("--lines", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    dtm = tifffile.imread(args.dtm).astype(np.float64)
    dtm_valid = np.isfinite(dtm) & (np.abs(dtm) < 1e30)
    assert dtm.shape == (args.lines, args.samples), (dtm.shape, args.lines, args.samples)

    cams = {}
    for path in sorted(args.dir.glob("*_ortho.raw")):
        tag = "left" if "_L_ortho" in path.name else ("right" if "_R_ortho" in path.name else None)
        if tag is None:
            continue
        image, fill = load(path, args.lines, args.samples)
        ok = np.isfinite(image)
        cams.setdefault(tag, []).append((path.name, image, ok, fill))

    total = int(dtm_valid.sum())
    report = {"scene": args.dir.name, "dtm_valid": total,
              "dtm_valid_fraction": round(float(dtm_valid.mean()), 6), "cameras": {}}
    union = np.zeros_like(dtm_valid)
    per_union = {}
    for tag in sorted(cams):
        entries = []
        for name, image, ok, fill in cams[tag]:
            entries.append({"image": name,
                            "valid_fraction": round(float(ok.mean()), 6),
                            "dtm_coverage": round(float((ok & dtm_valid).sum() / total), 6),
                            "fill": fill,
                            "if_p50": round(float(np.nanmedian(image)), 6),
                            "if_max": round(float(np.nanmax(image)), 6)})
        union_tag = np.zeros_like(dtm_valid)
        for _, _, ok, _ in cams[tag]:
            union_tag |= ok
        report["cameras"][tag] = {"images": entries,
                                  "union_dtm_coverage": round(float((union_tag & dtm_valid).sum() / total), 6)}
        union |= union_tag

    both = np.zeros_like(dtm_valid)
    if "left" in cams and "right" in cams:
        left = np.zeros_like(dtm_valid)
        right = np.zeros_like(dtm_valid)
        for _, _, ok, _ in cams["left"]:
            left |= ok
        for _, _, ok, _ in cams["right"]:
            right |= ok
        both = left & right
        lv = np.full(both.shape, np.nan)
        rv = np.full(both.shape, np.nan)
        for name, image, ok, fill in cams["left"]:
            slot = both & ok & ~np.isfinite(lv)
            lv[slot] = image[slot]
        for name, image, ok, fill in cams["right"]:
            slot = both & ok & ~np.isfinite(rv)
            rv[slot] = image[slot]
        report["overlap"] = {
            "dtm_fraction": round(float((both & dtm_valid).sum() / total), 6),
            "grid_fraction": round(float(both.mean()), 6),
            "residual_mean_if": round(float(np.nanmean(lv - rv)), 6) if np.isfinite(lv).any() else None,
            "residual_std_if": round(float(np.nanstd(lv - rv)), 6) if np.isfinite(lv).any() else None,
            "residual_pixels": int(np.isfinite(lv - rv).sum()),
        }
    # Per-acquisition coverage (L+R of the same observation). This is the scheduling unit:
    # a single acquisition rarely reaches the 0.95 target on its own.
    acqs = acquisition_unions((name, tag, ok)
                              for tag in sorted(cams) for name, _, ok, _ in cams[tag])
    report["acquisitions"] = {
        acq: {"camera_tags": sorted(entry["tags"]),
              "dtm_coverage": round(float((entry["mask"] & dtm_valid).sum() / total), 6),
              "grid_fraction": round(float(entry["mask"].mean()), 6)}
        for acq, entry in sorted(acqs.items())}
    report["union_dtm_coverage"] = round(float((union & dtm_valid).sum() / total), 6)
    report["union_grid_fraction"] = round(float(union.mean()), 6)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
