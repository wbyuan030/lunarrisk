"""P20-02 registration QA: report residual distribution over geographically spread matches.

Edge features are extracted from the ortho I/F image and from a hillshade of the DTM, matched
by nearest neighbour, and matched pairs are required to be spread over the shared valid band
(quadrants are computed inside the *shared* area, not over the whole frame) so that "spread"
is not an artefact of empty regions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tifffile

PIXEL_M = 2.0


def sobel(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    kernel_y = kernel_x.T
    padded = np.pad(image, 1, mode="edge")
    gx = np.zeros_like(image, dtype=np.float64)
    gy = np.zeros_like(image, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            gx += kernel_x[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]
            gy += kernel_y[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return np.hypot(gx, gy)


def hillshade(dtm: np.ndarray, valid: np.ndarray, sun_el_deg: float, sun_az_deg: float):
    filled = np.where(valid, dtm, np.nanmedian(dtm[valid]))
    gy, gx = np.gradient(filled, PIXEL_M)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    return (math.sin(math.radians(sun_el_deg)) * np.cos(slope)
            + math.cos(math.radians(sun_el_deg)) * np.sin(slope)
            * np.cos(math.radians(sun_az_deg) - aspect))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ortho", type=Path, required=True)
    parser.add_argument("--dtm", type=Path, required=True)
    parser.add_argument("--max-matches", type=int, default=6000)
    parser.add_argument("--shift-samples", type=int, default=1)
    parser.add_argument("--edge-percentile", type=float, default=99.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/p20_registration_qa.json"))
    args = parser.parse_args(argv)

    ortho = np.fromfile(args.ortho, dtype="<f4").reshape(14259, 2389).astype(np.float64)
    values, counts = np.unique(ortho, return_counts=True)
    fill = values[counts.argmax()]
    ortho = np.where(ortho == fill, np.nan, ortho)
    if args.shift_samples:
        ortho = np.roll(ortho, args.shift_samples, axis=1)

    dtm = tifffile.imread(args.dtm).astype(np.float64)
    dtm_valid = np.isfinite(dtm) & (np.abs(dtm) < 1e30)
    shared = np.isfinite(ortho) & dtm_valid

    e_ortho = sobel(np.where(shared, ortho, np.nanmedian(ortho[shared])))
    shade = hillshade(dtm, dtm_valid, 30.0, 270.0)
    e_shade = sobel(shade)

    # quadrants over the shared band only
    rows = np.where(shared.any(axis=1))[0]
    cols = np.where(shared.any(axis=0))[0]
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    rmid, cmid = (r0 + r1) // 2, (c0 + c1) // 2
    label = np.zeros_like(shared, dtype=np.uint8)
    label[:rmid, :cmid] = 1
    label[:rmid, cmid:] = 2
    label[rmid:, :cmid] = 3
    label[rmid:, cmid:] = 4

    results = {}
    for name, dtm_response in (("relief", sobel(np.where(dtm_valid, dtm, np.nanmedian(dtm[dtm_valid])))),
                               ("hillshade", e_shade)):
        threshold_o = np.percentile(e_ortho[shared], args.edge_percentile)
        threshold_d = np.percentile(dtm_response[shared], args.edge_percentile)
        points_o = np.argwhere((e_ortho >= threshold_o) & shared)
        points_d = np.argwhere((dtm_response >= threshold_d) & shared)
        def thin(arr):
            if len(arr) <= args.max_matches:
                return arr
            step = int(math.ceil(len(arr) / args.max_matches))
            return arr[::step]

        points_o = thin(points_o)
        points_d = thin(points_d)
        keys = {}
        residuals, quads, outliers = [], {1: 0, 2: 0, 3: 0, 4: 0}, 0
        for y, x in points_o:
            key = (y // 12, x // 12)
            if key in keys:
                continue
            keys[key] = True
            window = points_d[(abs(points_d[:, 0] - y) < 6) & (abs(points_d[:, 1] - x) < 6)]
            if not len(window):
                continue
            dy = float(np.mean(window[:, 0]) - y)
            dx = float(np.mean(window[:, 1]) - x)
            residuals.append(math.hypot(dx, dy) * PIXEL_M)
            quads[int(label[y, x])] += 1
        residuals = np.array(residuals)
        results[name] = {
            "features": {"ortho": int(len(points_o)), "dtm": int(len(points_d))},
            "matches": int(len(residuals)),
            "quadrant_matches": quads,
            "residual_m": {
                "mean": round(float(residuals.mean()), 3),
                "rms": round(float(np.sqrt((residuals ** 2).mean())), 3),
                "p50": round(float(np.percentile(residuals, 50)), 3),
                "p95": round(float(np.percentile(residuals, 95)), 3),
                "max": round(float(residuals.max()), 3),
            } if len(residuals) else None,
        }

    report = {
        "ortho": str(args.ortho), "dtm": str(args.dtm), "pixel_m": PIXEL_M,
        "shared_valid_fraction": round(float(shared.mean()), 6),
        "shared_band": {"rows": [int(r0), int(r1)], "cols": [int(c0), int(c1)]},
        "control_shift_samples": args.shift_samples,
        "control_shift_m": args.shift_samples * PIXEL_M,
        "feature_pairs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
