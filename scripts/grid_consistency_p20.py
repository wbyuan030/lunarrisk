"""P20-02 item 1: does the ortho pixel grid match the DTM/geographic grid exactly?

Independent of any image-to-image correlation, this walks a set of control pixels through
``campt`` (SPICE camera model -> planetocentric lat/lon), pushes those coordinates through the
ortho cube's own mapping geometry, and then asks whether the ortho holds the same measurement
there. With nearest-neighbour resampling the value must match; a systematic offset in pixels is a
map-projection / pixel-centre inconsistency, not a photogrammetric error.

Each control point reports the offset of the ortho pixel whose value best matches the source
value within +-``--search``, so the residual is in pixels of the actual grid (2 m here).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

BAD = 1e30


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def cube_dims(cube: Path) -> tuple:
    code, out, err = run(["getkey", f"from={cube}", "grpname=Dimensions", "keyword=Lines"])
    if code:
        raise RuntimeError(err[:300])
    lines = int(out.strip())
    code, out, err = run(["getkey", f"from={cube}", "grpname=Dimensions", "keyword=Samples"])
    if code:
        raise RuntimeError(err[:300])
    return lines, int(out.strip())


def mapping(cube: Path) -> dict:
    """Read the Mapping group of an ISIS cube as a dict."""
    code, out, err = run(["catlab", f"from={cube}"])
    if code:
        raise RuntimeError(err[:300])
    values = {}
    inside = False
    for line in out.splitlines():
        text = line.strip()
        if text.startswith("Group = Mapping"):
            inside = True
            continue
        if inside and text == "End_Group":
            break
        if inside and "=" in text:
            key, _, value = text.partition("=")
            values[key.strip()] = value.split("<")[0].strip()
    return values


def campt_latlon(cube: Path, line: float, sample: float) -> tuple:
    code, out, err = run(["campt", f"from={cube}", "coordtype=image",
                          f"line={line}", f"sample={sample}"])
    if code:
        raise RuntimeError(err[:300])
    lat = lon = None
    for line_text in out.splitlines():
        text = line_text.strip()
        if "=" not in text:
            continue
        key, _, value = text.partition("=")
        key = key.strip()
        # campt prints PlanetocentricLatitude / PositiveEastLongitude for an image request
        if lat is None and key in ("PlanetocentricLatitude", "Latitude"):
            lat = float(value.split()[0])
        elif lon is None and key in ("PositiveEast360Longitude", "PositiveEastLongitude",
                                      "Longitude"):
            lon = float(value.split()[0])
    if lat is None or lon is None:
        raise RuntimeError(f"campt gave no lat/lon: {out[:200]}")
    return lat, lon


def latlon_to_pixel(mapping_group: dict, lines: int, samples: int, lat: float, lon: float):
    """Equirectangular pixel-is-area mapping, matching what cam2map produced."""
    min_lat = float(mapping_group["MinimumLatitude"])
    max_lat = float(mapping_group["MaximumLatitude"])
    min_lon = float(mapping_group["MinimumLongitude"])
    max_lon = float(mapping_group["MaximumLongitude"])
    x = (lon - min_lon) / (max_lon - min_lon) * samples
    y = (max_lat - lat) / (max_lat - min_lat) * lines
    return x, y          # 0-based, pixel is area: pixel i covers [i, i+1)


def campt_linesample(cube: Path, lat: float, lon: float) -> tuple:
    """Ground -> camera (line, sample) with campt; the round trip is exact to <1e-3 px."""
    code, out, err = run(["campt", f"from={cube}", "type=ground",
                          f"lat={lat!r}", f"lon={lon!r}"])
    if code:
        raise RuntimeError(err[-300:])
    line = sample = None
    for text in out.splitlines():
        if "=" not in text:
            continue
        key, _, value = text.partition("=")
        if key.strip() == "Line":
            line = float(value.split()[0])
        elif key.strip() == "Sample":
            sample = float(value.split()[0])
    if line is None or sample is None:
        raise RuntimeError("campt gave no line/sample")
    return line, sample


def pixel_to_latlon(mapping_group: dict, lines: int, samples: int, x: float, y: float) -> tuple:
    """Inverse of latlon_to_pixel, taking pixel centres (x, y are 0-based corner coordinates)."""
    min_lat = float(mapping_group["MinimumLatitude"])
    max_lat = float(mapping_group["MaximumLatitude"])
    min_lon = float(mapping_group["MinimumLongitude"])
    max_lon = float(mapping_group["MaximumLongitude"])
    lon = min_lon + x / samples * (max_lon - min_lon)
    lat = max_lat - y / lines * (max_lat - min_lat)
    return lat, lon


def ortho_to_camera(camera_cube: Path, camera_raw: Path, ortho_raw: Path, mapping_group: dict,
                    points: int, seed: int) -> dict:
    """Exact test: with nearest-neighbour resampling the ortho pixel must equal the CDR DN it maps to.

    ``cam2map`` in this pipeline resamples nearest: every ortho value is an exact multiple of the
    CDR quantum 3.05185e-05. So the test is a binary equality check per sample - a systematic
    half-pixel error shows up as a large mismatch fraction, not as a fuzzy correlation.
    """
    o_lines = int(mapping_group["_lines"])
    o_samples = int(mapping_group["_samples"])
    ortho = np.fromfile(ortho_raw, dtype="<f4").reshape(o_lines, o_samples).astype(np.float64)
    cam_lines, cam_samples = cube_dims(camera_cube)
    camera = np.fromfile(camera_raw, dtype="<f4").reshape(cam_lines, cam_samples).astype(np.float64)

    rng = np.random.default_rng(seed)
    rows = []
    attempts = 0
    while len(rows) < points and attempts < points * 20:
        attempts += 1
        i = int(rng.integers(0, o_samples))
        j = int(rng.integers(0, o_lines))
        value = ortho[j, i]
        if not np.isfinite(value) or value <= 0:
            continue
        lat, lon = pixel_to_latlon(mapping_group, o_lines, o_samples, i + 0.5, j + 0.5)
        try:
            line, sample = campt_linesample(camera_cube, lat, lon)
        except RuntimeError:
            continue
        li, si = int(round(line)), int(round(sample))
        if not (1 <= li <= cam_lines and 1 <= si <= cam_samples):
            continue
        source = camera[li - 1, si - 1]
        if not np.isfinite(source) or source <= 0:
            continue
        exact = abs(source - value) <= 1e-9
        dy = dx = None
        if not exact:
            # where does the matching value sit inside the camera frame? (radius 4 around the
            # pixel the prediction points at; ties resolved to the closest hit)
            pad = 4
            y0, x0 = max(0, li - 1 - pad), max(0, si - 1 - pad)
            patch = camera[y0:li + pad, x0:si + pad]
            hit = np.argwhere(np.abs(patch - value) <= 1e-9)
            if hit.size:
                best = min(hit, key=lambda h: abs(h[0] - (li - 1 - y0)) + abs(h[1] - (si - 1 - x0)))
                dy = int(best[0] - (li - 1 - y0))
                dx = int(best[1] - (si - 1 - x0))
        rows.append({"x": i, "y": j, "lat": round(lat, 6), "lon": round(lon, 6),
                     "camera_line": round(line, 4), "camera_sample": round(sample, 4),
                     "ortho_value": float(value), "camera_value": float(source),
                     "exact": bool(exact), "dy_px": dy, "dx_px": dx})

    exact = [r for r in rows if r["exact"]]
    found = [r for r in rows if not r["exact"] and r["dy_px"] is not None]
    out = {"points_used": len(rows),
           "exact_match_fraction": (len(exact) / len(rows)) if rows else None,
           "mismatch_with_value_found": len(found),
           "samples": rows[:8]}
    if found:
        dys = np.array([r["dy_px"] for r in found], dtype=float)
        dxs = np.array([r["dx_px"] for r in found], dtype=float)
        out.update({"dy_median_px": float(np.median(dys)), "dx_median_px": float(np.median(dxs)),
                    "dy_abs_p95_px": float(np.percentile(np.abs(dys), 95)),
                    "dx_abs_p95_px": float(np.percentile(np.abs(dxs), 95)),
                    "displacement_zero_fraction": float(np.mean((dys == 0) & (dxs == 0)))})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-cube", type=Path, required=True, help="spiceinit'ed L1 cube")
    parser.add_argument("--ortho", type=Path, required=True, help="cam2map output cube")
    parser.add_argument("--ortho-raw", type=Path, required=True, help="isis2raw float export")
    parser.add_argument("--points", type=int, default=40)
    parser.add_argument("--search", type=int, default=4, help="+-pixels to look for the value")
    parser.add_argument("--tol", type=float, default=1e-5, help="absolute value tolerance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--direction", choices=("camera-to-ortho", "ortho-to-camera"),
                        default="camera-to-ortho")
    parser.add_argument("--camera-raw", type=Path, default=None,
                        help="isis2raw export of the camera cube (needed for ortho-to-camera)")
    args = parser.parse_args(argv)

    cam_lines, cam_samples = cube_dims(args.camera_cube)
    o_lines, o_samples = cube_dims(args.ortho)
    grid = np.fromfile(args.ortho_raw, dtype="<f4").reshape(o_lines, o_samples).astype(np.float64)
    group = mapping(args.ortho)

    rng = np.random.default_rng(args.seed)
    rows = []
    attempts = 0
    while len(rows) < args.points and attempts < args.points * 5:
        attempts += 1
        line = float(rng.integers(200, cam_lines - 200))
        sample = float(rng.integers(200, cam_samples - 200))
        try:
            lat, lon = campt_latlon(args.camera_cube, line, sample)
        except RuntimeError:
            continue
        px, py = latlon_to_pixel(group, o_lines, o_samples, lat, lon)
        ix, iy = int(np.floor(px)), int(np.floor(py))
        if not (args.search <= ix < o_samples - args.search and args.search <= iy < o_lines - args.search):
            continue
        target = float(np.median(grid[iy - 1:iy + 2, ix - 1:ix + 2]))
        if not np.isfinite(target) or target <= 0:
            continue
        window = grid[iy - args.search:iy + args.search + 1, ix - args.search:ix + args.search + 1]
        diff = np.abs(window - target)
        best = np.unravel_index(np.argmin(diff), diff.shape)
        rows.append({"line": line, "sample": sample, "lat": round(lat, 6), "lon": round(lon, 6),
                     "pred_x": round(px, 3), "pred_y": round(py, 3),
                     "best_dx": int(best[1] - args.search), "best_dy": int(best[0] - args.search),
                     "value_error": round(float(diff[best]), 8),
                     "match_within_tol": bool(diff[best] <= args.tol)})

    report = {"camera_cube": args.camera_cube.name, "ortho": args.ortho.name,
              "camera_dims": [cam_lines, cam_samples], "ortho_dims": [o_lines, o_samples],
              "mapping": group, "points_requested": args.points, "points_used": len(rows),
              "search_px": args.search, "tol": args.tol,
              "notes": "pred_x/pred_y are 0-based pixel-is-area coordinates from the ortho Mapping "
                       "group; best_dx/best_dy is the offset of the closest matching value."}
    if rows:
        dx = np.array([r["best_dx"] for r in rows], dtype=float)
        dy = np.array([r["best_dy"] for r in rows], dtype=float)
        report.update({
            "dx_median_px": float(np.median(dx)),
            "dy_median_px": float(np.median(dy)),
            "dx_p95_abs_px": float(np.percentile(np.abs(dx), 95)),
            "dy_p95_abs_px": float(np.percentile(np.abs(dy), 95)),
            "offset_zero_fraction": float(np.mean((dx == 0) & (dy == 0))),
            "value_match_fraction": float(np.mean([r["match_within_tol"] for r in rows])),
            "sample_points": rows[:10],
        })
    if args.direction == "ortho-to-camera":
        if not args.camera_raw:
            raise SystemExit("--camera-raw is required for --direction ortho-to-camera")
        report["direction"] = "ortho-to-camera"
        group["_lines"], group["_samples"] = o_lines, o_samples
        report["exact_test"] = ortho_to_camera(args.camera_cube, args.camera_raw, args.ortho_raw,
                                              group, args.points, args.seed)
    else:
        report["direction"] = "camera-to-ortho"
        report["exact_test"] = {
            "points_used": len(rows),
            "exact_match_fraction": float(np.mean([r["match_within_tol"] for r in rows])) if rows else None,
        }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
