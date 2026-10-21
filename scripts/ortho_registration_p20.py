"""P20-02: measured registration check for DTM-gridded NAC orthos.

Checks reported in metres on the known DTM grid:

1. ``overlap``   -- left camera vs right camera over their common footprint. Separate optics and
   SPICE states, so a near-zero window shift is evidence both were mapped onto one ground grid.
2. ``hillshade`` -- each ortho vs a hillshade rendered from the DTM. Used ONLY as a geometric
   QA check; never as a model input (P20-02 forbids terrain leakage into the model).

Both use normalised cross-correlation on integer shifts with parabolic subpixel refinement,
because phase-only correlation amplifies noise on smooth lunar texture (measured: sharpness ~11).

The hillshade needs the real solar azimuth, which the CDR labels do not carry and cam2map does
not propagate. ``campt`` gives incidence (this scene: 47.3 deg) but not azimuth, so the azimuth is
estimated per window from a coarse grid search; the reported azimuth spread is itself a QA signal
(a scene that only correlates at mutually inconsistent azimuths has no usable correspondence).

Windows are accepted only when >= ``--min-valid`` of their footprint is finite in BOTH inputs -
without that filter a NaN-filled patch silently correlates with nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

BAD_DTM = 1e30


def load_raw(path: Path, lines: int, samples: int) -> np.ndarray:
    """Read an isis2raw float export, turning the most frequent value (the NULL sentinel) into NaN."""
    raw = np.fromfile(path, dtype="<f4")
    assert raw.size == lines * samples, (path, raw.size, lines * samples)
    grid = raw.reshape(lines, samples).astype(np.float64)
    values, counts = np.unique(grid, return_counts=True)
    if counts.max() / grid.size >= 0.01:
        grid = np.where(grid == values[counts.argmax()], np.nan, grid)
    return grid


def corr_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r of two same-shaped patches."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def ncc_shift(a: np.ndarray, b: np.ndarray, radius: int):
    """Shift (dy, dx) that best maps b onto a within +-radius, plus peak NCC and subpixel refinement.

    Positive dy means b must move down to match a. Returns (dy, dx, peak).
    """
    a0 = a - a.mean()
    b0 = b - b.mean()
    energy = np.sqrt((a0 * a0).sum() * (b0 * b0).sum())
    if energy <= 0:
        return 0.0, 0.0, 0.0
    corr = fftconvolve(a0, b0[::-1, ::-1], mode="full")
    h, w = a.shape
    block = corr[h - 1 - radius:h + radius, w - 1 - radius:w + radius] / energy
    iy, ix = np.unravel_index(np.argmax(block), block.shape)
    peak = float(block[iy, ix])

    def refine(i, j, axis):
        """Parabolic subpixel offset along one axis, with (i, j) the integer peak."""
        n = block.shape[axis]
        here = i if axis == 0 else j
        if here == 0 or here == n - 1:
            return 0.0
        if axis == 0:
            lo, mid, hi = block[i - 1, j], block[i, j], block[i + 1, j]
        else:
            lo, mid, hi = block[i, j - 1], block[i, j], block[i, j + 1]
        denom = lo - 2 * mid + hi
        return float(np.clip(0.5 * (lo - hi) / denom, -0.5, 0.5)) if denom != 0 else 0.0

    return (iy - radius) + refine(iy, ix, 0), (ix - radius) + refine(iy, ix, 1), peak


def hillshade(dtm: np.ndarray, azimuth: float, altitude: float) -> np.ndarray:
    """Lambert shaded relief on the DTM's own grid (geometry QA only).

    cos(incidence) = n . s with n the surface normal and s the sun direction, both in
    (east, north, up). Written as a dot product rather than with an explicit aspect angle so the
    azimuth convention cannot drift (a ramp with positive dz/dx faces east and is dark when the
    sun azimuth is 90 deg).
    """
    gy, gx = np.gradient(np.nan_to_num(dtm, nan=0.0))
    norm = np.sqrt(1.0 + gx * gx + gy * gy)
    az, alt = np.deg2rad(azimuth), np.deg2rad(altitude)
    sun_east = np.sin(az) * np.cos(alt)
    sun_north = np.cos(az) * np.cos(alt)
    sun_up = np.sin(alt)
    shade = (-gx * sun_east - gy * sun_north + sun_up) / norm
    return np.clip(shade, 0.0, 1.0)


def smooth(array: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing with NaN filled by the local median value.

    Needed because a 2 m NAC DTM carries roughness far below the optical PSF; at sigma 0 the
    hillshade is dominated by that roughness and correlates with nothing (measured: r 0.39 at
    sigma 0 vs 0.65 at sigma 12 px on the same window).
    """
    if sigma <= 0:
        return array
    filled = np.nan_to_num(array, nan=float(np.nanmedian(array)))
    return gaussian_filter(filled, sigma)


def best_azimuth(image: np.ndarray, dtm: np.ndarray, mask: np.ndarray, altitude: float,
                 step: int = 15) -> tuple:
    """Azimuth whose hillshade correlates best with the image patch; returns (azimuth, corr)."""
    best = (-2.0, None)
    for azimuth in range(0, 360, step):
        shade = hillshade(dtm, azimuth, altitude)
        corr = corr_coefficient(image[mask], shade[mask])
        if corr > best[0]:
            best = (corr, azimuth)
    return best[1], best[0]


def windows(mask: np.ndarray, size: int, stride: int, min_valid: float, limit: int):
    """Yield (y, x) for windows with >= min_valid finite coverage, spread over the array."""
    ny, nx = mask.shape
    integral = np.zeros((ny + 1, nx + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.float64), axis=0), axis=1)

    def fraction(y, x):
        return (integral[y + size, x + size] - integral[y, x + size]
                - integral[y + size, x] + integral[y, x]) / (size * size)

    found = [(y, x) for y in range(0, ny - size + 1, stride)
             for x in range(0, nx - size + 1, stride) if fraction(y, x) >= min_valid]
    if len(found) > limit:
        picks = np.linspace(0, len(found) - 1, limit).round().astype(int)
        found = [found[i] for i in picks]
    return found


def slope_magnitude(dtm: np.ndarray) -> np.ndarray:
    """Local relief (|grad z|) used to rank windows by geometric information content."""
    gy, gx = np.gradient(np.nan_to_num(dtm, nan=0.0))
    return np.hypot(gx, gy)


def feature_windows(dtm: np.ndarray, mask: np.ndarray, size: int, count: int,
                    min_valid: float) -> list:
    """Top-``count`` non-overlapping valid windows ranked by mean local relief.

    Blind grid windows mostly measure albedo, which does not follow topography; ranking by relief
    puts the windows where optical texture and terrain actually correspond.
    """
    relief = slope_magnitude(dtm)
    ny, nx = dtm.shape
    integral_mask = np.zeros((ny + 1, nx + 1))
    integral_mask[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.float64), axis=0), axis=1)
    integral_relief = np.zeros((ny + 1, nx + 1))
    integral_relief[1:, 1:] = np.cumsum(np.cumsum(np.nan_to_num(relief, nan=0.0), axis=0), axis=1)

    def box(integral, y, x, s=size):
        return (integral[y + s, x + s] - integral[y, x + s] - integral[y + s, x] + integral[y, x])

    candidates = []
    for y in range(0, ny - size + 1, size):
        for x in range(0, nx - size + 1, size):
            if box(integral_mask, y, x) / (size * size) >= min_valid:
                candidates.append((box(integral_relief, y, x) / (size * size), y, x))
    candidates.sort(reverse=True)
    return [(y, x) for _, y, x in candidates[:count]]


def compare(a: np.ndarray, b: np.ndarray, mask: np.ndarray, size: int, stride: int,
            radius: int, pixel_m: float, min_valid: float, min_peak: float, limit: int,
            spots=None) -> dict:
    if spots is None:
        spots = windows(mask, size, stride, min_valid, limit)
    rows = []
    for y, x in spots:
        pa = a[y:y + size, x:x + size]
        pb = b[y:y + size, x:x + size]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            pa = np.nan_to_num(pa, nan=float(np.nanmedian(pa)))
            pb = np.nan_to_num(pb, nan=float(np.nanmedian(pb)))
        dy, dx, peak = ncc_shift(pa, pb, radius)
        rows.append({"y": y, "x": x, "dy": dy, "dx": dx, "peak": peak})

    out = {"windows_found": len(spots), "window_px": size, "search_radius_px": radius,
           "min_valid_fraction": min_valid, "min_peak": min_peak, "pixel_m": pixel_m}
    good = [r for r in rows if r["peak"] >= min_peak]
    out["windows_used"] = len(good)
    out["windows_rejected_weak_peak"] = len(rows) - len(good)
    if good:
        mags = np.hypot([r["dx"] for r in good], [r["dy"] for r in good])
        out.update({
            "dx_median_px": round(float(np.median([r["dx"] for r in good])), 3),
            "dy_median_px": round(float(np.median([r["dy"] for r in good])), 3),
            "shift_p50_m": round(float(np.percentile(mags, 50) * pixel_m), 2),
            "shift_p95_m": round(float(np.percentile(mags, 95) * pixel_m), 2),
            "shift_max_m": round(float(mags.max() * pixel_m), 2),
            "within_1px_fraction": round(float((mags <= 1.0).mean()), 4),
            "within_2px_fraction": round(float((mags <= 2.0).mean()), 4),
            "peak_median": round(float(np.median([r["peak"] for r in good])), 3),
            "worst_windows": sorted(good, key=lambda r: -np.hypot(r["dx"], r["dy"]))[:6],
        })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--dtm", type=Path, required=True)
    parser.add_argument("--lines", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--pixel-m", type=float, default=2.0)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--radius", type=int, default=64)
    parser.add_argument("--min-valid", type=float, default=0.999)
    parser.add_argument("--min-peak", type=float, default=0.3)
    parser.add_argument("--altitude", type=float, default=47.3,
                        help="solar elevation; campt gives incidence 47.3 deg for this scene")
    parser.add_argument("--azimuth", type=float, default=None,
                        help="solar azimuth from SPICE (scripts/solar_azimuth_p20.py); "
                             "when omitted it is searched per window as a diagnostic")
    parser.add_argument("--smooth-sigma", type=float, default=12.0,
                        help="gaussian sigma in pixels applied to both DTM and ortho before the "
                             "hillshade check (12 px = 24 m at 2 m/px)")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--select", choices=("grid", "relief"), default="relief",
                        help="how hillshade-check windows are chosen")
    parser.add_argument("--overlap-window", type=int, default=128,
                        help="smaller window for the L/R check: their common strip can be thin")
    parser.add_argument("--overlap-stride", type=int, default=64)
    parser.add_argument("--overlap-min-valid", type=float, default=0.95)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    dtm = tifffile.imread(args.dtm).astype(np.float64)
    dtm = np.where(np.isfinite(dtm) & (np.abs(dtm) < BAD_DTM), dtm, np.nan)
    assert dtm.shape == (args.lines, args.samples), (dtm.shape, args.lines, args.samples)

    images = {}
    for path in sorted(args.dir.glob("*_ortho.raw")):
        tag = "left" if "_L_ortho" in path.name else ("right" if "_R_ortho" in path.name else None)
        if tag:
            images.setdefault(tag, []).append((path.name, load_raw(path, args.lines, args.samples)))

    def mosaic(entries):
        out = np.full(entries[0][1].shape, np.nan)
        for _, image in entries:
            slot = ~np.isfinite(out) & np.isfinite(image)
            out[slot] = image[slot]
        return out

    report = {"scene": args.dir.name, "dtm": args.dtm.name, "pixel_m": args.pixel_m,
              "hillshade_altitude_deg": args.altitude,
              "hillshade_azimuth_source": ("fixed" if args.azimuth is not None
                                           else "per-window search"),
              "hillshade_smooth_sigma_px": args.smooth_sigma}

    if "left" in images and "right" in images:
        left, right = mosaic(images["left"]), mosaic(images["right"])
        mask = np.isfinite(left) & np.isfinite(right)
        report["overlap"] = {
            "images": {t: [n for n, _ in images[t]] for t in ("left", "right")},
            "overlap_pixels": int(mask.sum()),
            **compare(left, right, mask, args.overlap_window, args.overlap_stride, args.radius,
                      args.pixel_m, args.overlap_min_valid, args.min_peak, args.limit)}

    report["hillshade"] = {}
    dtm_smooth = smooth(dtm, args.smooth_sigma)
    for tag in sorted(images):
        image = smooth(mosaic(images[tag]), args.smooth_sigma)
        mask = np.isfinite(image) & np.isfinite(dtm)
        spots = (feature_windows(dtm, mask, args.window, args.limit, args.min_valid)
                 if args.select == "relief"
                 else windows(mask, args.window, args.stride, args.min_valid, args.limit))
        azimuths = []
        best = None
        for y, x in spots:
            crop_i = image[y:y + args.window, x:x + args.window]
            crop_d = dtm_smooth[y:y + args.window, x:x + args.window]
            finite = np.isfinite(crop_i) & np.isfinite(crop_d)
            if args.azimuth is None:
                az, corr = best_azimuth(crop_i, crop_d, finite, args.altitude)
            else:
                az = int(args.azimuth)
                corr = corr_coefficient(crop_i[finite],
                                        hillshade(crop_d, az, args.altitude)[finite])
            azimuths.append({"y": y, "x": x, "azimuth": az, "corr": round(corr, 3)})
            if best is None or corr > best[1]:
                best = (az, corr)
        shade = hillshade(dtm_smooth, best[0] if best else 315.0, args.altitude)
        entry = {"images": [n for n, _ in images[tag]], "window_select": args.select,
                 "azimuth_best": best[0] if best else None,
                 "corr_at_best_azimuth": round(best[1], 3) if best else None,
                 "azimuth_per_window": azimuths,
                 **compare(image, shade, mask, args.window, args.stride, args.radius,
                           args.pixel_m, args.min_valid, args.min_peak, args.limit,
                           spots=spots)}
        if azimuths:
            entry["azimuth_spread_deg"] = int(max(a["azimuth"] for a in azimuths)
                                              - min(a["azimuth"] for a in azimuths))
        report["hillshade"][tag] = entry

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
