"""P20-01 item 5: deterministic ortho mosaic on the shared DTM grid.

Policy is fixed before any score exists, and it is a *hard selection*, never a blend:

1. A pixel is eligible only where the camera is finite (not the isis2raw NULL sentinel).
2. Among eligible pixels the winner is the image with the highest priority, compared
   lexicographically: first the lower saturation share (a saturated pixel carries no photometry),
   then the higher sharpness. Mixing the two into one product would let a nearly-flat but
   unsaturated image beat a sharp saturated one for the wrong reason.
3. Nothing is averaged across images. Averaging would mix different illumination geometries
   (the probe scenes differ in solar elevation by >17 deg) and silently change radiometry, which
   P20-02 explicitly forbids.
4. Every output pixel records its source, so seamline length and per-image contribution are
   measurable, and the mosaic can be audited without redoing the projection.

Sharpness is the mean |laplacian| over the image's valid pixels; it ranks resampling quality, not
scientific value, and must not be read as a terrain-quality score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SATURATION_DN = 32000      # CDR stores 16-bit I/F; treat near-full-scale as saturated


def load_raw(path: Path, lines: int, samples: int):
    """Return (values with NULL as NaN, fill share)."""
    grid = np.fromfile(path, dtype="<f4").reshape(lines, samples).astype(np.float64)
    values, counts = np.unique(grid, return_counts=True)
    fill_share = float(counts.max() / grid.size)
    if fill_share >= 0.01:
        grid = np.where(grid == values[counts.argmax()], np.nan, grid)
    return grid, fill_share


def sharpness(image: np.ndarray) -> float:
    """Mean |laplacian| on valid pixels: ranks resampling quality, not science quality."""
    valid = np.isfinite(image)
    if not valid.any():
        return 0.0
    filled = np.where(valid, image, float(np.nanmedian(image)))
    lap = (4 * filled[1:-1, 1:-1] - filled[:-2, 1:-1] - filled[2:, 1:-1]
           - filled[1:-1, :-2] - filled[1:-1, 2:])
    interior = valid[1:-1, 1:-1]
    if not interior.any():
        return 0.0
    return float(np.abs(lap[interior]).mean())


def saturated_share(image: np.ndarray) -> float:
    valid = np.isfinite(image)
    if not valid.any():
        return 0.0
    return float((image[valid] >= SATURATION_DN).mean())


def weights(images) -> dict:
    """Per-image statistics and the lexicographic priority used by the selector."""
    stats = {}
    for name, image in images:
        stats[name] = {"valid_share": float(np.isfinite(image).mean()),
                       "saturation_share": saturated_share(image),
                       "sharpness": sharpness(image)}
    sharp = np.array([stats[n]["sharpness"] for n, _ in images], dtype=float)
    if sharp.max() > 0:
        sharp = sharp / sharp.max()
    for (name, _), rank in zip(images, sharp):
        stats[name]["sharpness_rank"] = float(rank)
        stats[name]["priority"] = [round(1.0 - stats[name]["saturation_share"], 6),
                                   round(float(rank), 6)]
    return stats


def priority_order(images, stats) -> list:
    """Image indices best-first: lower saturation, then sharper, then input order."""
    return sorted(range(len(images)),
                  key=lambda i: (-stats[images[i][0]]["priority"][0],
                                 -stats[images[i][0]]["priority"][1], i))


def mosaic(images, stats):
    """Hard per-pixel selection: each pixel goes to the first valid image in priority order."""
    shape = images[0][1].shape
    source = np.full(shape, -1, dtype=np.int16)
    for index in priority_order(images, stats):
        valid = np.isfinite(images[index][1])
        source[valid & (source == -1)] = index
    out = np.full(shape, np.nan)
    for index, (_, image) in enumerate(images):
        slot = source == index
        out[slot] = image[slot]
    return out, source


def seam_length(source: np.ndarray) -> int:
    """Pixels whose horizontal or vertical neighbour came from a different image."""
    total = 0
    vertical = source[1:, :] != source[:-1, :]
    total += int(vertical.sum())
    horizontal = source[:, 1:] != source[:, :-1]
    total += int(horizontal.sum())
    return total


def illumination_spread(illumination: dict) -> dict:
    """Worst-case spread of solar azimuth/elevation across the inputs, in degrees."""
    if not illumination:
        return {"images": 0}
    azimuths = [v["sun_azimuth_deg"] for v in illumination.values()]
    elevations = [v["sun_elevation_deg"] for v in illumination.values()]
    # circular spread for azimuth: 350 deg and 10 deg are 20 deg apart, not 340
    circular = [min(abs(a - b), 360 - abs(a - b)) for a in azimuths for b in azimuths]
    return {"images": len(azimuths),
            "azimuth_spread_deg": round(max(circular), 2),
            "elevation_spread_deg": round(max(elevations) - min(elevations), 2)}


def estimate_gains(images, stats) -> dict:
    """Multiplicative gain per image, fitted against the top-priority image on their overlap.

    Only meaningful when the illumination guard passed: the residual difference between two
    products of the same acquisition then comes from calibration/exposure, not from sun geometry,
    so a single gain per image is the least intrusive correction. The reference keeps gain 1.
    """
    order = priority_order(images, stats)
    ref_name, ref_image = images[order[0]]
    gains = {ref_name: 1.0}
    for index in order[1:]:
        name, image = images[index]
        overlap = np.isfinite(image) & np.isfinite(ref_image)
        if overlap.sum() < 32:
            gains[name] = 1.0
            continue
        ref_median = float(np.median(ref_image[overlap]))
        own_median = float(np.median(image[overlap]))
        gains[name] = ref_median / own_median if own_median > 0 else 1.0
    return {"reference": ref_name, "gains": gains}


def fit_photometry(images, stats, min_overlap: int = 32) -> dict:
    """Robust gain+offset per image against the top-priority reference.

    Each image is corrected to the reference over their overlap: gain from the ratio of medians,
    then offset from the median residual after that gain.

    A joint least-squares fit of all gains and offsets was tried first and rejected: with thin
    overlaps (85 px strips) the model is ill-conditioned, and the solution drifted to the clamps
    (all gains 0.5 with compensating offsets) while the seam step got worse, 0.0052 vs 0.0060.
    The sequential robust estimator cannot produce that degeneracy, and every parameter it writes
    is a median over real overlapping pixels.
    """
    order = priority_order(images, stats)
    ref_name, ref_image = images[order[0]]
    gains = {ref_name: 1.0}
    offsets = {ref_name: 0.0}
    pairs = 0
    for index in order[1:]:
        name, image = images[index]
        overlap = np.isfinite(image) & np.isfinite(ref_image)
        if overlap.sum() < min_overlap:
            gains[name], offsets[name] = 1.0, 0.0
            continue
        pairs += 1
        ref_values = ref_image[overlap]
        own_values = image[overlap]
        own_median = float(np.median(own_values))
        ref_median = float(np.median(ref_values))
        gain = ref_median / own_median if own_median > 0 else 1.0
        gain = float(np.clip(gain, 0.5, 2.0))
        offset = float(np.median(ref_values - gain * own_values))
        gains[name], offsets[name] = gain, offset
    return {"reference": ref_name, "gains": gains, "offsets": offsets, "pairs": pairs,
            "estimator": "median ratio + median residual"}


def apply_gains(images, gains: dict, offsets: dict | None = None):
    out = []
    for name, image in images:
        gain = gains.get(name, 1.0)
        offset = (offsets or {}).get(name, 0.0)
        if gain != 1.0 or offset != 0.0:
            out.append((name, image * gain + offset))
        else:
            out.append((name, image))
    return out


def seam_jump(values: np.ndarray, source: np.ndarray, control_stride: int = 7) -> dict:
    """Gradient-cancelled discontinuity at source boundaries, with an interior control.

    ``seam_step`` compares boundary pairs with every interior pair, which is biased: boundaries
    can sit on steeper terrain than the average interior pixel. This removes the local linear
    trend instead, at a horizontal boundary between x and x+1:

        jump = (v[x+1] - v[x]) - ((v[x] - v[x-1]) + (v[x+2] - v[x+1])) / 2

    and the same statistic is evaluated at interior positions as a control, so a value near the
    control means no visible step.
    """
    finite = np.isfinite(values)

    def horizontal(interior_only: bool):
        left_ok = finite[:, 1:-2] & finite[:, :-3]
        right_ok = finite[:, 2:-1] & finite[:, 3:]
        same_left = source[:, 1:-2] == source[:, :-3]
        same_right = source[:, 2:-1] == source[:, 3:]
        change = source[:, 1:-2] != source[:, 2:-1]
        base = left_ok & right_ok & same_left & same_right
        pick = base & (~change if interior_only else change)
        if not pick.any():
            return np.array([])
        own = values[:, 2:-1] - values[:, 1:-2]
        left = values[:, 1:-2] - values[:, :-3]
        right = values[:, 3:] - values[:, 2:-1]
        return np.abs(own - 0.5 * (left + right))[pick]

    def vertical(interior_only: bool):
        up_ok = finite[1:-2, :] & finite[:-3, :]
        down_ok = finite[2:-1, :] & finite[3:, :]
        same_up = source[1:-2, :] == source[:-3, :]
        same_down = source[2:-1, :] == source[3:, :]
        change = source[1:-2, :] != source[2:-1, :]
        base = up_ok & down_ok & same_up & same_down
        pick = base & (~change if interior_only else change)
        if not pick.any():
            return np.array([])
        own = values[2:-1, :] - values[1:-2, :]
        up = values[1:-2, :] - values[:-3, :]
        down = values[3:, :] - values[2:-1, :]
        return np.abs(own - 0.5 * (up + down))[pick]

    seam = np.concatenate([horizontal(False), vertical(False)])
    interior = np.concatenate([horizontal(True)[::control_stride],
                               vertical(True)[::control_stride]])
    out = {"seam_samples": int(seam.size), "control_samples": int(interior.size)}
    if seam.size:
        out["seam_jump_median"] = float(np.median(seam))
        out["seam_jump_p95"] = float(np.percentile(seam, 95))
    if interior.size:
        out["control_jump_median"] = float(np.median(interior))
        out["control_jump_p95"] = float(np.percentile(interior, 95))
    if seam.size and interior.size and np.median(interior) > 0:
        out["seam_over_control"] = round(float(np.median(seam) / np.median(interior)), 3)
    return out


def seam_step(values: np.ndarray, source: np.ndarray) -> dict:
    """Radiometric step across source boundaries, against the same statistic inside a source.

    A hard selection can still leave a visible band when two images differ in exposure, even at
    identical sun geometry, so the size of the jump must be measured rather than eyeballed.
    Horizontal and vertical neighbour pairs are handled separately because their shapes differ.
    """
    finite = np.isfinite(values)

    across_h = (finite[:, :-1] & finite[:, 1:]) & (source[:, :-1] != source[:, 1:])
    across_v = (finite[:-1, :] & finite[1:, :]) & (source[:-1, :] != source[1:, :])
    same_h = (finite[:, :-1] & finite[:, 1:]) & (source[:, :-1] == source[:, 1:])
    same_v = (finite[:-1, :] & finite[1:, :]) & (source[:-1, :] == source[1:, :])

    across = np.concatenate([np.abs(values[:, 1:] - values[:, :-1])[across_h],
                             np.abs(values[1:, :] - values[:-1, :])[across_v]])
    inside = np.concatenate([np.abs(values[:, 1:] - values[:, :-1])[same_h],
                             np.abs(values[1:, :] - values[:-1, :])[same_v]])

    out = {"seam_pairs": int(across.size), "interior_pairs": int(inside.size)}
    if across.size:
        out["seam_median"] = float(np.median(across))
        out["seam_p95"] = float(np.percentile(across, 95))
    if inside.size:
        out["interior_median"] = float(np.median(inside))
        out["interior_p95"] = float(np.percentile(inside, 95))
    if across.size and inside.size and np.median(inside) > 0:
        out["seam_over_interior_median"] = round(float(np.median(across) / np.median(inside)), 3)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--lines", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--out-raw", type=Path, required=True)
    parser.add_argument("--illumination", type=Path, default=None,
                        help="JSON {raw name: {sun_azimuth_deg, sun_elevation_deg}} from "
                             "scripts/solar_azimuth_p20.py")
    parser.add_argument("--max-azimuth-spread", type=float, default=10.0)
    parser.add_argument("--max-elevation-spread", type=float, default=5.0)
    parser.add_argument("--acquisition", action="append", default=None,
                        help="keep only these acquisition prefixes (e.g. M144483945); repeatable. "
                             "Mosaicking across acquisitions at low sun mixes different shadows.")
    parser.add_argument("--equalize", action="store_true",
                        help="fit one multiplicative gain per image against the top-priority "
                             "image; only valid once the illumination guard has passed")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    images = []
    for path in sorted(args.dir.glob("*_ortho.raw")):
        if "_" not in path.stem:
            continue
        if args.acquisition and not path.name.startswith(tuple(args.acquisition)):
            continue
        image, fill_share = load_raw(path, args.lines, args.samples)
        images.append((path.name, image))
        print(f"{path.name}: fill {fill_share:.4f} valid {np.isfinite(image).mean():.4f}")
    if not images:
        raise SystemExit(f"no *_ortho.raw under {args.dir}")

    illumination = json.loads(args.illumination.read_text(encoding="utf-8")) if args.illumination else {}
    spread = illumination_spread(illumination)
    if spread.get("images", 0) > 1:
        if (spread["azimuth_spread_deg"] > args.max_azimuth_spread
                or spread["elevation_spread_deg"] > args.max_elevation_spread):
            raise SystemExit(
                f"illumination spread too large to mosaic by hard selection "
                f"(azimuth {spread['azimuth_spread_deg']} deg, elevation "
                f"{spread['elevation_spread_deg']} deg); split the scene instead")

    stats = weights(images)
    gain_report = None
    if args.equalize:
        gain_report = fit_photometry(images, stats)
        images = apply_gains(images, gain_report["gains"], gain_report["offsets"])
    combined, source = mosaic(images, stats)

    counts = {images[i][0]: int((source == i).sum()) for i in range(len(images))}
    total = int(np.isfinite(combined).sum())
    report = {"dir": args.dir.name, "images": stats, "pixel_source": counts,
              "mosaic_valid_pixels": total,
              "mosaic_valid_share": round(total / combined.size, 6),
              "uncovered_pixels": int(combined.size - total),
              "seam_length_pixels": seam_length(source),
              "seam_step": seam_step(combined, source),
              "seam_jump": seam_jump(combined, source),
              "illumination": spread,
              "equalization": gain_report,
              "policy": "hard selection: lower saturation, then sharper; no blending"}
    for name in counts:
        report["images"][name]["contributed_share"] = (
            round(counts[name] / total, 6) if total else 0.0)

    args.out_raw.parent.mkdir(parents=True, exist_ok=True)
    np.nan_to_num(combined, nan=0.0).astype("<f4").tofile(args.out_raw)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
