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
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    images = []
    for path in sorted(args.dir.glob("*_ortho.raw")):
        if "_" not in path.stem:
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
    combined, source = mosaic(images, stats)

    counts = {images[i][0]: int((source == i).sum()) for i in range(len(images))}
    total = int(np.isfinite(combined).sum())
    report = {"dir": args.dir.name, "images": stats, "pixel_source": counts,
              "mosaic_valid_pixels": total,
              "mosaic_valid_share": round(total / combined.size, 6),
              "uncovered_pixels": int(combined.size - total),
              "seam_length_pixels": seam_length(source),
              "illumination": spread,
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
