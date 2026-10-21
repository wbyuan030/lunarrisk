"""P20-02: crater-centroid control points for ortho registration.

NCC and mutual information fail between a NAC ortho and a DTM hillshade because the optical image
is albedo-dominated while the hillshade carries geometry only. Small craters are the feature both
products must agree on, so: detect circular blobs (scale-normalised LoG, both polarities) in each
image, match by mutual nearest neighbour, then search the dominant displacement bin. A tight
dominant cluster at (0, 0) means both products place the same landforms at the same pixel.

``--self-check`` runs the same detector on two products known to share a grid (left vs right camera)
so the method itself is validated before it is used as evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter
from scipy.spatial import cKDTree

BAD_DTM = 1e30


def load_raw(path: Path, lines: int, samples: int) -> np.ndarray:
    grid = np.fromfile(path, dtype="<f4").reshape(lines, samples).astype(np.float64)
    values, counts = np.unique(grid, return_counts=True)
    if counts.max() / grid.size >= 0.01:
        grid = np.where(grid == values[counts.argmax()], np.nan, grid)
    return grid


def fill(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=float(np.nanmedian(array)))


def blob_response(image: np.ndarray, sigma: float) -> np.ndarray:
    """Scale-normalised Laplacian of Gaussian; positive at dark-centre blobs, negative at bright."""
    return -(sigma ** 2) * gaussian_laplace(fill(image), sigma)


def detect(image: np.ndarray, sigmas, threshold: float, min_sep: int, max_blobs: int = 4000):
    """Return [(y, x, sigma, strength)] for the strongest polarity at each location."""
    response = None
    scale = None
    for sigma in sigmas:
        resp = blob_response(image, sigma)
        if response is None:
            response, scale = resp, np.full(resp.shape, sigma)
        else:
            better = np.abs(resp) > np.abs(response)
            response = np.where(better, resp, response)
            scale = np.where(better, sigma, scale)
    magnitude = np.abs(response)
    if not np.isfinite(magnitude).any() or magnitude.max() <= 0:
        return []                                  # flat input: keepless by construction
    peaks = magnitude == maximum_filter(magnitude, size=min_sep)
    peaks &= magnitude > magnitude.max() * 1e-6    # drop degenerate plateaux (constant patches)
    if not peaks.any():
        return []
    cut = np.percentile(magnitude[peaks], 100.0 - threshold * 100.0)
    ys, xs = np.nonzero(peaks & (magnitude >= cut))
    order = np.argsort(-magnitude[ys, xs])[:max_blobs]
    return [(int(ys[i]), int(xs[i]), float(scale[ys[i], xs[i]]), float(magnitude[ys[i], xs[i]]))
            for i in order]


def match(points_a, points_b, max_dist: float):
    """Mutual nearest neighbour matches within max_dist (KD-tree: a 1e5x1e5 loop never returns)."""
    if not points_a or not points_b:
        return []
    a = np.array([(p[0], p[1], p[2]) for p in points_a])
    b = np.array([(p[0], p[1], p[2]) for p in points_b])
    tree_b = cKDTree(b[:, :2])
    tree_a = cKDTree(a[:, :2])

    def nearest(src, tree):
        distance, index = tree.query(src[:, :2], k=1, distance_upper_bound=max_dist)
        return [int(i) if np.isfinite(d) else -1 for d, i in zip(distance, index)]

    fwd, back = nearest(a, tree_b), nearest(b, tree_a)
    pairs = []
    for i, j in enumerate(fwd):
        if j >= 0 and back[j] == i:
            dy = float(b[j, 0] - a[i, 0])
            dx = float(b[j, 1] - a[i, 1])
            pairs.append({"y": float(a[i, 0]), "x": float(a[i, 1]),
                          "dy": dy, "dx": dx, "size_ratio": float(b[j, 2] / a[i, 2])})
    return pairs


def dominant_offset(pairs, bin_px: float = 1.0):
    """Mode of the displacement distribution plus its inlier set."""
    if not pairs:
        return {"matches": 0}
    dys = np.array([p["dy"] for p in pairs])
    dxs = np.array([p["dx"] for p in pairs])
    ky = np.round(dys / bin_px).astype(int)
    kx = np.round(dxs / bin_px).astype(int)
    keys, counts = np.unique(np.stack([ky, kx], axis=1), axis=0, return_counts=True)
    best = int(np.argmax(counts))
    cy, cx = keys[best][0] * bin_px, keys[best][1] * bin_px
    inliers = [p for p in pairs if abs(p["dy"] - cy) <= 1.5 and abs(p["dx"] - cx) <= 1.5]
    magnitudes = np.hypot([p["dy"] for p in inliers], [p["dx"] for p in inliers])
    return {"matches": len(pairs),
            "dominant_dy_px": float(cy), "dominant_dx_px": float(cx),
            "dominant_inliers": len(inliers),
            "dominant_share": round(len(inliers) / len(pairs), 4),
            "dy_median_px": float(np.median(dys)), "dx_median_px": float(np.median(dxs)),
            "inlier_p50_px": float(np.percentile(magnitudes, 50)),
            "inlier_p95_px": float(np.percentile(magnitudes, 95)),
            "inlier_max_px": float(magnitudes.max())}


def hillshade(dtm: np.ndarray, azimuth: float, altitude: float) -> np.ndarray:
    gy, gx = np.gradient(fill(dtm))
    norm = np.sqrt(1.0 + gx * gx + gy * gy)
    az, alt = np.deg2rad(azimuth), np.deg2rad(altitude)
    shade = (-gx * np.sin(az) * np.cos(alt) - gy * np.cos(az) * np.cos(alt)
             + np.sin(alt)) / norm
    return np.clip(shade, 0.0, 1.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-raw", type=Path, required=True)
    parser.add_argument("--right-raw", type=Path, default=None,
                        help="second product: another camera, or omit and pass --dtm")
    parser.add_argument("--dtm", type=Path, default=None)
    parser.add_argument("--lines", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--pixel-m", type=float, default=2.0)
    parser.add_argument("--dtm-mode", choices=("dem", "hillshade"), default="dem")
    parser.add_argument("--azimuth", type=float, default=100.0)
    parser.add_argument("--altitude", type=float, default=42.6)
    parser.add_argument("--sigmas", default="2,3,5,8,12")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="fraction of strongest peaks kept")
    parser.add_argument("--min-sep", type=int, default=7)
    parser.add_argument("--max-dist", type=float, default=8.0)
    parser.add_argument("--max-blobs", type=int, default=4000)
    parser.add_argument("--crop", default=None, help="y0,x0,size to restrict the analysis")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    sigmas = [float(s) for s in args.sigmas.split(",")]
    left = load_raw(args.left_raw, args.lines, args.samples)
    if args.right_raw:
        right = load_raw(args.right_raw, args.lines, args.samples)
        reference = "ortho"
    elif args.dtm:
        import tifffile
        dtm = tifffile.imread(args.dtm).astype(np.float64)
        dtm = np.where(np.isfinite(dtm) & (np.abs(dtm) < BAD_DTM), dtm, np.nan)
        if args.dtm_mode == "hillshade":
            right = hillshade(dtm, args.azimuth, args.altitude)
            reference = "dtm_hillshade"
        else:
            # A crater is a bowl: LoG runs directly on elevation, so detection does not depend on
            # the illumination model at all (shading splits rims into a bright and a dark half).
            right = dtm
            reference = "dtm_elevation"
    else:
        raise SystemExit("need either --right-raw or --dtm")

    if args.crop:
        y0, x0, size = (int(v) for v in args.crop.split(","))
        left = left[y0:y0 + size, x0:x0 + size]
        right = right[y0:y0 + size, x0:x0 + size]
    mask = np.isfinite(left) & np.isfinite(right)
    a = detect(np.where(mask, left, np.nan), sigmas, args.threshold, args.min_sep, args.max_blobs)
    b = detect(np.where(mask, right, np.nan), sigmas, args.threshold, args.min_sep, args.max_blobs)
    pairs = match(a, b, args.max_dist)
    stats = dominant_offset(pairs)

    report = {"left": args.left_raw.name,
              "reference": reference if args.right_raw is None else args.right_raw.name,
              "pixel_m": args.pixel_m, "overlap_pixels": int(mask.sum()),
              "blobs_left": len(a), "blobs_reference": len(b), "max_blobs": args.max_blobs,
              "crop": args.crop,
              "max_dist_px": args.max_dist, "sigmas": sigmas,
              "threshold": args.threshold, "min_sep": args.min_sep,
              **stats}
    if stats.get("matches"):
        report["dominant_offset_m"] = round(
            float(np.hypot(stats["dominant_dy_px"], stats["dominant_dx_px"]) * args.pixel_m), 2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
