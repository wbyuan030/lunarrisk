#!/usr/bin/env python3
"""P19-02: file-level probe of native LROC NAC DTM products (Gate 2).

Reads the real PDS DTM (float32 GeoTIFF + PDS3 label) of single scenes that can never
enter the frozen V2 set, verifies the label against the file itself (byte layout, size,
GeoTIFF GeoKeys, NoData), and computes the raw physical label statistics at the native
post spacing plus at coarser block aggregations.

It never downloads anything, never claims a 2-7 m product is a 3 m ground truth, and never
reports model scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

MISSING_CONSTANT = -1.0e32  # DLR "missing" constant used alongside NoDATA
GEOKEY_NAMES = {
    1024: "GTModelTypeGeoKey", 1025: "GTRasterTypeGeoKey", 1026: "GTCitationGeoKey",
    2048: "GeographicTypeGeoKey", 2049: "GeogCitationGeoKey",
    2050: "GeogGeodeticDatumGeoKey", 2051: "GeogPrimeMeridianGeoKey",
    2052: "GeogLinearUnitsGeoKey", 2054: "GeogAngularUnitsGeoKey",
    2056: "GeogEllipsoidGeoKey", 2057: "GeogSemiMajorAxisGeoKey",
    2058: "GeogSemiMinorAxisGeoKey", 3072: "ProjectedCSTypeGeoKey",
    3073: "PCSCitationGeoKey", 3076: "ProjLinearUnitsGeoKey",
    3078: "ProjStdParallel1GeoKey", 3080: "ProjNatOriginLongGeoKey",
    3081: "ProjNatOriginLatGeoKey", 3082: "ProjFalseEastingGeoKey",
    3083: "ProjFalseNorthingGeoKey", 3088: "ProjCenterLongGeoKey",
    3089: "ProjCenterLatGeoKey", 3092: "ProjScaleAtNatOriginGeoKey",
    4096: "ProjCoordTransGeoKey",
}

NUMERIC_KEYS = {
    "MAP_SCALE", "A_AXIS_RADIUS", "B_AXIS_RADIUS", "C_AXIS_RADIUS", "LINES", "LINE_SAMPLES",
    "SAMPLE_BITS", "BANDS", "CENTER_LATITUDE", "CENTER_LONGITUDE", "MAXIMUM_LATITUDE",
    "MINIMUM_LATITUDE", "EASTERNMOST_LONGITUDE", "WESTERNMOST_LONGITUDE", "MAP_RESOLUTION",
    "RECORD_BYTES", "FILE_RECORDS", "LINE_PROJECTION_OFFSET", "SAMPLE_PROJECTION_OFFSET",
}


def join_continuations(text: str) -> list[str]:
    """Join PDS3 multi-line quoted values (the DTM NoDATA note spans three physical lines)."""
    lines: list[str] = []
    pending: str | None = None
    for raw in text.splitlines():
        line = raw.split("/*")[0] if "/*" in raw else raw
        if pending is not None:
            pending = f"{pending} {line.strip()}"
            if line.count('"') % 2 == 1:
                lines.append(pending)
                pending = None
            continue
        if line.count('"') % 2 == 1:
            pending = line.rstrip()
            continue
        lines.append(line)
    if pending is not None:
        lines.append(pending)
    return lines


def parse_pds_label(text: str) -> dict:
    """Extract the keywords the pipeline depends on from a PDS3 label."""
    out: dict = {"objects": {}}
    object_name = None
    for raw_line in join_continuations(text):
        if "/*" in raw_line:  # strip PDS comments
            raw_line = raw_line.split("/*")[0]
        line = raw_line.rstrip()
        if not line.strip():
            continue
        match = re.match(r"\s*(OBJECT|END_OBJECT)\s*=\s*([A-Z_]+)?", line)
        if match:
            if match.group(1) == "OBJECT":
                object_name = (match.group(2) or "").strip()
                out["objects"].setdefault(object_name, {})
            else:
                object_name = None
            continue
        match = re.match(r"\s*(\^?[A-Za-z_0-9]+)\s*=\s*(.+?)\s*$", line)
        if not match:
            continue
        key, value = match.group(1).upper(), match.group(2)
        target = out["objects"][object_name] if object_name else out
        if key == "^IMAGE":
            parts = [p.strip().strip('"') for p in value.strip("()").split(",")]
            target["IMAGE_FILE"] = parts[0]
            if len(parts) > 1:
                number = re.match(r"(\d+)", parts[1])
                target["IMAGE_OFFSET_BYTES"] = int(number.group(1)) if number else None
            continue
        value = value.strip().strip('"')
        value = re.sub(r"\s*<[^>]*>", "", value).strip()
        if key in NUMERIC_KEYS:
            try:
                target[key] = float(value) if "." in value or "e" in value.lower() else int(value)
                continue
            except ValueError:
                pass
        target[key] = value
    image = out["objects"].get("IMAGE", {})
    # Real LROC DTM labels carry no NoDATA keyword: the sentinel is only described in the
    # IMAGE DESCRIPTION prose, so it has to be recovered from there (and cross-checked
    # against the GeoTIFF GDAL_NODATA tag).
    description = str(image.get("DESCRIPTION", ""))
    out["NODATA"] = _to_float(image.get("NODATA"))
    if out["NODATA"] is None:
        match = re.search(r"NoDATA value is\s*([-+0-9.eE]+)", description)
        out["NODATA"] = _to_float(match.group(1)) if match else None
        if out["NODATA"] is not None:
            out["NODATA_SOURCE"] = "IMAGE.DESCRIPTION"
    else:
        out["NODATA_SOURCE"] = "IMAGE.NoDATA"
    out["MISSING"] = _to_float(image.get("MISSING_CONSTANT"))
    if out["MISSING"] is None:
        match = re.search(r"missing constant of\s*([-+0-9.eE]+)", description)
        out["MISSING"] = _to_float(match.group(1)) if match else None
    out["SAMPLE_TYPE"] = image.get("SAMPLE_TYPE")
    out["LINES"] = image.get("LINES", out.get("LINES"))
    out["LINE_SAMPLES"] = image.get("LINE_SAMPLES", out.get("LINE_SAMPLES"))
    out["SAMPLE_BITS"] = image.get("SAMPLE_BITS", out.get("SAMPLE_BITS"))
    projection = out["objects"].get("IMAGE_MAP_PROJECTION", {})
    out["PROJECTION"] = projection.get("MAP_PROJECTION_TYPE")
    out["MAP_SCALE_M"] = projection.get("MAP_SCALE")
    out["SPHERE_RADIUS_KM"] = projection.get("A_AXIS_RADIUS")
    out["CENTER_LAT"] = projection.get("CENTER_LATITUDE")
    out["MIN_LAT"] = projection.get("MINIMUM_LATITUDE")
    out["MAX_LAT"] = projection.get("MAXIMUM_LATITUDE")
    out["WEST_LON"] = projection.get("WESTERNMOST_LONGITUDE")
    out["EAST_LON"] = projection.get("EASTERNMOST_LONGITUDE")
    return out


def ascii_tag(path: Path, page, tag_code: int) -> str | None:
    """Decode an ASCII TIFF tag, falling back to a raw read when tifffile drops it.

    tifffile parses GDAL_NODATA against the sample dtype, so a float32 raster carrying
    '-3.40282266e+38' raises and the parsed value is lost; the bytes are still in the file.
    """
    tag = page.tags.get(tag_code)
    if tag is None:
        return None
    for candidate in (getattr(tag, "value", None),):
        if isinstance(candidate, bytes):
            return candidate.decode("ascii", "replace").strip("\x00 ")
        if isinstance(candidate, str):
            return candidate.strip("\x00 ")
    offset, count = getattr(tag, "valueoffset", None), getattr(tag, "count", None)
    if offset and count:
        with open(path, "rb") as handle:
            handle.seek(int(offset))
            return handle.read(int(count)).decode("ascii", "replace").strip("\x00 ")
    return None


def image_pointer(label: dict) -> tuple[str | None, int | None]:
    """The ^IMAGE pointer is declared at the label root on LROC products."""
    for scope in (label, label.get("objects", {}).get("IMAGE", {})):
        if scope.get("IMAGE_FILE"):
            return scope["IMAGE_FILE"], scope.get("IMAGE_OFFSET_BYTES")
    return None, None


def _to_float(value):
    if value is None:
        return None
    match = re.match(r"[-+]?[0-9.]+(?:e[-+]?\d+)?", str(value))
    return float(match.group(0)) if match else None


def valid_mask(elevation: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """Float DTMs carry a NoDATA sentinel and a smaller 'missing' constant; both are invalid.

    Both sentinels are far outside any real lunar elevation (-1e32 / -3.4e38), so a generous
    magnitude cut rejects them without needing the exact literal from each label.
    """
    mask = np.isfinite(elevation)
    if nodata is not None:
        mask &= np.abs(elevation - nodata) > 1.0
    mask &= (np.abs(elevation) < 1.0e30)
    return mask


def interior_mask(mask: np.ndarray) -> np.ndarray:
    """Valid pixels whose full 3x3 neighbourhood is valid (gradients need neighbours)."""
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    interior = np.ones_like(mask)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            interior &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return interior


def slope_degrees(elevation: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Horn-free central-difference slope magnitude in degrees on a filled array."""
    gy, gx = np.gradient(elevation, pixel_size_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def label_statistics(elevation: np.ndarray, pixel_size_m: float, nodata: float | None = None,
                     thresholds=(10.0, 15.0), unsafe_threshold_deg: float = 10.0,
                     unsafe_fraction: float = 0.05) -> dict:
    """Raw area-fraction statistics over valid pixels, matching the frozen proxy rule."""
    mask = valid_mask(elevation, nodata)
    interior = interior_mask(mask)
    filled = np.where(mask, elevation, float(np.median(elevation[mask])) if mask.any() else 0.0)
    slope = slope_degrees(filled, pixel_size_m)
    evaluated = slope[interior]
    result = {
        "valid_fraction": float(mask.mean()),
        "evaluated_fraction_of_total": float(interior.mean()),
        "evaluated_pixels": int(interior.sum()),
        "elevation_m": {
            "min": float(elevation[mask].min()), "max": float(elevation[mask].max()),
            "median": float(np.median(elevation[mask])),
        } if mask.any() else None,
        "slope_deg": {
            "p50": float(np.percentile(evaluated, 50)), "p90": float(np.percentile(evaluated, 90)),
            "p99": float(np.percentile(evaluated, 99)), "max": float(evaluated.max()),
        } if evaluated.size else None,
    }
    for threshold in thresholds:
        key = f"g{int(threshold)}"
        result[key] = float((evaluated > threshold).mean()) if evaluated.size else None
    if result.get("g10") is not None:
        result["unsafe_proxy_rule"] = bool(result["g10"] > unsafe_fraction)
    return result


def block_mean(elevation: np.ndarray, factor: int, nodata: float | None = None,
               min_valid_fraction: float = 1.0) -> tuple[np.ndarray, int]:
    """Aggregate k x k blocks (mean of valid pixels) to a coarser post spacing."""
    if factor <= 1:
        return elevation, 1
    mask = valid_mask(elevation, nodata).astype(np.float64)
    lines = (elevation.shape[0] // factor) * factor
    samples = (elevation.shape[1] // factor) * factor
    if not lines or not samples:
        raise ValueError("array smaller than one block")
    values = np.where(mask > 0, elevation, 0.0)[:lines, :samples]
    counts = mask[:lines, :samples]
    sums = values.reshape(lines // factor, factor, samples // factor, factor).sum(axis=(1, 3))
    count = counts.reshape(lines // factor, factor, samples // factor, factor).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, sums / np.maximum(count, 1), np.nan)
    keep = count >= min_valid_fraction * factor * factor
    return np.where(keep, mean, np.nan), factor


def layout_check(path: Path, offset_1based: int | None, expected_bytes: int | None,
                 lines: int | None, samples: int | None) -> dict:
    """Verify the PDS ^IMAGE pointer against the physical byte layout of the TIFF.

    PDS byte pointers are 1-based, so the 0-based start is ``offset - 1``; a reader that
    forgets this reads the data one byte early. The raw float32 view at that offset must be
    bit-identical to what the GeoTIFF reader returns.
    """
    info: dict = {"label_pointer_is_1_based": bool(offset_1based)}
    if not offset_1based:
        return info
    start = int(offset_1based) - 1
    info["zero_based_start"] = start
    info["bytes_after_offset"] = path.stat().st_size - start
    info["expected_bytes"] = expected_bytes
    if expected_bytes and info["bytes_after_offset"] == expected_bytes and lines and samples:
        raw = np.fromfile(path, dtype="<f4", count=int(lines) * int(samples),
                          offset=start).reshape(int(lines), int(samples))
        info["raw_offset_view"] = raw
    return info


def geotiff_metadata(path: Path) -> dict:
    """GeoKey / nodata / pixel-scale metadata straight from the GeoTIFF."""
    import tifffile

    with tifffile.TiffFile(path) as handle:
        page = handle.pages[0]
        meta = {"shape": list(page.shape), "dtype": str(page.dtype)}
        tags = {tag.name: tag.value for tag in page.tags.values()}
        scale = tags.get("ModelPixelScaleTag")
        meta["model_pixel_scale"] = [round(float(v), 6) for v in scale] if scale is not None else None
        tiepoint = tags.get("ModelTiepointTag")
        meta["model_tiepoint"] = [float(v) for v in tiepoint] if tiepoint is not None else None
        meta["gdal_nodata"] = ascii_tag(path, page, 42113)
        geo_keys = tags.get("GeoKeyDirectoryTag")
        if geo_keys:
            entries = np.asarray(geo_keys).reshape(-1, 4)[1:]
            ascii_params = tags.get("GeoAsciiParamsTag", "")
            doubles = np.asarray(tags.get("GeoDoubleParamsTag", []), dtype=float)
            resolved = {}
            for key_id, location, count, offset in entries:
                name = GEOKEY_NAMES.get(int(key_id), str(int(key_id)))
                if location == 0:
                    resolved[name] = int(offset)
                elif location == 34736:
                    resolved[name] = [float(v) for v in doubles[offset:offset + count]]
                else:
                    text = str(ascii_params)[offset:offset + count].strip("|\x00")
                    resolved[name] = text
            meta["geo_keys"] = resolved
    return meta


def probe_scene(tif_path: Path, label_path: Path, scales=(1, 2, 4, 8)) -> dict:
    import tifffile

    label = parse_pds_label(label_path.read_text(encoding="utf-8", errors="replace"))
    meta = geotiff_metadata(tif_path)
    elevation = tifffile.imread(tif_path).astype(np.float64)
    pixel = float(label["MAP_SCALE_M"] or 0.0) or None
    record = {
        "product_id": label.get("PRODUCT_ID"),
        "product_version": label.get("PRODUCT_VERSION_ID"),
        "data_set_id": label.get("DATA_SET_ID"),
        "instrument": label.get("INSTRUMENT_NAME"),
        "label_projection": label["PROJECTION"],
        "label_map_scale_m": label["MAP_SCALE_M"],
        "label_sphere_radius_km": label["SPHERE_RADIUS_KM"],
        "label_center_lat": label["CENTER_LAT"],
        "label_lat_range": [label["MIN_LAT"], label["MAX_LAT"]],
        "label_lon_range": [label["WEST_LON"], label["EAST_LON"]],
        "label_lines": label["LINES"],
        "label_line_samples": label["LINE_SAMPLES"],
        "label_sample_type": label["SAMPLE_TYPE"],
        "label_sample_bits": label["SAMPLE_BITS"],
        "label_nodata": label["NODATA"],
        "label_missing": label["MISSING"],
        "label_image_file": image_pointer(label)[0],
        "label_image_offset_bytes": image_pointer(label)[1],
        "geotiff": meta,
        "size_bytes": tif_path.stat().st_size,
    }
    expected_bytes = None
    if label["LINES"] and label["LINE_SAMPLES"] and label["SAMPLE_BITS"]:
        expected_bytes = int(label["LINES"]) * int(label["LINE_SAMPLES"]) * int(label["SAMPLE_BITS"]) // 8
    record["expected_data_bytes"] = expected_bytes
    offset = image_pointer(label)[1]
    record["shape_matches_label"] = (list(elevation.shape) == [label["LINES"], label["LINE_SAMPLES"]])
    file_dtype = np.dtype(meta["dtype"])
    record["dtype_matches_label"] = bool(
        file_dtype.itemsize == 4 and file_dtype.kind == "f"
        and label["SAMPLE_TYPE"] == "PC_REAL" and int(label["SAMPLE_BITS"]) == 32)
    layout = layout_check(tif_path, offset, expected_bytes, label["LINES"], label["LINE_SAMPLES"])
    record.update({key: value for key, value in layout.items() if key != "raw_offset_view"})
    if "raw_offset_view" in layout:
        record["raw_offset_view_identical_to_tiff"] = bool(np.array_equal(
            layout["raw_offset_view"].astype(np.float64), elevation, equal_nan=True))
    geotiff_nodata = meta.get("gdal_nodata")
    nodata_value = label["NODATA"]
    if nodata_value is None and geotiff_nodata:
        nodata_value = _to_float(geotiff_nodata)
        record["nodata_source"] = "GEOTIFF_GDAL_NODATA"
    else:
        record["nodata_source"] = label.get("NODATA_SOURCE", "LABEL")
    record["effective_nodata"] = nodata_value
    record["scales"] = {}
    for factor in scales:
        if pixel is None:
            break
        if factor > 1:
            try:
                grid, actual = block_mean(elevation, factor, label["NODATA"])
            except ValueError:
                continue
            if grid.shape[0] < 8 or grid.shape[1] < 8:
                continue
            stats = label_statistics(np.where(np.isnan(grid), MISSING_CONSTANT, grid),
                                     pixel * actual, nodata=MISSING_CONSTANT)
            stats["post_spacing_m"] = pixel * actual
        else:
            stats = label_statistics(elevation, pixel, nodata_value)
            stats["post_spacing_m"] = pixel
        record["scales"][f"factor_{factor}"] = stats
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/probe_p19"))
    parser.add_argument("--out", type=Path, default=Path("outputs/p19_io_probe.json"))
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args(argv)

    scenes = sorted(path.name[:-4] for path in args.data.glob("NAC_DTM_*.TIF"))
    if not scenes:
        raise SystemExit(f"no NAC_DTM_*.TIF under {args.data}")
    report = {"data_dir": str(args.data), "scenes": {}}
    import hashlib
    for name in scenes:
        tif = args.data / f"{name}.TIF"
        label_path = args.data / f"{name}.LBL"
        if not label_path.exists():
            raise SystemExit(f"missing label for {name}")
        record = probe_scene(tif, label_path, tuple(args.scales))
        record["tif_sha256"] = hashlib.sha256(tif.read_bytes()).hexdigest()
        record["label_sha256"] = hashlib.sha256(label_path.read_bytes()).hexdigest()
        report["scenes"][name] = record
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, record in report["scenes"].items():
        native = record["scales"].get("factor_1", {})
        print(f"{name}: shape={record['geotiff']['shape']} dtype={record['geotiff']['dtype']} "
              f"proj={record['label_projection']} scale={record['label_map_scale_m']}m "
              f"valid={native.get('valid_fraction'):.4f} g10={native.get('g10'):.4f} "
              f"unsafe={native.get('unsafe_proxy_rule')} "
              f"raw_identical={record.get('raw_offset_view_identical_to_tiff')}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
