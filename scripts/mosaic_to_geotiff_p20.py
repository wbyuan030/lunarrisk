"""P20-01: export a mosaic raw as a GeoTIFF carrying the DTM's own georeferencing.

The mosaic shares the DTM's pixel grid exactly (that is what cam2map produced and what the grid
check verified), so the correct georeferencing is the DTM's, copied tag for tag. Deriving it again
from lat/lon bounds would be a second, independent chance to get the projection wrong.

Tags copied: ModelPixelScaleTag, ModelTiepointTag, GeoKeyDirectoryTag, GeoDoubleParamsTag,
GeoAsciiParamsTag. GDAL_NODATA is set to "nan" because the mosaic uses NaN for uncovered pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

GEO_TAGS = {33550: ("scale", "d"), 33922: ("tiepoint", "d"), 34735: ("geokeys", "H"),
            34736: ("double_params", "d"), 34737: ("ascii_params", "s"), 42113: ("nodata", "s")}


def read_geo_tags(path: Path) -> dict:
    """Collect the georeferencing tags of an existing GeoTIFF, keyed by name."""
    found = {}
    with tifffile.TiffFile(path) as handle:
        page = handle.pages[0]
        for code, (name, _) in GEO_TAGS.items():
            tag = page.tags.get(code)
            if tag is not None:
                found[name] = tag.value
        found["shape"] = page.shape
    missing = {"scale", "tiepoint", "geokeys"} - set(found)
    if missing:
        raise ValueError(f"{path} lacks georeferencing tags {sorted(missing)}")
    return found


def build_extratags(tags: dict, nodata: str = "nan") -> list:
    """tifffile extratags list, in tag-code order."""
    out = []
    for code in sorted(GEO_TAGS):
        name, kind = GEO_TAGS[code]
        if name == "nodata":
            value = nodata
        elif name not in tags:
            continue
        else:
            value = tags[name]
        if kind == "s":
            value = value if value.endswith("\x00") else value + "\x00"
            out.append((code, "s", len(value), value, False))
        else:
            values = tuple(np.atleast_1d(value).tolist())
            out.append((code, kind, len(values), values, False))
    return out


def export(raw: Path, dtm: Path, out: Path, lines: int, samples: int) -> dict:
    tags = read_geo_tags(dtm)
    if tags["shape"] != (lines, samples):
        raise ValueError(f"grid mismatch: DTM {tags['shape']} vs {lines}x{samples}")
    data = np.fromfile(raw, dtype="<f4").reshape(lines, samples).astype(np.float32)
    data[data == 0] = np.nan                     # the raw writer stores uncovered pixels as 0
    tifffile.imwrite(out, data, extratags=build_extratags(tags), photometric="minisblack")
    check = read_geo_tags(out)
    return {"out": out.name, "shape": list(data.shape),
            "finite_fraction": float(np.isfinite(data).mean()),
            "georeferencing_matches_dtm": all(
                np.array_equal(np.atleast_1d(tags[k]), np.atleast_1d(check[k]))
                for k in ("scale", "tiepoint", "geokeys")),
            "tags_written": [name for name in ("scale", "tiepoint", "geokeys",
                                               "double_params", "ascii_params", "nodata")
                             if name in check]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dtm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lines", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = export(args.raw, args.dtm, args.out, args.lines, args.samples)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
