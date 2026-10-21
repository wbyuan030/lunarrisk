"""Derive the sun azimuth/elevation for a NAC scene from SPICE (campt sub-solar point).

The CDR labels carry no illumination geometry and cam2map does not propagate any, so the
hillshade used for QA needs a solar direction. ``campt`` reports the sub-solar point, and the
azimuth of the sun seen from the scene follows from spherical trigonometry:

    az = atan2(sin(dLon) cos(lat_s), cos(lat) sin(lat_s) - sin(lat) cos(lat_s) cos(dLon))

with dLon = lon_s - lon. Elevation is 90 deg - incidence, taken from campt directly.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


def campt_geometry(cube: Path, line: float, sample: float) -> dict:
    proc = subprocess.run(["campt", f"from={cube}", "coordtype=image",
                           f"line={line}", f"sample={sample}"],
                          capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-400:])
    wanted = {"PlanetocentricLatitude": "lat", "PositiveEast360Longitude": "lon",
              "SubSolarLatitude": "sub_lat", "SubSolarLongitude": "sub_lon",
              "Incidence": "incidence", "Emission": "emission", "Phase": "phase"}
    out = {}
    for text in proc.stdout.splitlines():
        if "=" not in text:
            continue
        key, _, value = text.partition("=")
        key = key.strip()
        if key in wanted:
            out[wanted[key]] = float(value.split()[0])
    missing = {"lat", "lon", "sub_lat", "sub_lon"} - set(out)
    if missing:
        raise RuntimeError(f"campt output missing {sorted(missing)}")
    return out


def solar_azimuth(lat: float, lon: float, sub_lat: float, sub_lon: float) -> float:
    """Azimuth of the sub-solar point as seen from (lat, lon), degrees clockwise from north."""
    d_lon = math.radians(sub_lon - lon)
    la, sl = math.radians(lat), math.radians(sub_lat)
    east = math.sin(d_lon) * math.cos(sl)
    north = math.cos(la) * math.sin(sl) - math.sin(la) * math.cos(sl) * math.cos(d_lon)
    return math.degrees(math.atan2(east, north)) % 360.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--point", action="append", default=None,
                        help="line,sample (repeatable); defaults to the cube centre")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.point:
        points = [tuple(float(v) for v in p.split(",")) for p in args.point]
    else:
        lines = int(subprocess.run(["getkey", f"from={args.cube}", "grpname=Dimensions",
                                    "keyword=Lines"], capture_output=True, text=True).stdout)
        samples = int(subprocess.run(["getkey", f"from={args.cube}", "grpname=Dimensions",
                                      "keyword=Samples"], capture_output=True, text=True).stdout)
        points = [(lines / 2, samples / 2), (lines * 0.1, samples * 0.1),
                  (lines * 0.9, samples * 0.9)]

    rows = []
    for line, sample in points:
        geom = campt_geometry(args.cube, line, sample)
        azimuth = solar_azimuth(geom["lat"], geom["lon"], geom["sub_lat"], geom["sub_lon"])
        rows.append({"line": line, "sample": sample, **geom,
                     "sun_azimuth_deg": round(azimuth, 2),
                     "sun_elevation_deg": round(90.0 - geom["incidence"], 2)})

    report = {"cube": args.cube.name, "points": rows,
              "sun_azimuth_deg": round(sum(r["sun_azimuth_deg"] for r in rows) / len(rows), 2),
              "sun_elevation_deg": round(sum(r["sun_elevation_deg"] for r in rows) / len(rows), 2),
              "azimuth_spread_deg": round(max(r["sun_azimuth_deg"] for r in rows)
                                          - min(r["sun_azimuth_deg"] for r in rows), 2)}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
