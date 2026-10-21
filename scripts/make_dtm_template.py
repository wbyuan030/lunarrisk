"""Build a label-only ISIS map template (<scene>/dtm_map.cub) from <scene>/<dtm>.grid.json.

The template carries only a Mapping group; cam2map accepts it together with
pixres=map defaultrange=map and then emits exactly the DTM grid. Values come from
the DTM's own PDS label, parsed by pilot_p20.sh, so template and DTM cannot drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(grid: dict) -> str:
    return (
        "Group = Mapping\n"
        f"  ProjectionName     = {grid['projection'].capitalize()}\n"
        f"  CenterLongitude    = {float(grid['clon'])}\n"
        f"  CenterLatitude     = {float(grid['clat'])}\n"
        "  TargetName         = MOON\n"
        f"  EquatorialRadius   = {float(grid['radius']):.1f} <meters>\n"
        f"  PolarRadius        = {float(grid['radius']):.1f} <meters>\n"
        "  LatitudeType       = Planetocentric\n"
        "  LongitudeDirection = PositiveEast\n"
        "  LongitudeDomain    = 360\n"
        f"  MinimumLatitude    = {float(grid['minlat'])}\n"
        f"  MaximumLatitude    = {float(grid['maxlat'])}\n"
        f"  MinimumLongitude   = {float(grid['minlon'])}\n"
        f"  MaximumLongitude   = {float(grid['maxlon'])}\n"
        f"  Scale              = {float(grid['ppd'])} <pixels/degree>\n"
        "End_Group\n"
        "End\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path, help="<scene>/<dtm>.grid.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    out = args.out or args.grid.parent / "dtm_map.cub"
    if out.exists() and not args.force and out.stat().st_size > 200:
        print(f"{out} exists ({out.stat().st_size} B), keep")
        return 0
    out.write_text(build(grid), encoding="utf-8")
    print(f"{out} written ({out.stat().st_size} B) for {grid['dtm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
