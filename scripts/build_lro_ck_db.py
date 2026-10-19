#!/usr/bin/env python3
"""Generate the ISIS kernel-database entry file for LRO spacecraft-bus attitude CKs.

ISIS 8 resolves CK kernels through ``<ISISDATA>/<mission>/kernels/ck/kernels.????.conf``.
For LROC NAC that configuration lists ``moc_kernels.????.db`` (spacecraft bus attitude) and
``lroc_kernels.????.db`` (instrument attitude). The USGS data area ships the latter but no
``moc_kernels`` file, so spiceinit loads the camera CK yet never the bus attitude and fails
with "the ck rotation from frame -85000 cannot be found".

LRO CK file names carry their own coverage window as ``<prefix>_YYYYDDD_YYYYDDD_vNN.bc``, so
the database entries are derived from the names (with a one-day margin), which is exactly the
convention the USGS generator uses. When a name does not match, the script falls back to
measuring the kernel with SPICE (needs the leapsecond and SCLK kernels furnished).
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

NAME_RE = re.compile(r"_(?P<y1>\d{4})(?P<d1>\d{3})_(?P<y2>\d{4})(?P<d2>\d{3})_")
ET_MIN = -378691200.0   # 1988-01-01
ET_MAX = 2524608000.0   # 2050-01-01


ISIS_TIME = "{y:-04d} {mon} {d:02d} 00:00:00.000000 TDB"
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def isis_time(moment: dt.datetime) -> str:
    """ISIS kernel databases spell times as e.g. '2011 MAR 02 00:01:38.163911 TDB'."""
    return ISIS_TIME.format(y=moment.year, mon=MONTHS[moment.month - 1], d=moment.day)


def doy_to_date(year: int, doy: int) -> dt.datetime:
    return dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1)


def filename_bounds(kernel: Path) -> tuple[str, str] | None:
    match = NAME_RE.search(kernel.name)
    if not match:
        return None
    start = doy_to_date(int(match["y1"]), int(match["d1"])) - dt.timedelta(days=1)
    stop = doy_to_date(int(match["y2"]), int(match["d2"])) + dt.timedelta(days=1)
    return isis_time(start), isis_time(stop)


def prepare_pool(isisdir: Path) -> list[str]:
    import spiceypy

    lsk = sorted((isisdir / "base/kernels/lsk").glob("*.tls"))
    sclk = sorted((isisdir / "lro/kernels/sclk").glob("*.tsc"))
    if not lsk or not sclk:
        raise SystemExit("need a leapsecond kernel (.tls) and an SCLK kernel (.tsc) under ISISDATA")
    spiceypy.furnsh(str(lsk[-1]))
    spiceypy.furnsh(str(sclk[-1]))
    return [lsk[-1].name, sclk[-1].name]


def measured_bounds(kernel: Path, frame: int) -> tuple[str, str] | None:
    import spiceypy
    from spiceypy.utils.support_types import SPICEDOUBLE_CELL

    window = SPICEDOUBLE_CELL(2)
    spiceypy.wninsd(ET_MIN, ET_MAX, window)
    try:
        cover = spiceypy.ckcov(str(kernel), frame, False, "SEGMENT", 0.0, "TDB", window)
    except Exception as exc:  # noqa: BLE001
        print(f"  {kernel.name}: ckcov failed ({type(exc).__name__})")
        return None
    if cover.card < 2:
        return None
    def convert(et: float) -> str:
        iso = spiceypy.et2utc(et, "ISOC", 0)  # 2011-03-01T00:00:00
        return isis_time(dt.datetime.strptime(iso[:10], "%Y-%m-%d"))
    return convert(float(cover[0])), convert(float(cover[cover.card - 1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isisdir", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=-85000)
    parser.add_argument("--pattern", default="moc*.bc")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    print("furnished:", prepare_pool(args.isisdir))
    ckdir = args.isisdir / "lro/kernels/ck"
    out = args.out or (ckdir / "moc_kernels.0001.db")

    groups = []
    for kernel in sorted(ckdir.glob(args.pattern)):
        bounds = filename_bounds(kernel) or measured_bounds(kernel, args.frame)
        if bounds is None:
            print(f"skip {kernel.name}: cannot determine coverage")
            continue
        groups.append(
            "Group = KernelDb\n"
            f"  File = {kernel.name}\n"
            f"  Time = (\"{bounds[0]}\",\n            \"{bounds[1]}\")\n"
            "  Type = Reconstructed\n"
            "EndGroup\n")
        print(f"  {kernel.name}: {bounds[0]} .. {bounds[1]}")
    if not groups:
        raise SystemExit("no kernel produced coverage information")
    out.write_text("\n".join(groups), encoding="utf-8")
    print(f"wrote {out} ({len(groups)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
