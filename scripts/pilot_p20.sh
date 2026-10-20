#!/usr/bin/env bash
# P2.0 P20-03 pilot: run the verified CDR -> ortho chain for the pilot scenes.
# Usage: pilot_p20.sh <scene-list-file>   (lines: DTM_NAME ACQ_L PRODUCT_L ACQ_R PRODUCT_R)
set -euo pipefail
export ISISROOT=/opt/isis-env
export ISISDATA=/opt/isis-env/data
export PATH=$ISISROOT/bin:$PATH
IS=$ISISDATA/lro/kernels
DATA=/data/lunarrisk/data/pilot_p20
LIST=${1:?scene list file}
CDR_BASE=https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0

while read -r dtm acq_l prod_l acq_r prod_r; do
  [ -z "${dtm:-}" ] && continue
  echo "########## $dtm"
  mkdir -p "$DATA/$dtm"
  cd "$DATA/$dtm"

  # DTM (2 m) from the RDR bundle
  if [ ! -f NAC_DTM_${dtm}.TIF ]; then
    curl -fL --retry 2 --max-time 1800 \
      "https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/SDP/NAC_DTM/${dtm}/NAC_DTM_${dtm}.TIF" \
      -o NAC_DTM_${dtm}.TIF
    curl -fL --max-time 300 \
      "https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/SDP/NAC_DTM/${dtm}/NAC_DTM_${dtm}.LBL" \
      -o NAC_DTM_${dtm}.LBL
  fi

  # CDR pair
  for prod in "$prod_l" "$prod_r"; do
    [ -f "${prod}.IMG" ] || curl -fL --retry 2 --max-time 1800 \
      "${CDR_BASE}/${prod%/*}/${prod##*/}" -o "${prod##*/}.IMG"
  done

  python3 - "$dtm" <<'PY'
import json
import re
import sys

dtm = sys.argv[1]
label = open(f"NAC_DTM_{dtm}.LBL", encoding="latin-1").read()


def num(key):
    match = re.search(rf"^\s*{key}\s*=\s*(-?[0-9.]+)", label, re.M)
    if not match:
        raise SystemExit(f"{dtm}.LBL missing {key}")
    return match.group(1)


info = {
    "dtm": dtm,
    "projection": ("polarstereographic"
                   if abs(float(num("CENTER_LATITUDE"))) > 60 else "equirectangular"),
    "clat": num("CENTER_LATITUDE"),
    "clon": num("CENTER_LONGITUDE"),
    "scale": num("MAP_SCALE"),
    "radius": num("A_AXIS_RADIUS"),
    "lines": num("LINE_LAST_PIXEL"),
    "samples": num("SAMPLE_LAST_PIXEL"),
    "ppd": num("MAP_RESOLUTION"),
    "minlat": num("MINIMUM_LATITUDE"),
    "maxlat": num("MAXIMUM_LATITUDE"),
    "minlon": num("WESTERNMOST_LONGITUDE"),
    "maxlon": num("EASTERNMOST_LONGITUDE"),
}
json.dump(info, open(f"{dtm}.grid.json", "w"), indent=2)
print(json.dumps(info))
PY
  echo "########## $dtm prepared"
done < "$LIST"
