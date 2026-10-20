#!/usr/bin/bash
# P20-03 pilot driver: orthorectify every camera of a prepared scene, then export raw.
# Usage: run_p20_ortho.sh <scene-dir> <cam-ck> <bus-ck> <spk>
set -euo pipefail
export ISISROOT=/opt/isis-env
export ISISDATA=/opt/isis-env/data
export PATH=$ISISROOT/bin:$PATH
SCENE=${1:?scene dir}
CK_CAM=${2:?camera ck}
CK_BUS=${3:?bus ck}
SPK=${4:?spk}
cd "$SCENE"
IS=$ISISDATA/lro/kernels

for img in *.IMG; do
  [ -e "$img" ] || continue
  size=$(stat -c%s "$img")
  tag=$(python3 -c "import sys;print('L' if sys.argv[1].endswith('LC.IMG') else 'R')" "$img")
  echo "=== $img (${size} B)"
  [ "$size" -gt 1000000 ] || { echo "SKIP truncated"; continue; }
  cube="${img%.IMG}_$tag.cub"
  [ -f "$cube" ] || python3 /data/lunarrisk/scripts/prepare_lroc_cdr_cube.py "$img" \
    --isisdir "$ISISDATA" --out "$cube" --ck "$CK_CAM" "$CK_BUS" \
    --frame-kernel "$IS/fk/lro_frames_2014049_v01.tf" --spk "$SPK" \
    --ik "$IS/ik/lro_lroc_v20.ti" --iak "$IS/iak/lro_instrumentAddendum_v05.ti" \
    --sclk "$IS/sclk/lro_clkcor_2026287_v00.tsc" \
    --lsk "$ISISDATA/base/kernels/lsk/naif0012.tls" \
    --pck "$ISISDATA/base/kernels/pck/pck00010.tpc" \
    --tspk "$IS/tspk/moon_pa_de421_1900-2050.bpc" "$IS/tspk/de421.bsp" \
    || { echo "spiceinit FAILED"; continue; }
  echo -n "  $tag NaifIkCode="; getkey from="$cube" grpname=Kernels keyword=NaifIkCode
  ortho="${img%.IMG}_${tag}_ortho.cub"
  [ -f "$ortho" ] || cam2map from="$cube" map=dtm_map.cub to="$ortho" pixres=map defaultrange=map \
    || { echo "cam2map FAILED"; continue; }
  echo -n "  ortho Lines="; getkey from="$ortho" grpname=Dimensions keyword=Lines
  echo -n "  ortho Samples="; getkey from="$ortho" grpname=Dimensions keyword=Samples
  isis2raw from="$ortho" to="${img%.IMG}_${tag}_ortho.raw" storageorder=BSQ endian=LSB
done
echo "ortho images: $(ls *_ortho.raw 2>/dev/null | wc -l)"
