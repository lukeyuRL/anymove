#!/bin/bash
# Per clip: read every joint in both views, fit the rig, and write the animation as a GLB with renders and a
# side-by-side comparison of the AI clip and the rig.
# tools/capture.sh CREATURE VIDEO.mp4 NAME [views=persp,side]     -> runs/CREATURE/clips/NAME/
#   det.npz (joints read), overlay.mp4, motion.json (the fit), render/animation.glb, showcase.mp4
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
c=$1; video=${2:?video}; name=${3:?name}; views=${4:-persp,side}
# Prefer high-res checkpoint when present; fall back to reader.pt.
# Fifth argument or MOCAP_READER_PT overrides the choice.
reader=${5:-${MOCAP_READER_PT:-}}
if [ -z "$reader" ]; then
  if [ -f "runs/$c/reader_hr.pt" ]; then reader="runs/$c/reader_hr.pt"
  else reader="runs/$c/reader.pt"; fi
fi
[ -f "$reader" ] || { echo "missing reader checkpoint: $reader" >&2; exit 1; }
rig=assets/$c/rigged; out=runs/$c/clips/$name
# the 3/4 showcase camera: creature.json "camera" (frame it to the creature's biggest move; see anymove-capture)
read -r cam target lens < <($PY -c "import json; c = json.load(open('assets/$c/creature.json')).get('camera', {}); print(c.get('cam', '-3.2,-3.0,1.6'), c.get('target', '0,0,0.5'), c.get('lens', 50))")
mkdir -p "$out"
$PY -m mocap.detect "$video" "$reader" "runs/$c/ref/ref.png" "$out/det.npz" "$out/overlay.mp4" | tail -1
# The visibility-aware solver is experimental and stays disabled by default.
# Use a NEW capture name when comparing, to preserve existing motion.json.
solver_flags=()
if [ "${MOCAP_SOLVER_VISIBILITY:-0}" = "1" ] && [ "${MOCAP_TEMPORAL_ASSOCIATION:-0}" = "1" ]; then
  echo "Choose either visibility weighting or temporal association, not both" >&2; exit 2
fi
if [ "${MOCAP_SOLVER_VISIBILITY:-0}" = "1" ]; then
  [ "$c" = "dragon" ] || { echo "visibility experiment supports Dragon only" >&2; exit 2; }
  solver_flags+=(--visibility-aware)
fi
if [ "${MOCAP_TEMPORAL_ASSOCIATION:-0}" = "1" ]; then
  [ "$c" = "dragon" ] || { echo "temporal association supports Dragon only" >&2; exit 2; }
  solver_flags+=(--temporal-association)
fi
$PY -m mocap.solve "$out/det.npz" "$rig/rig_info.json" "$out/motion.json" "${solver_flags[@]}" | grep -v "iter"
$BLENDER -b -P blender/apply_motion.py -- "$rig/rigged.blend" "$rig/rig_info.json" "$out/motion.json" "$out/render" \
  --views "$views" --samples 24 --cam "$cam" --target "$target" --lens "$lens" --glb > "$out/blender.log" 2>&1
$PY -m mocap.showcase "$video" "$out/det.npz" "$out/render" "$out/showcase.mp4" "$c $name"
if command -v ffmpeg > /dev/null; then
  ffmpeg -loglevel error -y -i "$out/showcase.mp4" -c:v libx264 -crf 20 -pix_fmt yuv420p "$out/showcase_h264.mp4"
fi
echo "done: $out (animation: $out/render/animation.glb)"
