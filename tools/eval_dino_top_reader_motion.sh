#!/usr/bin/env bash
# Top image-reader predictions -> opt-in three-view Solver -> Blender 3D GT A/B.
# NO Blender truth enters the camera-observation NPZ being fed to Solver.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature dino
base=${1:?missing original Blender GT base directory}
render=${2:?missing synchronized Top render directory}
top_reader=${3:?missing actual Top Reader prediction directory}
out=${4:?missing NEW output directory}
if [ -e "$out" ]; then
    echo "refusing overwrite of $out" >&2
    exit 2
fi
rig=assets/dino/rigged/rig_info.json
raw="$base/det_baseline.npz"
truth="$base/gt_render/joints.json"
baseline="$base/motion_baseline.json"
for p in "$rig" "$raw" "$truth" "$baseline" \
         "$render/cam_top.json" \
         "$top_reader/top_reader_detections.npz" \
         "$top_reader/top_reader_2d_gt_report.json"; do
    [ -f "$p" ] || { echo "missing actual Top Reader evidence: $p" >&2; exit 2; }
done
mkdir -p "$out"
"$PY" -m tools.make_dino_top_reader_detection \
    "$raw" "$top_reader/top_reader_detections.npz" "$rig" "$render" \
    "$out/det_top_reader.npz" > "$out/detection_bridge.log"

"$PY" -m mocap.solve "$out/det_top_reader.npz" "$rig" \
    "$out/motion_top_reader.json" \
    --no-align --experimental-dino-top \
    --top-camera "$render/cam_top.json" > "$out/solver.log" 2>&1 || {
       tail -n 60 "$out/solver.log" >&2; exit 1
    }

"$PY" -m tools.evaluate_dino_full_body "$truth" "$rig" \
    "$out/full_body_quality" \
    --variant "baseline=$baseline" \
    --variant "top_image_reader=$out/motion_top_reader.json" \
    --detection "baseline=$raw" \
    --detection "top_image_reader=$out/det_top_reader.npz" \
    --source-motion "$base/gt_motion.json"
echo "REAL TOP IMAGE READER A/B: $out/full_body_quality/full_body_report.json"
echo "This original Reader was not trained for Top; do not promote without independent held-out validation."
