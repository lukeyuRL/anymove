#!/usr/bin/env bash
# Build Dino synthetic-motion ground truth in the REAL Blender rig, and
# compare identical-video Base / P1 guarded / optional P2 on ALL 92 joints.
#
# Windows Git Bash (from F:/anymove):
#   export PYTHON=python.exe
#   export BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
#   bash tools/eval_dino_full_body.sh runs/dino/gt_experiments/seed7_smoke 24 7
#   bash tools/eval_dino_full_body.sh runs/dino/gt_experiments/seed7_full 124 7 \
#       runs/dino/reader_hr_quality_p2_eval_snapshot.pt
#
# First 24-frame run is a pipeline check. The procedural animation is
# NON-natural and does not replace evaluation on real or AI Dino videos.
# Input/data artifacts under the output directory are never overwritten.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature dino

out=${1:?usage: bash tools/eval_dino_full_body.sh NEW_OUTPUT_DIR [frames=24] [seed=7] [P2_checkpoint.pt]}
frames=${2:-24}
seed=${3:-7}
p2=${4:-}
if [[ ! "$frames" =~ ^[0-9]+$ ]] || ((frames < 3 || frames > 1000)); then
  echo "frames must be integer 3..1000" >&2; exit 2
fi
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "seed must be a nonnegative integer" >&2; exit 2
fi
if [ -e "$out" ]; then
  echo "refusing to overwrite earlier ground-truth experiment: $out" >&2; exit 2
fi
rig=assets/dino/rigged/rig_info.json
blend=assets/dino/rigged/rigged.blend
reader=${DINO_GT_READER:-runs/dino/reader_hr_v1.pt}
ref=runs/dino/ref/ref.png
for path in "$rig" "$blend" "$reader" "$ref"; do
  [ -f "$path" ] || { echo "missing Dino ground-truth input: $path" >&2; exit 2; }
done
if [ -n "$p2" ]; then
  [ -f "$p2" ] || { echo "missing P2 Quality Reader checkpoint: $p2" >&2; exit 2; }
fi
if ! command -v "$BLENDER" >/dev/null 2>&1 && [ ! -x "$BLENDER" ]; then
  echo "BLENDER executable unavailable; set BLENDER to installed Blender 5 path" >&2
  exit 2
fi
engine=${DINO_GT_RENDER_ENGINE:-cycles}
samples=${DINO_GT_SAMPLES:-8}
if [[ "$engine" != "cycles" && "$engine" != "workbench" ]] ||
   [[ ! "$samples" =~ ^[0-9]+$ ]] || ((samples < 1 || samples > 128)); then
  echo "DINO_GT_RENDER_ENGINE must be cycles/workbench and samples 1..128" >&2; exit 2
fi

mkdir -p "$out"
echo "DINO GT: $out, frames $frames, seed $seed, engine $engine"
"$PY" -m mocap.exam make "$rig" "$out/gt_motion.json" "$seed" "$frames"

"$BLENDER" -b -P blender/apply_motion.py -- \
  "$blend" "$rig" "$out/gt_motion.json" "$out/gt_render" \
  --views front,side --engine "$engine" --samples "$samples" \
  --save-blend > "$out/blender_gt.log" 2>&1 || {
    echo "Blender GT failed; last Blender log lines:" >&2
    tail -n 45 "$out/blender_gt.log" >&2
    exit 1
  }

# A render series may fail after writing the Blender joint truth. Refuse a
# silently truncated MP4 even if its frame count superficially looks valid.
for view in front side; do
  count=$(find "$out/gt_render/$view" -maxdepth 1 -name 'f_*.png' -type f | wc -l)
  if ((count != frames)); then
    echo "incomplete Blender $view render: got $count, expected $frames" >&2
    exit 1
  fi
done
"$PY" -m tools.audit_dino_gt_camera \
  "$out/gt_render/joints.json" \
  "$out/gt_render/cam_front.json" "$out/gt_render/cam_side.json" \
  > "$out/camera_layout_audit.json"

"$PY" -m mocap.exam video "$out/gt_render" "$out/gt_video.mp4" 24
"$PY" -m mocap.detect \
  "$out/gt_video.mp4" "$reader" "$ref" \
  "$out/det_baseline.npz" "$out/overlay_baseline.mp4" --no-calib
"$PY" -m mocap.solve \
  "$out/det_baseline.npz" "$rig" "$out/motion_baseline.json" --no-align

# P1 guarded v2: same synthetic video and same baseline Reader NPZ.
"$PY" -m mocap.track \
  "$out/gt_video.mp4" "$out/det_baseline.npz" \
  "$out/det_p1_guarded.npz" --guard-large-disagreements
"$PY" -m mocap.solve \
  "$out/det_p1_guarded.npz" "$rig" "$out/motion_p1_guarded.json" --no-align

variants=(
  --variant "baseline=$out/motion_baseline.json"
  --variant "p1_guarded=$out/motion_p1_guarded.json"
)
detections=(
  --detection "baseline=$out/det_baseline.npz"
  --detection "p1_guarded=$out/det_p1_guarded.npz"
)
if [ -n "$p2" ]; then
  "$PY" -m mocap.detect \
    "$out/gt_video.mp4" "$p2" "$ref" \
    "$out/det_p2_quality.npz" "$out/overlay_p2_quality.mp4" --no-calib
  "$PY" -m tools.make_dino_p2_weighted_detection \
    "$out/det_baseline.npz" "$out/det_p2_quality.npz" \
    "$out/det_p2_weighted.npz" --checkpoint "$p2"
  "$PY" -m mocap.solve \
    "$out/det_p2_weighted.npz" "$rig" \
    "$out/motion_p2_weighted.json" --no-align
  variants+=(--variant "p2_weighted=$out/motion_p2_weighted.json")
  detections+=(--detection "p2_weighted=$out/det_p2_weighted.npz")
fi

"$PY" -m tools.evaluate_dino_full_body \
  "$out/gt_render/joints.json" "$rig" "$out/full_body_quality" \
  "${variants[@]}" "${detections[@]}" \
  --source-motion "$out/gt_motion.json"

echo "DINO FULL-BODY GT COMPLETE"
echo "Blender GT: $out/gt_render/joints.json"
echo "GT video: $out/gt_video.mp4"
echo "3D QA report: $out/full_body_quality/full_body_report.json"
echo "Joint CSV: $out/full_body_quality/by_joint.csv"
echo "Frame CSV: $out/full_body_quality/by_frame.csv"
echo "Original Dino 2200 poses and test_walk motion remain unchanged."
