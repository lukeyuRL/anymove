#!/usr/bin/env bash
# Actual P2 Quality Reader -> same-raw-UV weighted Dino Solver -> A/B video review.
#
# From Windows Git Bash (F:/anymove):
# export PYTHON=python.exe
# bash tools/eval_dino_p2.sh VIDEO.mp4 CLIP_DIR READER_QUALITY_SNAPSHOT.pt [TAG]
#
# VIDEO: runs/dino/videos/test_walk/s1/h3max.mp4
# CLIP_DIR: existing capture directory with det.npz and motion.json.
# TAG: defaults to p2_eval; NEW tag required if outputs already exist.
#
# All outputs stay under CLIP_DIR/TAG. Original Reader, motion and checkpoints
# are immutable. The P2 synthetic visibility score is NOT used by Solver.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature dino

video=${1:?usage: bash tools/eval_dino_p2.sh VIDEO.mp4 CLIP_DIR SNAPSHOT.pt [TAG]}
clip=${2:?missing existing Dino capture directory}
model=${3:?missing copied quality Reader snapshot}
tag=${4:-p2_eval}
if [[ ! "$tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "tag must use only letters/numbers/dot/underscore/dash" >&2; exit 2
fi
for p in "$video" "$model" "$clip/det.npz" "$clip/motion.json" "runs/dino/ref/ref.png" "assets/dino/rigged/rig_info.json"; do
  [ -f "$p" ] || { echo "missing required Dino P2 input: $p" >&2; exit 2; }
done
folder="$clip/$tag"
if [ -e "$folder" ]; then
  echo "refusing overwrite of existing P2 experiment folder: $folder" >&2
  exit 2
fi
mkdir -p "$folder"
echo "P2 model snapshot: $model"
echo "P2 run output: $folder"

"$PY" -m mocap.detect "$video" "$model" "runs/dino/ref/ref.png" \
  "$folder/det_p2_quality.npz" "$folder/overlay_p2_joints.mp4"

"$PY" -m tools.review_dino_p2 "$video" "$folder/det_p2_quality.npz" \
  "$folder/quality_review.mp4" --raw-det "$clip/det.npz" \
  --report "$folder/quality_review.json"

"$PY" -m tools.make_dino_p2_weighted_detection \
  "$clip/det.npz" "$folder/det_p2_quality.npz" \
  "$folder/det_p2_weighted.npz" --checkpoint "$model"

"$PY" -m mocap.solve "$folder/det_p2_weighted.npz" \
  assets/dino/rigged/rig_info.json "$folder/motion_p2_weighted.json"

"$PY" -m tools.compare_dino_p2_motion \
  "$clip/det.npz" "$folder/det_p2_weighted.npz" \
  "$clip/motion.json" "$folder/motion_p2_weighted.json" \
  assets/dino/rigged/rig_info.json --out "$folder/p2_motion_comparison.json"

echo "DINO P2 EXPERIMENT COMPLETE"
echo "Video preview: $folder/quality_review.mp4"
echo "Quality report: $folder/quality_review.json"
echo "Motion candidate: $folder/motion_p2_weighted.json"
echo "Motion comparison: $folder/p2_motion_comparison.json"
echo "NOT validated 3D ground truth: visually compare the actual Dino animation."
