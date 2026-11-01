#!/usr/bin/env bash
# Dino Joint-Specific View Reliability — use PRESPECIFIED 0.5 view multipliers.
#
# From isolated Dino GT repo (not the P2 training checkout):
#   export PYTHON=python.exe MOCAP_CREATURE=assets/dino
#   bash tools/eval_dino_view_reliability.sh \
#     runs/dino/gt_experiments/seed7_smoke_24
#
# Builds three weighted NPZ detections and solves each using the SAME
# original 2D Reader coordinates, unchanged source GT and original Solver.
# No Reader retraining, no P1/P2 stacking, no GT-target-informed weights.
# A NEW output directory is mandatory; every input is preserved untouched.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature dino

base=\${1:?usage: bash tools/eval_dino_view_reliability.sh EXISTING_GT_RUN_DIR [NEW_OUTPUT_DIR]}
target=\${2:-"$base/view_reliability_half_v1"}
rig=assets/dino/rigged/rig_info.json
for path in "$rig" "$base/gt_render/joints.json" "$base/gt_motion.json" \
            "$base/det_baseline.npz" "$base/motion_baseline.json" \
            "$base/camera_layout_audit.json"; do
  if [ ! -f "$path" ]; then
    echo "missing prior Dino GT artifact: $path" >&2
    exit 2
  fi
done
# Require independently audited Blender camera projections.
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="blender_cameras_match_anymove_orthographic_layout", "Blender camera did not pass 2-view audit"' \
  "$base/camera_layout_audit.json"
if [ -e "$target" ]; then
  echo "refusing to overwrite existing view reliability experiment: $target" >&2
  exit 2
fi

mkdir -p "$target"
echo "Dino view-weight ablation on EXACT existing GT: $base"
echo "Output: $target"
echo "All anatomical multipliers are fixed at 0.5 BEFORE Ground Truth comparison."

variants=(--variant "baseline=$base/motion_baseline.json")
detections=(--detection "baseline=$base/det_baseline.npz")

for profile in tail_front_half toes_side_half combined_half; do
  det="$target/det_\${profile}.npz"
  motion="$target/motion_\${profile}.json"
  "$PY" -m tools.make_dino_view_ablations \
    "$base/det_baseline.npz" "$rig" "$det" --profile "$profile"
  "$PY" -m mocap.solve "$det" "$rig" "$motion" --no-align
  variants+=(--variant "$profile=$motion")
  detections+=(--detection "$profile=$det")
done

"$PY" -m tools.evaluate_dino_full_body \
  "$base/gt_render/joints.json" "$rig" \
  "$target/full_body_quality" \
  "\${variants[@]}" "\${detections[@]}" \
  --source-motion "$base/gt_motion.json"

echo "DINO JOINT VIEW RELIABILITY A/B COMPLETE"
echo "Ground truth: $base/gt_render/joints.json"
echo "Results: $target/full_body_quality/full_body_report.json"
echo "Per Joint: $target/full_body_quality/by_joint.csv"
echo "Per Frame: $target/full_body_quality/by_frame.csv"
echo "Prespecified profiles only. This is controlled synthetic GT, NOT natural Dino video."
