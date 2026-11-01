#!/usr/bin/env bash
# Non-production Top GT Oracle A/B.
# bash tools/eval_dino_top_oracle.sh GT_BASE TOP_RENDER_DIR NEW_SCOPE_DIR tail_only
#
# Input: independent Blender gt_render/joints.json + real top image renders.
# Output: new SHA-guarded top NPZ, Motion, 3D Truth Report in a new folder.
# NEVER use this script as actual Dino Top Reader validation: it LEAKS GT.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature dino

base=${1:?missing Blender GT base}
render=${2:?missing synchronized Top render dir}
output=${3:?missing new variant output dir}
scope=${4:?missing tail_only/all_joints}
if [[ "$scope" != "tail_only" && "$scope" != "all_joints" ]]; then
  echo "Oracle scope must be tail_only or all_joints" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "Refusing overwrite of existing Top Oracle output: $output" >&2
  exit 2
fi
rig=assets/dino/rigged/rig_info.json
truth="$base/gt_render/joints.json"
raw="$base/det_baseline.npz"
baseline="$base/motion_baseline.json"
for p in "$rig" "$truth" "$raw" "$baseline" \
         "$render/cam_top.json" "$render/top_render_audit.json"; do
    [[ -f "$p" ]] || { echo "Missing exact Top Oracle source: $p" >&2; exit 2; }
done

mkdir -p "$output"
det="$output/det_top_oracle.npz"
motion="$output/motion_top_oracle.json"
"$PY" -m tools.make_dino_top_oracle \
    "$raw" "$rig" "$truth" "$render" "$det" --scope "$scope" \
    > "$output/top_oracle_input.log"
echo "Top GT ORACLE LEAKAGE test: $scope"
"$PY" -m mocap.solve "$det" "$rig" "$motion" \
    --no-align --experimental-dino-top --allow-gt-top-oracle \
    --top-camera "$render/cam_top.json" \
    > "$output/top_oracle_solve.log" 2>&1 || {
      tail -n 60 "$output/top_oracle_solve.log" >&2
      exit 1
    }
"$PY" -m tools.evaluate_dino_full_body \
    "$truth" "$rig" "$output/full_body_quality" \
    --variant "baseline=$baseline" \
    --variant "top_oracle_$scope=$motion" \
    --detection "baseline=$raw" \
    --detection "top_oracle_$scope=$det" \
    --source-motion "$base/gt_motion.json"
echo "Top ORACLE ground-truth upper-bound QA: $output/full_body_quality/full_body_report.json"
