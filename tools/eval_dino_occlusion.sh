#!/usr/bin/env bash
# Controlled Dino Blender geometry / observed-only Solver A/B.
# bash tools/eval_dino_occlusion.sh runs/dino/gt_experiments/seed7_smoke_24 \
#   occlusion_v1 mesh,p2_error,p2_error_temporal,oracle_gt \
#   runs/dino/reader_hr_quality_p2_eval_snapshot.pt
# None of the original Reader, Motion, Blender GT or checkpoints are modified.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature dino
base=${1:?missing GT base directory}
tag=${2:?missing new experiment tag}
profiles=${3:?missing comma-separated profiles}
p2=${4:-}
threshold=${DINO_OCCLUSION_ERROR_THRESHOLD_PX:-10}
strength=${DINO_OCCLUSION_TEMPORAL_STRENGTH:-2}
if [[ ! "$tag" =~ ^[A-Za-z0-9_.-]+$ ]] ||
   [[ ! "$threshold" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   [[ ! "$strength" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "invalid experiment tag, threshold or temporal strength" >&2; exit 2
fi
if [ -e "$base/$tag" ]; then
    echo "refusing overwrite of $base/$tag" >&2; exit 2
fi
rig=assets/dino/rigged/rig_info.json
gt="$base/gt_render/joints.json"
anim="$base/gt_render/animation.blend"
raw="$base/det_baseline.npz"
motion="$base/motion_baseline.json"
for path in "$rig" "$gt" "$anim" "$raw" "$motion"; do
    [ -f "$path" ] || { echo "missing verified Dino GT input: $path" >&2; exit 2; }
done
IFS=',' read -ra modes <<< "$profiles"
if (( ${#modes[@]} < 1 )); then echo "no modes" >&2; exit 2; fi
need_p2=0
for mode in "${modes[@]}"; do
    case "$mode" in
        mesh|p2_error|mesh_p2|oracle_gt|p2_error_temporal|mesh_temporal|mesh_p2_temporal|oracle_gt_temporal) ;;
        *) echo "unsupported occlusion experiment: $mode" >&2; exit 2;;
    esac
    if [[ "$mode" == *p2* ]]; then need_p2=1; fi
done
if ((need_p2)) && [ ! -f "$p2" ]; then
    echo "a fixed P2 model-only snapshot is required" >&2; exit 2
fi
out="$base/$tag"
mkdir -p "$out"
native_path() {
    "$PY" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1"
}
"$BLENDER" -b "$(native_path "$anim")" \
    -P "$(native_path "blender/export_dino_mesh_visibility.py")" -- \
    "$(native_path "$rig")" "$(native_path "$gt")" \
    "$(native_path "$out/mesh_geometry.npz")" > "$out/blender_geometry.log" 2>&1 || {
        echo "Blender geometry failed:" >&2
        tail -n 40 "$out/blender_geometry.log" >&2
        exit 1
    }
"$PY" -m tools.audit_dino_gt_camera "$gt" \
    "$base/gt_render/cam_front.json" "$base/gt_render/cam_side.json" \
    > "$out/camera_audit.json"
if ((need_p2)); then
    "$PY" -m mocap.detect "$base/gt_video.mp4" "$p2" \
        runs/dino/ref/ref.png "$out/det_p2_quality.npz" --no-calib
fi

variants=(--variant "baseline=$motion")
detections=(--detection "baseline=$raw")
for variant in "${modes[@]}"; do
    mode="$variant"
    temporal=0
    if [[ "$mode" == *_temporal ]]; then
        mode=${mode%_temporal}
        temporal=1
    fi
    mask_args=(--mode "$mode" --threshold-px "$threshold")
    if [[ "$mode" == mesh* ]]; then
        mask_args+=(--geometry "$out/mesh_geometry.npz" --geometry-truth "$gt")
    fi
    if [[ "$mode" == *p2* ]]; then
        mask_args+=(--quality "$out/det_p2_quality.npz")
    fi
    if [[ "$mode" == oracle_gt ]]; then
        mask_args+=(--truth "$gt")
    fi
    det="$out/det_$variant.npz"
    solved="$out/motion_$variant.json"
    "$PY" -m tools.make_dino_occlusion_ablations \
        "$raw" "$rig" "$det" "${mask_args[@]}" > "$out/mask_$variant.log"
    solve_flags=(--no-align)
    if ((temporal)); then
        solve_flags+=(--masked-temporal-strength "$strength")
    fi
    echo "Solving new masked Dino variant: $variant"
    "$PY" -m mocap.solve "$det" "$rig" "$solved" \
        "${solve_flags[@]}" > "$out/solver_$variant.log" 2>&1 || {
            echo "Solver failed ($variant):" >&2
            tail -n 50 "$out/solver_$variant.log" >&2
            exit 1
        }
    variants+=(--variant "$variant=$solved")
    detections+=(--detection "$variant=$det")
    tail -n 4 "$out/solver_$variant.log"
done

"$PY" -m tools.evaluate_dino_full_body \
    "$gt" "$rig" "$out/ground_truth_comparison" \
    "${variants[@]}" "${detections[@]}" \
    --source-motion "$base/gt_motion.json"
echo "DINO OCCLUSION A/B COMPLETE: $out/ground_truth_comparison/full_body_report.json"
echo "The GT oracle variant uses truth leakage; never deploy it."
