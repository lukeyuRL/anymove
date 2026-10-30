#!/bin/bash
# Train the 672/192 high-res reader; original reader.pt is never overwritten.
# tools/train.sh CREATURE [epochs=20] [batch=8] [input=672] [heatmap=192] [dataset=runs/CREATURE/dataset_hr] [out=runs/CREATURE/reader_hr.pt] [workers=4]
# For a controlled 448/128 baseline, use the same dataset but a distinct output .pt.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
epochs=${2:-20}; batch=${3:-8}; size=${4:-672}; heatmap=${5:-192}
data=${6:-runs/$1/dataset_hr}; out=${7:-runs/$1/reader_hr.pt}; workers=${8:-4}
[ -d "$data" ] || { echo "missing dataset: $data; run tools/dataset.sh first" >&2; exit 1; }
extras=()
if [ "${MOCAP_LOW_VRAM:-0}" = "1" ]; then
  extras+=(--grad-accum "${MOCAP_GRAD_ACCUM:-8}" --gradient-checkpointing)
fi
if [ "${MOCAP_FREEZE_BACKBONE:-0}" = "1" ]; then
  extras+=(--freeze-backbone)
fi
"$PY" -m mocap.train "$data" "assets/$1/rigged/rig_info.json" "$out" "$epochs" "$batch" --input-size "$size" --heatmap-size "$heatmap" --workers "$workers" "${extras[@]}" 2>&1 | tee "runs/$1/train_hr.log"
