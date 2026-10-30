#!/bin/bash
# High-res dataset: one RGBA render per view at full layout scale (default 1.0).
# tools/dataset.sh CREATURE [count=2200] [procs=3] [seed=500] [scale=1.0] [out=runs/CREATURE/dataset_hr]
# Keeps original runs/CREATURE/dataset untouched. Do not mix scale values in a resumable dataset.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
need_creature "${1:-}"
a=assets/$1; n=${2:-2200}; p=${3:-3}; seed=${4:-500}; scale=${5:-1.0}
out=${6:-runs/$1/dataset_hr}
if (( n < 1 || p < 1 )); then echo "count and procs must be positive" >&2; exit 2; fi
per=$(( (n + p - 1) / p ))
mkdir -p "$out"
pids=()
for k in $(seq 0 $((p - 1))); do
  start=$((k * per))
  (( start < n )) || break
  take=$((n - start))
  (( take > per )) && take=$per
  "$BLENDER" -b -P blender/render_dataset.py -- "$a/rigged/rigged.blend" "$a/rigged/rig_info.json" "$out" "$start" "$take" "$seed" "$scale" > "runs/$1/dataset_hr_$k.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then echo "Blender render failed; inspect runs/$1/dataset_hr_*.log" >&2; exit 1; fi
"$PY" tools/show_samples.py "$out" "runs/$1/dataset_hr_samples.jpg" 8
echo "$(find "$out" -name '*_meta.json' | wc -l) complete pose metadata files in $out"
