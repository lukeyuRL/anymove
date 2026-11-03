#!/usr/bin/env bash
# Execute inside a Runpod GPU Pod; requires /workspace to be a mounted network volume.
# All logs and checkpoints are written to /workspace, never an ephemeral image layer.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PY="${ANYMOVE_PYTHON:-python3}"
MODE="${ANYMOVE_MODE:-quality}"
CREATURE="${ANYMOVE_CREATURE:-dragon}"
VOLUME_ROOT="${ANYMOVE_VOLUME_ROOT:-/workspace}"
DATASET="${ANYMOVE_DATASET:-$VOLUME_ROOT/anymove-data/$CREATURE/dataset_hr}"
OUTPUT_ROOT="${ANYMOVE_OUTPUT_ROOT:-$VOLUME_ROOT/anymove-results}"
RUN_ID="${ANYMOVE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$MODE}"
EPOCHS="${ANYMOVE_EPOCHS:-20}"
BATCH="${ANYMOVE_BATCH:-2}"
ACCUM="${ANYMOVE_GRAD_ACCUM:-1}"
WORKERS="${ANYMOVE_WORKERS:-0}"
INPUT_SIZE="${ANYMOVE_INPUT_SIZE:-672}"
HEATMAP_SIZE="${ANYMOVE_HEATMAP_SIZE:-192}"
MAX_MINUTES="${ANYMOVE_MAX_TRAIN_MINUTES:-180}"
EXPERIMENT_ID="${ANYMOVE_EXPERIMENT_ID:-${CREATURE}_${MODE}_v1}"

die() { echo "RUNPOD TRAINING: $*" >&2; exit 2; }
[[ "$MODE" == quality || "$MODE" == hr ]] || die "ANYMOVE_MODE must be quality or hr"
[[ "$CREATURE" =~ ^[a-zA-Z0-9_-]+$ ]] || die "invalid creature name"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{1,79}$ ]] || die "invalid run id"
[[ "$EXPERIMENT_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]{1,79}$ ]] || die "invalid experiment id"
for value in "$EPOCHS" "$BATCH" "$ACCUM" "$INPUT_SIZE" "$HEATMAP_SIZE" "$MAX_MINUTES"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "epochs, batch, accum, resolutions and max minutes must be positive integers"
done
[[ "$WORKERS" =~ ^(0|[1-9][0-9]*)$ ]] || die "workers must be nonnegative"
(( MAX_MINUTES <= 720 )) || die "training time cap must be <= 720 minutes"
[[ -f "assets/$CREATURE/rigged/rig_info.json" ]] || die "missing tracked rig_info.json"
[[ -f "assets/$CREATURE/creature.json" ]] || die "missing creature.json"
command -v "$PY" >/dev/null || die "python not found: $PY"
command -v timeout >/dev/null || die "GNU timeout is required"
command -v mountpoint >/dev/null || die "mountpoint is required"
[[ -d "$VOLUME_ROOT" ]] && mountpoint -q "$VOLUME_ROOT" || \
  die "$VOLUME_ROOT is not a mount point; refusing to risk checkpoints on ephemeral storage"

"$PY" tools/runpod/preflight.py --dataset "$DATASET" \
  --rig-info "assets/$CREATURE/rigged/rig_info.json" \
  --creature "assets/$CREATURE/creature.json" \
  --min-poses "${ANYMOVE_MIN_POSES:-25}"

mkdir -p "$OUTPUT_ROOT"
# Full optimizer/scaler checkpoints belong to an EXPERIMENT, not to a Pod run.
# A replacement Pod with the same data/settings resumes after the last saved epoch.
export MOCAP_RESUME_DIR="${MOCAP_RESUME_DIR:-$VOLUME_ROOT/anymove-checkpoints/$EXPERIMENT_ID}"
export MOCAP_KEEP_CHECKPOINTS="${MOCAP_KEEP_CHECKPOINTS:-3}"
mkdir -p "$MOCAP_RESUME_DIR"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || die "run directory already exists: $RUN_DIR"
mkdir "$RUN_DIR"
export ANYMOVE_RUN_DIR="$RUN_DIR" ANYMOVE_RUN_ID="$RUN_ID"
export ANYMOVE_TRAIN_MODE="$MODE" ANYMOVE_TRAIN_DATASET="$DATASET"
export ANYMOVE_TRAIN_RESUME="$MOCAP_RESUME_DIR"
export ANYMOVE_TRAIN_GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

write_status() {
  ANYMOVE_RUN_STATE="$1" ANYMOVE_RUN_EXIT="$2" "$PY" - <<'PY'
import datetime, json, os, pathlib
folder = pathlib.Path(os.environ["ANYMOVE_RUN_DIR"])
dest = folder / "status.json"
old = json.loads(dest.read_text()) if dest.exists() else {}
old.update({
    "state": os.environ["ANYMOVE_RUN_STATE"],
    "exit_code": int(os.environ["ANYMOVE_RUN_EXIT"]),
    "run_id": os.environ["ANYMOVE_RUN_ID"],
    "mode": os.environ["ANYMOVE_TRAIN_MODE"],
    "dataset": os.environ["ANYMOVE_TRAIN_DATASET"],
    "git_sha": os.environ["ANYMOVE_TRAIN_GIT_SHA"],
    "resume_dir": os.environ["ANYMOVE_TRAIN_RESUME"],
    "pod_id": os.environ.get("RUNPOD_POD_ID", ""),
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
})
if "started_utc" not in old:
    old["started_utc"] = old["updated_utc"]
tmp = dest.with_suffix(".json.tmp")
tmp.write_text(json.dumps(old, indent=2, sort_keys=True), encoding="utf-8")
tmp.replace(dest)
PY
}
cleanup() {
  result=$?
  trap - EXIT
  if (( result == 0 )); then
    write_status succeeded 0
  elif (( result == 124 )); then
    write_status timed_out "$result"
  else
    write_status failed "$result"
  fi
  echo "Run $RUN_ID finished with exit code $result; check $RUN_DIR/status.json" >&2
  echo "STOP THE GPU POD via Codex Runpod MCP or runpodctl; hard TTL remains active." >&2
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_status preparing 0

if [[ "${ANYMOVE_INSTALL_DEPS:-1}" == 1 ]]; then
  echo "Installing pinned CUDA dependencies..."
  "$PY" -m pip install --disable-pip-version-check "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
  "$PY" -m pip install --disable-pip-version-check -r requirements.txt
fi
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available; refusing CPU-only training")
print("CUDA device:", torch.cuda.get_device_name(0), "Torch:", torch.__version__, flush=True)
PY

args=("$DATASET" "assets/$CREATURE/rigged/rig_info.json" "$RUN_DIR/reader.pt"
      "$EPOCHS" "$BATCH" --input-size "$INPUT_SIZE" --heatmap-size "$HEATMAP_SIZE"
      --workers "$WORKERS" --grad-accum "$ACCUM")
if [[ "$MODE" == quality ]]; then args+=(--train-quality); fi
if [[ "${ANYMOVE_GRADIENT_CHECKPOINTING:-0}" == 1 ]]; then args+=(--gradient-checkpointing); fi
if [[ "${ANYMOVE_FREEZE_BACKBONE:-0}" == 1 ]]; then args+=(--freeze-backbone); fi
write_status running 0
echo "Run: $RUN_ID; model: $RUN_DIR/reader.pt; time cap: $MAX_MINUTES minutes"
echo "Dataset: $DATASET; mode: $MODE; git: $ANYMOVE_TRAIN_GIT_SHA"
echo "Full-state backup / resume: $MOCAP_RESUME_DIR (same experiment ID required to resume)"
timeout --signal=TERM --kill-after=45s "${MAX_MINUTES}m" \
  "$PY" -u -m mocap.train "${args[@]}" 2>&1 | tee "$RUN_DIR/train.log"
echo "Training succeeded. Checkpoint: $RUN_DIR/reader.pt"
