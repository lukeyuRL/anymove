#!/usr/bin/env bash
# Run locally from Codex/Git Bash after reviewing live GPU price and capacity.
# Creates exactly one GPU Pod with independent server-side stop/terminate guards.
set -Eeuo pipefail

fail() { echo "RUNPOD PROVISION: $*" >&2; exit 2; }
[[ "${ANYMOVE_APPROVE_SPEND:-}" == YES ]] || \
  fail "set ANYMOVE_APPROVE_SPEND=YES only after reviewing price and budget"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be exported locally; never commit it}"
: "${RUNPOD_GPU_ID:?Set RUNPOD_GPU_ID to an available type from Runpod}"
: "${RUNPOD_NETWORK_VOLUME_ID:?Select an EXISTING network volume in the GPU data center}"
: "${ANYMOVE_GPU_HOURLY_USD:?Set the verified current hourly GPU rate}"
: "${ANYMOVE_GPU_BUDGET_USD:?Set an explicit maximum GPU spend estimate}"
command -v runpodctl >/dev/null || fail "runpodctl CLI is required"

STOP_HOURS="${ANYMOVE_STOP_HOURS:-4}"
TERM_HOURS="${ANYMOVE_TERMINATE_HOURS:-6}"
[[ "$STOP_HOURS" =~ ^[1-9][0-9]*$ ]] || fail "invalid stop duration"
[[ "$TERM_HOURS" =~ ^[1-9][0-9]*$ ]] || fail "invalid terminate duration"
(( STOP_HOURS <= 12 )) || fail "stop duration exceeds the 12-hour safety limit"
(( TERM_HOURS > STOP_HOURS && TERM_HOURS <= 24 )) || fail "terminate must be later than stop and <= 24h"
[[ "$ANYMOVE_GPU_HOURLY_USD" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "invalid hourly GPU rate"
[[ "$ANYMOVE_GPU_BUDGET_USD" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "invalid budget"
awk -v rate="$ANYMOVE_GPU_HOURLY_USD" -v hours="$STOP_HOURS" \
    -v cap="$ANYMOVE_GPU_BUDGET_USD" 'BEGIN { if (rate <= 0 || cap <= 0 || rate * hours > cap) exit 1 }' || \
    fail "estimated rate * stop-hours exceeds GPU budget; aborting"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}"
NAME="${RUNPOD_POD_NAME:-anymove-$(date -u +%Y%m%d-%H%M%S)}"

echo "GPU type: $RUNPOD_GPU_ID"
echo "GPU estimated maximum before stop: USD $(awk -v r="$ANYMOVE_GPU_HOURLY_USD" -v h="$STOP_HOURS" 'BEGIN { printf "%.2f", r*h }')"
echo "Network volume: $RUNPOD_NETWORK_VOLUME_ID (storage charges are separate)"
echo "Hard stop: ${STOP_HOURS}h; hard terminate: ${TERM_HOURS}h"
echo "Image: $IMAGE"
runpodctl pod create \
  --name "$NAME" \
  --image "$IMAGE" \
  --gpu-id "$RUNPOD_GPU_ID" \
  --gpu-count 1 \
  --cloud-type SECURE \
  --network-volume-id "$RUNPOD_NETWORK_VOLUME_ID" \
  --container-disk-in-gb 40 \
  --stop-after "${STOP_HOURS}h" \
  --terminate-after "${TERM_HOURS}h" \
  --ssh
echo "Created. Record the Pod ID; get SSH access with: runpodctl pod get <POD_ID>"
echo "After success/failure: stop it immediately with Runpod MCP pod-action or runpodctl pod stop <POD_ID>."
