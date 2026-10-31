# AnyMove → Runpod GPU training (Codex + official Runpod MCP)

Status: **integration prepared; real Runpod account/GPU/SSH test pending**.
Based on `iClone_High_Res`. This is for the existing **two-view DINOv2 Reader**,
especially the opt-in `--train-quality` synthetic-cutout quality head. It does
**not** enable four-view inference or train a learned 3D motion prior. P1
temporal tracking is a separate post-detection stage.

## Architecture and trust boundaries

- **Codex + Runpod MCP:** find a suitable GPU, review the live cost, list
  existing network volumes/Pods, inspect status, and stop the Pod on exit.
- **Local Runpod CLI:** `tools/runpod/provision.sh` creates ONE Pod only after
  explicit authorization. The Runpod-provided `--stop-after` (default 4h)
  and `--terminate-after` (default 6h) provide independent timeout guards
  even if Codex disconnects. These are not a guaranteed dollar ceiling.
- **Inside Pod:** `tools/runpod/train.sh` checks the real dataset, installs the
  pinned dependencies, checks CUDA, trains, emits status/metrics, and stores
  an atomically published inference checkpoint after each epoch and full model/optimizer/scaler backup in a stable experiment directory.
- **Result:** `/workspace/anymove-results/<run-id>/reader.pt`,
  `status.json` and `train.log`, plus resumable full-state files under
  `/workspace/anymove-checkpoints/<experiment-id>/`. Git SHA, run ID, and Pod ID are recorded
  in status (no secrets).
- **Safety:** no Runpod API key is transferred into the GPU Pod. Once a run
  finishes Codex must immediately call the Runpod MCP Pod stop action.
  If the controller disappears, the service-side stop/terminate TTLs
  are the fallback; the GPU may continue billing until the stop deadline.
  Network volumes continue to accrue storage fees after the Pod stops.

## 1. Connect Codex to the official Runpod MCP

Use your existing local Codex CLI. In `~/.codex/config.toml` (on Windows,
`%USERPROFILE%\.codex\config.toml`) add:

```toml
[mcp_servers.runpod]
url = "https://mcp.runpod.io/"
bearer_token_env_var = "RUNPOD_API_KEY"

[mcp_servers.runpod-docs]
url = "https://docs.runpod.io/mcp"
```

Export `RUNPOD_API_KEY` into the environment **before launching Codex**.
The key stays on your local machine; do not paste it into chats, commit it,
put it in the Pod environment, or insert it into the TOML file. Use a Runpod
API key appropriate for management of your own Pods. The official MCP also
offers OAuth, but the separate `runpodctl` CLI requires its own authentication.
Verify the installed Codex version recognizes this MCP configuration and that
it can list your Pods before proceeding.

References:
- https://github.com/runpod/runpod-mcp
- https://github.com/runpod/runpod-plugins-official
- https://docs.runpod.io/get-started/mcp-servers

## 2. Prepare and transfer dataset BEFORE training

The generated Dragon dataset is deliberately **not in Git**. From
`F:/anymove`, on branch `feature/runpod-managed-training`:

```bash
python.exe tools/runpod/preflight.py \
  --dataset runs/dragon/dataset_hr \
  --min-poses 2200
```

This checks pose metadata, both PNGs per pose, resolution from
`assets/dragon/creature.json`, joint count, uniform render scale and blur
profile, and the modulo-25 validation split. If it fails, repair/re-render
locally; do not rent a GPU and hope the missing data appears.

Choose/create a **Runpod Network Volume** in a region with your intended GPU,
with capacity for the dataset, source checkout, and checkpoints. Storage is
separately billable. Copy the local `runs/dragon/dataset_hr/` directory into
`/workspace/anymove-data/dragon/dataset_hr/` on that volume. Use a Runpod
supported transfer route, e.g. a direct TCP SSH/SCP/rsync connection advertised
by your Pod, or Runpod's S3-compatible Network Volume transfer if available.
Do not assume a Git clone transfers the dataset: `runs/` is ignored.

Validate the uploaded copy **on the volume** before launching training:

```bash
python3 tools/runpod/preflight.py \
  --dataset /workspace/anymove-data/dragon/dataset_hr \
  --min-poses 2200
```

Use the regular POSIX-capable **Network Volume** for thousands of PNGs and
source checkout. Runpod **Global Volumes** are a different object-backed
storage product, and should not be used as the live dataset or Git checkout.
A Global Volume can optionally hold immutable completed full-state backups
through `MOCAP_RESUME_DIR`; see `docs/TRAIN_RESUME.md`.

The tracked `assets/dragon/rigged/rig_info.json` and
`assets/dragon/creature.json` come from the Git checkout. The Blender
`rigged.blend` file is not required for Reader training on already-rendered
PNG data, but would be required for new dataset generation.

## 3. Review rate, budget and provision one Pod

Run `runpodctl gpu list` and `runpodctl network-volume list` (or their MCP
equivalents); check **current GPU availability, price, VRAM, datacenter, volume
size, and account balance**. The following settings are EXAMPLES, not a
verified current price or an instruction to spend money:

```bash
export RUNPOD_GPU_ID="NVIDIA GeForce RTX 4090"
export RUNPOD_NETWORK_VOLUME_ID="<EXISTING_VOLUME_ID>"
export ANYMOVE_GPU_HOURLY_USD="<VERIFIED_RATE>"
export ANYMOVE_GPU_BUDGET_USD="<USER_APPROVED_MAX_GPU_ESTIMATE>"
export ANYMOVE_STOP_HOURS=4
export ANYMOVE_TERMINATE_HOURS=6
export ANYMOVE_APPROVE_SPEND=YES
bash tools/runpod/provision.sh
```

The script refuses to create a Pod unless all values are present, the
estimated GPU rate × stop-hours is within the supplied estimate, and the
stop duration is at most 12 hours. **This calculation excludes network
storage, container storage, taxes, and any other fees.** The service-side
stop and terminate flags, rather than the Codex process lifetime, enforce
the time guard. Record the new Pod ID and get SSH details from
`runpodctl pod get <POD_ID>` or the Runpod console.

The default image is Runpod's documented CUDA 12.8/PyTorch example; the job
installs the repository-pinned `torch==2.11.0` CUDA build and requirements
before training. If the selected Pod's GPU/driver/image is incompatible,
stop it rather than repeatedly attempting an expensive full run.

## 4. Run training inside the Pod

After connecting with SSH, check `/workspace` is the attached persistent
network-volume mount. Then:

```bash
git clone --single-branch --branch feature/runpod-managed-training \
  https://github.com/lukeyuRL/anymove.git /workspace/anymove-src
cd /workspace/anymove-src

export ANYMOVE_DATASET=/workspace/anymove-data/dragon/dataset_hr
export ANYMOVE_RUN_ID=dragon_quality_attempt_001
export ANYMOVE_EXPERIMENT_ID=dragon_quality_v1
export ANYMOVE_MODE=quality
export ANYMOVE_EPOCHS=20
export ANYMOVE_BATCH=2
export ANYMOVE_GRAD_ACCUM=1
export ANYMOVE_MIN_POSES=2200
export ANYMOVE_MAX_TRAIN_MINUTES=180
bash tools/runpod/train.sh
```

This runs the existing `mocap.train` model, including the new upstream
`EpochCheckpoints` implementation. Completed epochs produce both a normal
inference Reader checkpoint and an immutable full training-state backup.
The latter restores optimizer, scaler and optimizer step when a later Pod
restarts the **same experiment with exactly matching training settings**. `quality` enables the new synthetic quality heads;
`hr` trains the standard high-res Reader. Override input/heatmap, workers,
or gradient accumulation through `ANYMOVE_INPUT_SIZE`,
`ANYMOVE_HEATMAP_SIZE`, `ANYMOVE_WORKERS`, `ANYMOVE_GRAD_ACCUM`;
enable memory-saving checkpointing via
`ANYMOVE_GRADIENT_CHECKPOINTING=1`. Use a distinct `ANYMOVE_RUN_ID` for
each Pod attempt. Keep `ANYMOVE_EXPERIMENT_ID` unchanged when resuming a
training experiment, but choose a NEW `ANYMOVE_RUN_ID` per attempt.
Never run two writers against the same experiment directory.
**Existing run directories are never overwritten.**

### Monitoring, completion, and GPU shutdown

In another Pod shell, poll:

```bash
cd /workspace/anymove-src
python3 tools/runpod/status.py /workspace/anymove-results/dragon_quality_attempt_001
tail -n 30 /workspace/anymove-results/dragon_quality_attempt_001/train.log
```

The status reader reports the latest optimizer step, loss, LR factor, peak
allocated VRAM, elapsed time, last held-out Front/Side pixel errors, and
whether a checkpoint exists. `succeeded` is reported only when the training
process exits successfully. A checkpoint's presence alone is **not**
proof that the final epoch completed.

The Codex controller should poll reasonably (for example, every 5–10 min),
check for `succeeded`, `failed`, or `timed_out`, retain the resulting
logs/checkpoint on the network volume, and **immediately STOP the Pod** using
Runpod MCP `pod-action` with action `stop`, or
`runpodctl pod stop <POD_ID>`. Do not terminate the network volume.
If immediate shutdown is missed, the independent TTL stops the Pod.
The `--terminate-after` setting subsequently removes the Pod itself,
not the separately allocated network volume.

## 5. Acceptance gates / known limitations

- Run the offline tests and `bash -n`; these require no Runpod account or GPU.
- Test a ~30-pose smoke dataset **on the target GPU** before renting for
  the full 2,200 poses. Set `ANYMOVE_MIN_POSES=25`, a new run ID and
  a short time cap with a DIFFERENT `ANYMOVE_EXPERIMENT_ID`. Check the
  checkpoint is loadable via `torch.load` and that a new Pod can resume the
  completed epoch using its saved full-state files.
- Confirm the volume survives a Pod stop/replacement and that Codex MCP can
  list, inspect, and stop the actual Pod.
- The latest `iClone_High_Res` trainer already implements opt-in full-state
  epoch checkpoint/resume. This workflow enables it by default with
  `MOCAP_RESUME_DIR` on persistent storage. It restores the model,
  optimizer, AMP scaler, and optimizer step at an epoch boundary; a partially
  completed epoch repeats. Per-sample augmentation is not bitwise reproducible.
  A different dataset fingerprint, architecture, mode, batch or epoch schedule
  fails resume rather than silently continuing. Read `docs/TRAIN_RESUME.md`.
  Keep the source tree and metadata stable between attempts.
- The synthetic quality head does **not** imply verified real visibility.
  After training, use the established held-out data and known 3D animation
  tests before activating any quality-driven Solver weighting.
- Neither MCP nor the current scripts automatically upload data from Colab
  or `F:/anymove`; the original files must be staged to the network volume.
- **Real GPU provisioning, billing verification, SSH transfer and training have
  not been tested by this commit.** The controls are intentionally fail-closed.

## Codex task description (copy into an authorized Codex session)

> Work on `lukeyuRL/anymove` branch `feature/runpod-managed-training`.
> Read `docs/RUNPOD_TRAINING.md`. Use Runpod MCP for discovery, current
> price/capacity, Pod status, and shutdown. Do not create a paid Pod until I
> have supplied an exact GPU budget, an approved GPU/region, and an existing
> network volume. Before training, verify all 2,200 Dragon poses and both
> view PNGs exist on persistent `/workspace` storage. Then perform a short
> CUDA smoke run, train the `quality` Reader with an independent run ID,
> report the latest loss, validation error, VRAM and checkpoint status every
> 5–10 minutes, and stop the Pod immediately on success or failure. Keep the
> hard server-side stop/terminate TTLs active and never place my API key inside
> the Pod or Git. Do not enable the quality head in the production Solver
> without separate validation.
