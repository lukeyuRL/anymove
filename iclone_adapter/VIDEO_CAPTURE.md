# Video / Detections to Dragon motion.json — external Python 3.11 / 3.14

This is the missing upstream piece before iClone v0.5.0 direct import.
The original AnyMove methods remain unchanged:

- Existing two-camera MP4: mocap.detect using the Dragon-specific DINOv2
  trained reader.pt + ref.png -> det.npz
- Existing det.npz: skip DINOv2 and go straight to mocap.solve
- mocap.solve: two-view sync, triangulation, rig-constrained fitting, Root
  Translation, Root Rotation, and 41 nonroot Euler XYZ tracks -> motion.json

Use Python 3.11 or 3.14 and the repo's requirements. For MP4 source this means
Torch + CUDA GPU, OpenCV, transformers, and the original trained Reader;
no heavy dependencies are imported into iClone's Python 3.8 host.

## One-time Dragon prerequisite

- assets/dragon/rigged/rig_info.json (checked into repo)
- runs/dragon/reader.pt (download from release examples-v1, or train using
  tools/dataset.sh dragon and tools/train.sh dragon)
- runs/dragon/ref/ref.png (release examples-v1 or tools/rig.sh dragon)
- original two-camera Dragon MP4, with front on left and side on right.
  An ordinary single-view video cannot be interpreted by this reader.

The example release includes the Dragon reader and sample clip; see
tools/examples.txt for SHA-256 entries. Installation on Windows may use
Git Bash tools/fetch_examples.sh dragon --readers --clips, but the
reader and video artifacts must be independently present before capture.

## External CLI

Start in F:\anymove (or your actual clone root) with Python 3.11 or 3.14.

    .venv\Scripts\python.exe -m iclone_adapter.video_capture ^
        --input F:\my_dragon_video.mp4 ^
        --reader runs\dragon\reader.pt ^
        --reference runs\dragon\ref\ref.png ^
        --output F:\anymove_test\capture_001

Or run a solve without GPU inference using the supplied example:

    .venv\Scripts\python.exe -m iclone_adapter.video_capture ^
        --input examples\dragon_roar\det.npz ^
        --output F:\anymove_test\capture_example

The output directory must not already exist. A capture_receipt.json
records source hash and status; det.npz, optional overlay.mp4 and
motion.json remain available for inspection. The worker rejects invalid
joint layout/order, nonfinite joint/confidence values, non-24-FPS
videos, incomplete Motion JSON, and severe quality gate problems.

Quality scores follow AnyMove's guidance: paw sync below 0.5 or large
reprojection error blocks automatic iClone import; manually inspect
the original video and reroll or explicitly use --allow-low-quality if
you accept the risk. No silent FPS resampling or auto-retargeting occurs.

This is one part of the existing AnyMove pipeline. Model generation,
rig preparation and Reader training are separate ONE-TIME tasks.
Prompt -> AI video generation is provided by mocap.generate / fal
but may incur remote API charges and is never silently launched.


## v0.6.0 — iClone UI end-to-end from MP4 / det.npz

The FIRST button of the panel is now:

CAPTURE Dragon MP4 / det.npz -> motion.json -> iClone + iProject

On click, it:
1. Verifies a clean selected original Dragon FBX at 24 FPS and Frame 0,
   before running expensive inference. Unknown rigs and already-keyed
   Dragon projects are rejected without modification.
2. Lets the user pick an already existing two-camera Dragon MP4 (front
   left and side right) or a detection NPZ.
3. For MP4, resolves the Dragon-trained reader.pt and ref.png (prompts
   for file paths if absent); for NPZ, reader and reference are not used.
4. Finds the repo external .venv/Scripts/python.exe or asks for an
   installed Python 3.11 executable. NEVER runs Torch inside iClone.
5. Chooses an existing output parent and creates a unique capture
   folder without overwriting. Starts a Qt QProcess in external Python
   3.11: mocap.detect (MP4), then mocap.solve, with stdout streamed
   to the iClone diagnostics panel. The iClone UI remains responsive.
6. If the Motion Solver fails or quality gates are not met, stops the
   import; preserves capture_receipt.json, det.npz, overlay.mp4 if
   available, and motion.json if generated.
7. Rechecks the iClone scene when Motion is ready; asks for another
   explicit confirmation before writing. On consent, calls the already
   tested direct_import pipeline: bone validation, per-frame Key write,
   all-frame controller readback, SHA-256 receipt and iProject SaveProject.

The button does NOT call fal.ai or charge for generating video, nor does
it retrain the Dragon DINOv2 Reader. Video source and trained Reader are
user-provided prerequisites; one-time Rig and Reader preparation are
documented in docs/PIPELINE.md.

Check the latest tool version v0.6.0 in the iClone window. This new
QProcess/UI orchestration has offline mock tests, but its GPU inference
and in-app Qt event integration still require one actual Windows/iClone
8 run before claiming operational acceptance.

## v0.7.5 — Python 3.14 worker support

The embedded iClone interpreter stays separate. The external workers
accept Python 3.11 or Python 3.14. The v0.6.0 step-by-step description
above is historical: target Dragon validation is now done immediately
before scene mutation, not before video generation or pose solving.

Check `.venv\Scripts\python.exe --version` on Windows. For Dragon
video detection, confirm the NVIDIA GPU is usable by PyTorch before
making a billable video request. Offline tests cover the version gate;
they do not exercise the physical Windows GPU, DINOv2 weights or fal.ai.
