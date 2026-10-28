# Video / Detections to Dragon motion.json — external Python 3.11

This is the missing upstream piece before iClone v0.5.0 direct import.
The original AnyMove methods remain unchanged:

- Existing two-camera MP4: mocap.detect using the Dragon-specific DINOv2
  trained reader.pt + ref.png -> det.npz
- Existing det.npz: skip DINOv2 and go straight to mocap.solve
- mocap.solve: two-view sync, triangulation, rig-constrained fitting, Root
  Translation, Root Rotation, and 41 nonroot Euler XYZ tracks -> motion.json

Use Python 3.11 and the repo's requirements. For MP4 source this means
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

Start in F:\anymove (or your actual clone root) with Python 3.11.

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
