# AnyMove Motion Quality Roadmap — keep the Video → 3D Solver backbone

Status: P1 Temporal Joint Tracking implemented opt-in; P2 and P3 require
quality-supervised training data and verified clean motion sequences.

Do not switch to 3D-first or N-view in this workstream. Preserve
reader.pt, reader_hr.pt, two-view source-video layout, 50 Dragon joints,
camera layout guard, motion.json FK/Root and the existing iClone adapter.

## P1 — Reader + optical-flow Temporal Joint Tracking

The DINOv2 Reader currently independently detects every joint every frame.
The new **pyramidal Lucas-Kanade + forward/backward check** tracks the
image pixels between frames and fuses with Reader locations conservatively.
It does *not* train or deploy CoTracker3. It is not a semantic bone tracker.
It explicitly abstains from changing the Reader when flow is unreliable.
Consecutive flow-only extrapolation is limited to three frames; no hallucinated
long occlusion trajectory. Raw Reader peaks are retained and optional bounded
track-weight multipliers influence the Solver only for the NEW tracked NPZ.
Detection files are immutable inputs and written under distinct names.

From Windows Git Bash at F:/anymove:

```bash
git pull --ff-only
export MOCAP_CREATURE=assets/dragon

python.exe -m mocap.track \
  F:/anymove_test/dragon/01_clip/capture_dragon_ai_prompt_001/generated/h3max.mp4 \
  runs/dragon/clips/hr_roar/det.npz \
  runs/dragon/clips/hr_roar/det_tracked.npz

python.exe -m mocap.solve \
  runs/dragon/clips/hr_roar/det_tracked.npz \
  assets/dragon/rigged/rig_info.json \
  runs/dragon/clips/hr_roar/motion_tracked.json

python.exe tools/compare_visibility.py \
  runs/dragon/clips/hr_roar/motion.json \
  runs/dragon/clips/hr_roar/motion_tracked.json
```

Outputs: `det_tracked.npz`, `det_tracked.tracking.json` and
`motion_tracked.json`. The NPZ includes `front_uv_reader`,
`side_uv_reader` (unaltered source detections), `front_track_weight`
and `side_track_weight` in addition to the original Solver keys.
The old `det.npz` and `motion.json` stay unchanged.

Or to run a complete separate capture with a NEW name:

```bash
MOCAP_TEMPORAL_TRACKING=1 bash tools/capture.sh dragon \
  F:/anymove_test/dragon/01_clip/capture_dragon_ai_prompt_001/generated/h3max.mp4 \
  hr_roar_tracked
```

This full capture preserves original Reader detections as `det_reader.npz`.
The normal `overlay.mp4` still shows raw Reader joint positions, not the
post-tracking positions. Use the JSON/NPZ to compare.

Keep these QA checkpoints:
- false identifications on the back paws (not merely mean 2D pixel error)
- joint velocity spikes, occlusion reappearance, side/front timing
- full 3D hind-paw error on known Blender sequences
- Root trajectory, foot sliding and iClone scene import

## P2 — Visibility and uncertainty supervision

Uncalibrated heatmap peak amplitude is NOT `P(visible)`. Future Reader
quality output must have separate supervised visibility and localization
uncertainty heads, versioned checkpoints, held-out calibration and per-view
metrics. A cutout-augmented synthetic training target is only *synthetic
mask visibility*, NOT Blender true self-occlusion. For true visibility,
measure whether the surface corresponding to a bone is externally visible
under the camera, rather than raycasting naively from the camera to an
internal skeleton joint. Never claim that an occluded internal joint is
physically visible merely because its image projection is inside the mesh.

## P3 — Motion Prior / Motion Denoising

The current 2,200 independent Blender Pose examples are not Motion
Sequence training data. A learned motion prior must use genuine temporal
animation sequences with known skeleton, matching FPS, and verified
provenance. Do not train from the same unsupervised solver output and
then evaluate against it; that can teach the model the original mistakes.
A first opt-in denoiser can test temporal regularization under strict
joint-angle and Root invariants, but it must not be called a *learned*
motion prior before a checkpoint is trained from independent verified
motion sequences.

All experiments are opt-in until same-clip A/B and known-ground-truth
motion comparisons demonstrate quality gains.


### P2 implementation: experimental learned Reader quality head

A **new, opt-in checkpoint type** adds two per-joint prediction maps to the
DINOv2 Reader: a synthetic-cutout visibility logit and a log predicted
localization-error scale. Existing 448 and 672 readers load with the exact
original state_dict and are *not* modified.

The training augmentation masks random rectangles near labeled 2D joints,
and supplies supervised binary labels based on which keypoints that patch
covers. Localization-error supervision comes from the Reader's detached
actual 2D heatmap decode residual against the synthetic ground-truth
coordinates. This *is* a learnable head, but is deliberately **NOT**
called real Blender mesh self-occlusion visibility, Bayesian uncertainty or
a calibrated probability. Real self-occlusion needs depth/surface-part
annotation and held-out measurement later.

Train a separate experimental quality Reader on the **existing** Dragon
high-res dataset (the train script injects its own synthetic occlusion):

```bash
git pull --ff-only
export PYTHON=python.exe
MOCAP_TRAIN_QUALITY=1 bash tools/train.sh dragon \
  20 2 672 192 runs/dragon/dataset_hr \
  runs/dragon/reader_hr_quality.pt 0
```

An RTX 3070 8GB can be checked with batch 2 as in the earlier reader
experiment. No VRAM estimate or live throughput is claimed for this new
head until it runs on the user's GPU.

Use `mocap.detect` with this checkpoint in a separate output NPZ. It
automatically discovers `quality_spec.version=1` in the checkpoint.
Extra optional outputs include
`front_synthetic_visibility_score` / `side_synthetic_visibility_score`
and `front_expected_error_px` / `side_expected_error_px`. Those scores
are **not** passed into the current Solver weights, and the old required
`front_uv`, `side_uv`, `front_conf`, `side_conf` remain unchanged.

```bash
export MOCAP_CREATURE=assets/dragon
python.exe -m mocap.detect \
  F:/anymove_test/dragon/01_clip/capture_dragon_ai_prompt_001/generated/h3max.mp4 \
  runs/dragon/reader_hr_quality.pt runs/dragon/ref/ref.png \
  runs/dragon/clips/hr_roar/det_quality.npz \
  runs/dragon/clips/hr_roar/overlay_quality.mp4
```

Do NOT treat `synthetic_visibility_score` as calibrated occlusion
probability. To promote this experiment, measure true known-mask AUC,
held-out pixel-error calibration (e.g. expected-vs-actual error bins),
and same-clip 3D foot motion with independent Ground Truth. A separate
Blender body-part visibility mask is still an open item.
