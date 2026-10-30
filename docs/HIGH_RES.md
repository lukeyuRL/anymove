# AnyMove high-resolution reader (iClone_High_Res)

## Scope and unchanged interfaces

This branch upgrades synthetic **two-view** (Front + Side) training. It does NOT
enable 4-view generation, change the 1280x720 camera layout, change the camera
calibration, change the solver's layout projection, or alter any existing
`det.npz`, `motion.json`, Root Flight, Blender or iClone motion contract.

The source rendering scale, model input size, and heatmap resolution are independent:

| | Legacy | High resolution |
|---|---:|---:|
| Dragon render, each view | 384x432 at scale .6 | 640x720 at scale 1.0 |
| Reader square input | 448x448 | 672x672 |
| Reader heatmap output | 128x128 | 192x192 |
| Dragon two-view video canvas | 1280x720 | 1280x720 (unchanged) |
| Model file | runs/dragon/reader.pt | runs/dragon/reader_hr.pt |
| Dataset directory | runs/dragon/dataset | runs/dragon/dataset_hr |

The DINOv2 ViT-B/14 with registers model in pinned transformers 5.20.2
interpolates positional embeddings internally when the input patch grid changes.
The input size must be divisible by 14. The model keeps the same learned head
parameter shapes; high-res checkpoints record `reader_spec` containing
`version`, `input_size`, and `heatmap_size`. A checkpoint without that field
is always treated as legacy 448/128. Do not infer the size from filename.

## Set up and generate data

Run commands from the repository root. Set `MOCAP_CREATURE` via the scripts
or explicitly for direct Python/Blender invocations.

```bash
# Full-resolution Dragon samples (2,200 3D poses, two RGBA images each).
tools/dataset.sh dragon 2200 3 500 1.0
# Review: runs/dragon/dataset_hr_samples.jpg
```

The dataset metadata still records 2D joint coordinates in *unscaled camera
layout pixels*. The reader multiplies these by the stored render scale to map
to the actual training image. A sample is only resumed if metadata matches the
requested scale and both view PNGs exist; mixed scales are rejected by training.
All output paths are resolved to absolute paths **before** opening the Blender rig,
and each image is checked before its JSON metadata is written. In Windows,
`F:/anymove/runs/dragon/dataset_hr_smoke` should contain 00000_front.png,
00000_side.png, and 00000_meta.json in the same directory. If an older version
accidentally saved PNGs under C:, update this branch and rerun the same
`tools/dataset.sh` command. Existing metadata with missing PNGs will be
re-rendered rather than skipped; do not manually move unrelated C: files.
The default rendering policy now produces approximately **80% unblurred**
poses and **20% light motion blur**, with a randomized shutter of
**0.2-0.4 frames** for the blurred samples. This is a provisional training
configuration, not a proven optimal setting. The Meta JSON includes
`blur`, `blur_shutter` (null for unblurred samples), `blur_profile`,
and `sample_seed`. The contact sheet displays `sharp` or the shutter value.
Each pose's random seed depends only on its index so resuming an interrupted
render or changing the Blender process count does not alter later poses.

When reusing the same dataset directory after this change, samples with
the *old* blur policy (which have no `blur_profile`) are **automatically
re-rendered**, even if both PNGs already exist. This applies to the existing
30-pose smoke dataset: pull the branch, then rerun exactly the same command.
It overwrites that dataset's old samples; use a new directory to retain an
old-policy dataset for a controlled comparison. The trainer rejects datasets
mixing different blur-profile versions.

The output directory is separate to avoid silently skipping old low-res images.
If the rig/joints change, delete or use a new dataset directory; checking
render scale does NOT detect a changed rig.

You may pass a larger render scale as argument five (e.g. `2.0`) **only
with a new output directory** as argument six. Higher source rendering resolution
does not make 672-input DINOv2 operate at that resolution: each view is still
letterboxed and sampled to 672x672.

## Train and compare like for like

```bash
# High-res reader on a GPU with enough VRAM:
tools/train.sh dragon 20 8 672 192

# A/B baseline on the SAME high-res synthetic poses, different output:
tools/train.sh dragon 20 8 448 128 runs/dragon/dataset_hr runs/dragon/reader_448_baseline.pt

# Test source file and model semantics:
python -m unittest discover -s tests -p 'test_iclone_*.py' -v
python -m compileall -q mocap blender tools iclone_adapter tests
```

### RTX 3070 8GB / Windows Git Bash procedure

The default batch 8 is inappropriate for 8GB VRAM. Use a **physical batch of
1** with gradient accumulation 8 and DINOv2 gradient checkpointing. The latter
trades additional compute for lower activation memory. The trainer forces
`sdpa` attention when supported, selects BF16 on compatible CUDA hardware,
or otherwise uses FP16 with GradScaler. Gradient accumulation is normalized
for the last short group, and the log prints allocated peak GPU memory every
50 optimizer updates. A physical batch 1 with gradient accumulation 8 has
approximately the same effective batch as 8, but is not bitwise identical.

If using Windows Git Bash, first set the Python and Blender paths to the
executables installed on your own computer. The checked-in shell helper
recognizes `PYTHON` and `BLENDER` environment variables. The project
requires `assets/dragon/rigged/rigged.blend`; it is not tracked in Git.

```bash
# In Git Bash from the checked-out iClone_High_Res repo:
export PYTHON="$(pwd)/.venv/Scripts/python.exe"
# Export BLENDER only if blender is not already on PATH:
# export BLENDER="/c/Program Files/Blender Foundation/<your-version>/blender.exe"
"$PYTHON" -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

# 30 sample smoke dataset. Blender must finish and close before training.
bash tools/dataset.sh dragon 30 1 500 1.0 runs/dragon/dataset_hr_smoke

# 1 epoch smoke check, checkpointing+accumulation enabled:
MOCAP_LOW_VRAM=1 bash tools/train.sh dragon 1 1 672 192 \
    runs/dragon/dataset_hr_smoke runs/dragon/reader_hr_smoke.pt 0

# Full dataset; do not reuse the smoke dataset output:
bash tools/dataset.sh dragon 2200 1 500 1.0

# 20-epoch complete train, with independent reader_hr.pt:
MOCAP_LOW_VRAM=1 bash tools/train.sh dragon 20 1 672 192 \
    runs/dragon/dataset_hr runs/dragon/reader_hr.pt 0
```

If the first 672 sample fails with `CUDA out of memory`, use the same
full-resolution source dataset with one of these progressively smaller
readers, each saved to its own checkpoint filename:

```bash
MOCAP_LOW_VRAM=1 bash tools/train.sh dragon 20 1 560 160 \
    runs/dragon/dataset_hr runs/dragon/reader_560.pt 0
MOCAP_LOW_VRAM=1 bash tools/train.sh dragon 20 1 448 128 \
    runs/dragon/dataset_hr runs/dragon/reader_448_fulldataset.pt 0
```

As a fallback only, `MOCAP_FREEZE_BACKBONE=1` freezes DINOv2 and trains
the heatmap head, reducing training memory substantially, but this changes
the learning problem and may reduce accuracy. It requires its own validation.

Do not confuse CPU RAM with GPU VRAM. `--workers 0` reduces Windows
DataLoader process overhead, not DINOv2 activation VRAM. CUDA OOM with
batch 1 after checkpointing requires lower resolution, freezing the
backbone, or a GPU with more VRAM. For inference at input >448, the default
`MOCAP_DETECT_BATCH` is 1 (set the environment variable to a higher value
only if there is sufficient VRAM). No measured RTX 3070 throughput or
memory result is implied by this source change.

Source image resolution alone does not improve correctness without new training.

The 4% held-out pose split is the same index-modulo-25 scheme as before.
Compare **held-out mean joint error in original camera layout pixels**, separately
for Front and Side, not pixels in the new 192x192 heatmap. Then compare 3D
joint error using `mocap.exam` and a fixed ground-truth animation.
A smaller 2D error is not by itself evidence of better iClone motion quality.
For a fair A/B, run multiple seeds and avoid selecting hyperparameters on the
held-out test data. The current randomly augmented train batches are not
bitwise deterministic.

## Detection and downstream

```bash
# Explicitly use high-res Reader. Output schema remains unchanged:
export MOCAP_CREATURE=assets/dragon
python -m mocap.detect VIDEO.mp4 runs/dragon/reader_hr.pt \
  runs/dragon/ref/ref.png runs/dragon/high_res_det.npz
python -m mocap.solve runs/dragon/high_res_det.npz \
  assets/dragon/rigged/rig_info.json runs/dragon/high_res_motion.json

# Existing capture script selects reader_hr.pt if present; otherwise reader.pt.
tools/capture.sh dragon VIDEO.mp4 high_res_check

# Or override the checkpoint:
MOCAP_READER_PT=runs/dragon/reader.pt \
  tools/capture.sh dragon VIDEO.mp4 legacy_check
```

The iClone Dragon video capture UI likewise prefers `reader_hr.pt` if it
exists, otherwise `reader.pt`, and lets the operator select a checkpoint.
The prompt-to-video worker still explicitly receives the selected Reader path.
The original iClone FK, motion writer and Root Flight logic are untouched.

Use the existing 1280x720 reference/clip for detection. A 4K input video
does not increase the effective camera layout resolution after calibration.
This branch has NOT switched the video generator or solver to N-view.

## Validation gates before treating this as production

1. Run the pure contract tests and Python compilation (CI).
2. Render a small Dragon test dataset with Blender. Inspect joint overlays,
   confirm both images and the scale/size metadata agree.
3. On the target CUDA system, train both 448/128 and 672/192 readers with a
   fixed pose dataset; capture validation errors, VRAM and runtime.
4. Evaluate identical known-ground-truth clips with each checkpoint. Check
   raw heatmap peaks, detector coordinates, 3D joint errors and sync scores.
5. Import one new high-res `motion.json` into a CLEAN Dragon iClone scene.
   Verify wing animation, Root Z flight direction, scale 1.0 and all-frame
   root translation. Do not assume old motion QA proves new reader quality.

No CUDA training, paid video generation or live iClone import is implied by
committing this source. GPU/Blender/iClone checks require the actual runtime.

## Experimental Dragon visibility-aware solver (opt-in)

The 672px Reader reduced synthetic validation error but did not uniformly
improve reprojection on the generated Dragon roar clip. Front hind paws are
often hidden behind the body/forelegs; their cross-view height correlations
were negative despite well-synchronized front paws. This is still a
hypothesis, **not verified occlusion ground truth**.

The optional solver computes a **reliability proxy** from already detected
Front/Side joint tracks. The only target joints are `knee_L/R`,
`hock_L/R`, `hpaw_L/R`, and `htoe_L/R`. It requires a discrepancy in
Front-versus-Side pixel heights above 6 px; a nearby same-side front-leg
landmark strengthens the penalty. Height discrepancy reaches maximum
strength at 24 px, and the risk is median-filtered over five frames. The
Front reliability multiplier remains between 0.35 and 1.0; **every Side
weight and all other Front joint weights remain unchanged**.

The multiplier affects the Front reprojection loss and the Front contribution
to height during initial triangulation. It is *not* a trained visibility
classifier, an occlusion probability, or proof that the Side estimate is
correct. Cases where both views are equally wrong cannot be detected from
this signal alone.

**The default solver remains unchanged.** For an A/B comparison on the same
already-produced 124-frame high-res `det.npz`, with no extra paid video
generation and without re-running DINOv2:

```bash
git pull --ff-only
export MOCAP_CREATURE=assets/dragon
export PYTHON="$(pwd)/.venv/Scripts/python.exe"

DET=runs/dragon/clips/hr_roar/det.npz
RIG=assets/dragon/rigged/rig_info.json
BASE=runs/dragon/clips/hr_roar/motion.json
VIS=runs/dragon/clips/hr_roar/motion_visibility.json

"$PYTHON" -m mocap.solve "$DET" "$RIG" "$VIS" --visibility-aware
"$PYTHON" tools/compare_visibility.py "$BASE" "$VIS"
```

A sidecar named `motion_visibility.visibility.json` is written next to
`motion_visibility.json`. It contains a profile ID, thresholds,
per-frame weights for eight hind joints, affected-frame summaries,
reprojection errors, and the detection SHA-256 for traceability. The
`motion.json` fields, bone order, 24-FPS root coordinates, iClone
Reader/Writer interfaces and original `det.npz` remain unchanged.

For an optional *full* capture with the same source video, use a NEW clip
name to preserve old results:

```bash
MOCAP_SOLVER_VISIBILITY=1 bash tools/capture.sh dragon \
  F:/anymove_test/dragon/01_clip/capture_dragon_ai_prompt_001/generated/h3max.mp4 \
  hr_roar_visibility
```

The iClone UI does not turn on this experimental mode automatically. The
existing Motion Import and Root Flight code are untouched.

**Important:** The paw-agreement score is computed from the reader's raw
Front/Side detections *before* weighting. It will remain around 0.36 for
the provided high-res roar clip and still fails the iClone quality threshold
of 0.5. Do not suppress that gate. Better or worse reprojection error also
does not establish 3D correctness; before adopting this solver, compare
hind paw trajectories, wing motion, root Z and scale in a clean iClone
scene, and run a Blender known-ground-truth 3D joint test.
