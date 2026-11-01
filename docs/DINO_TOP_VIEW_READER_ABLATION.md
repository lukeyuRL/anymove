# Dino synchronized Top View: actual Joint Reader vs Ground Truth Oracle

Branch: `feature/dino-motion-quality-p123`. All tests are opt-in and were
run in `F:\anymove_gt_eval_20261101` (an isolated clone).
**Do not promote to iClone/AnyMove defaults or claim synchronized Top is
available for existing two-view AI video.** No changes to the original
`F:\anymove` training datasets, checkpoints or `motion.json` contract.

## Why this experiment

Two independent Dino Blender GT videos (Seed 7 / 24 frames and Seed 11 /
124 frames) showed Front Tail mean 2D Reader error 10.03 and 7.58 layout
pixels, while Side Tail error was 0.73 and 0.84 pixels. Front carries the
unique X coordinate, so hard-masking its Tail observation lost the very
axis that needed recovery.

At Seed 11 Frame 47, Tail08 original 3D GT X error was -66.77 cm
(Front 2D Reader position error 71.55px). Soft P2 confidence weighting
increased its X error to -73.63cm. The problem is not necessarily
anatomical mesh occlusion: the virtual Tail08 bone tip is ~1.05m
outside the visible skin; Blender mesh raycast marks its skin-support
status Unknown.

**Hypothesis:** a synchronized independent Top camera can provide more
accurate X/Y observations. Verify whether real image-inferred Top
keypoints help, distinguishing them from perfect Blender GT keypoints.

## Exact Top camera contract

The same keyed Blender `gt_render/animation.blend` and Blender's
evaluated 92-joint world truth are used throughout. Top is a separate
880×720 orthographic view from +Z looking along -Z, rotated +90° so
the ~16m-long Dino body lies across the image width:

```text
top_u = 880/2 + world_Y / metres_per_pixel
top_v = 720/2 + world_X / metres_per_pixel
```

The camera JSON `cam_top.json` contains its true Blender R,t,
pixel scale and image resolution. Both actual Reader and Oracle use
the same checked matrix, never a fabricated unrelated third view.
Before rendering, the script verifies every one of the 92 GT keypoints
is in-frame in all T frames.

The original `Front+Side` Reader remains frozen, with
672×672 model input / 192×192 heatmaps. Original UV/peaks, FPS,
keypoint ordering and calibrated two-view video are preserved.
Only nine Tail keypoints can provide Top Solver constraints.
Top input confidence is the clipped raw Reader peak, **not a calibrated
visibility probability**.

## Two very different Top data sources

| Source | Coordinates | Diagnostic meaning |
|---|---|---|
| Blender Top GT Oracle | `top_uv` computed from **known world-joint truth** | Geometry+Solver potential upper bound; **leaked GT; never deploy** |
| Actual Top image Reader | `top_uv` returned by `mocap.detect.run` after reading rendered Top pixels | Real inference within controlled synthetic domain; no GT keypoint coordinates fed to Solver |

The Top Oracle requires explicit
`--experimental-dino-top --allow-gt-top-oracle --top-camera`.
The actual Top image Reader requires
`--experimental-dino-top --top-camera` and must **not** be labeled
`--allow-gt-top-oracle`.
Both are SHA-bound to the source camera/Reader data.
New Solver output stores separate source provenance and never
affects production default.

## Seed 7 / 24-frame results (completed and checked)

| GT measurement | Front+Side Baseline | Top actual image Reader (Tail-only) | Top GT Oracle (Tail-only) |
|---|---:|---:|---:|
| Full-body mean 3D error (cm) | **3.35111** | 3.15800 | 2.90948 |
| Full-body P95 error (cm) | 9.37482 | 8.03799 | 7.54015 |
| Tail mean 3D error (cm) | **6.41011** | 4.15283 | 2.19665 |
| Tail P95 3D error (cm) | 16.86528 | 10.21199 | 6.70882 |
| Root mean 3D error (cm) | 2.02491 | 1.94751 | 1.42162 |

The actual Top Reader improved average Tail position error by **35.2%**
and full-body mean by **5.8%** in this one synthetic motion.
However **Tail08 mean worsened** from baseline 9.54625cm to
9.87029cm; Oracle improved it to 5.9797cm.
Do not report that all joints improved.

### Top image 2D GT after freezing the Front/Side Reader

| Measurement | Seed 7 · 24 frames | Seed 11 · 124 frames |
|---|---:|---:|
| All 92 Top Reader mean 2D GT error | 32.661px | 33.442px |
| Tail 9 joints mean Top GT error | 5.378px | 5.319px |
| Tail 9 joints P95 Top GT error | 12.554px | 13.059px |
| Tail08 Top mean error | 11.504px | 11.566px |

The whole-body Top Reader is **not** ready for all-joint use.
The Tail-only transfer looks more promising but the Tail08 error
remains high, despite 124-frame Reader stability.
At Seed 11 Frame 47 Tail08 Top 2D GT error is 13.177px
versus Front 2D GT error ~71.55px.
The trained Front/Side model has **not** been fine-tuned on Top images.

## Current software entrypoints

`blender/render_dino_top.py` — synchronized Top image renderer from the
actual keyed Blender rig, strict GT/Camera/Rig/frame safety checks.

`mocap/dino_top_camera.py` — validated orthographic X/Y camera matrix,
NumPy and differentiable Torch projection.

`tools/eval_dino_top_reader.py` — run frozen Reader **on actual Top
RGBA image pixels** and compare predicted 2D coordinates with Blender
projected truth, never use GT coordinates as Reader outputs.

`tools/make_dino_top_reader_detection.py` — combine raw original
Front/Side NPZ with **actual** Top pixel detections, only Tail9
enabled and clipped peaks. Refuse source/rig/frame mismatch.
`tools/make_dino_top_oracle.py` is a separate, GT-leakage-only path.

`tools/eval_dino_top_reader_motion.sh` — actual Top Reader
to opt-in 3D Solver → full-body Blender GT A/B.
`tools/eval_dino_top_oracle.sh` does independent upper-bound only.

`tools/review_dino_top_reader.py` — per-frame QA MP4: green
image-Reader positions, red Blender-projection marks, differences in
pixels. GT used **only for visual evaluation**, never to solve.

## Reproduction / local assets

```bash
cd /f/anymove_gt_eval_20261101
export PYTHON=python.exe
export BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
export MOCAP_CREATURE=assets/dino

# The original 24-frame GT / Top images already exist.
bash tools/eval_dino_top_reader_motion.sh \
  runs/dino/gt_experiments/seed7_smoke_24 \
  runs/dino/gt_experiments/seed7_smoke_24/top_oracle_v2_rotated \
  runs/dino/gt_experiments/seed7_smoke_24/top_oracle_v2_rotated/reader_transfer_baseline \
  runs/dino/gt_experiments/seed7_smoke_24/top_oracle_v2_rotated/eval_ANOTHER_UNIQUE_NAME
```

Use new output paths only; commands **refuse overwrite**.
The complete GT and QA artifacts remain on the authorized user's
computer, not in the public GitHub repo:

- Seed 7 actual: `runs/dino/gt_experiments/seed7_smoke_24/top_oracle_v2_rotated/eval_image_reader/full_body_quality/`
- Seed 7 Oracle: `runs/dino/gt_experiments/seed7_smoke_24/top_oracle_v2_rotated/eval_tail_only/full_body_quality/`
- Seed 11 Top rendering and actual 2D check: `runs/dino/gt_experiments/seed11_full_124/top_view_v2_rotated/`

## Go/no-go principles

- Confirm actual Top Reader 3D gains on the independent 124-frame
  sequence, especially Tail08, not only 24-frame averages.
- Evaluate all per-joint and per-frame error (MAE, P95, max), Root and
  non-tail regressions using *identical Blender 3D truth*.
- Confirm real AI video can provide a physically consistent,
  synchronized third view. Blender GT image rerender is not the same
  as multi-view AI video inference.
- Before any production integration, train/evaluate a dedicated Top
  Reader with pose-matched Top samples and separately held-out
  continuous sequences. Reuse Dino 92-joint contract and preserve
  Front/Side Reader as a fallback.
