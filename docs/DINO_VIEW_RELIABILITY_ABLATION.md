# Dino Joint-specific Two-view Reliability — Verified A/B

Branch: feature/dino-motion-quality-p123
Date: 2026-11-01
Status: experiment complete; **all three fixed-downweight profiles REJECTED**.
The production / original Dino Reader, camera layout, imported 3D Motion JSON,
existing P1/P2 checkpoint and iClone adapter were NOT modified.

## Hypothesis and exact control

The same frozen 672x672 / heatmap 192 baseline Reader had large,
anatomically localized **Blender truth 2D localization** errors:

| Camera, joint group | Seed 7 (24f) | Seed 11 (124f) |
|---|---:|---:|
| Front, Tail (mean 2D px) | 10.026 | 7.577 |
| Side, Tail | 0.730 | 0.839 |
| Front, Feet/Toes | 0.851 | 0.933 |
| Side, Feet/Toes | 1.459 | 1.562 |

The experiment asked whether the original two-view 3D solver could
be improved by fixed per-view, per-group confidence multipliers.

The three profiles were **preregistered before any 3D GT result was inspected**:

1. tail_front_half: anatomical Tail (9 keypoints) Front weight 0.5.
2. toes_side_half: Feet/Toes (34 keypoints) Side weight 0.5.
3. combined_half: both at the same 0.5 factors.

All remaining observations have multiplier 1.0. These arrays use the
existing optional front_track_weight / side_track_weight NPZ keys.
They are NOT optical-flow tracking, are NOT an anatomical mesh visibility
head, and are NOT P2 expected-error scores. Multipliers are applied to
bounded original Reader peaks *before* existing fit; the 3D Solver's
reprojection loss squares the resulting confidence, so 0.5 is approximately
0.25x its original data-term coefficient before robustification.

No source UV, original confidence, FPS, joint identity/order, affine
camera calibration, Blender GT or solver code is changed.
Each variant is SHA-256-bound to its exact source detection.

## Executed environment and provenance

Authorized remote Windows computer: LukeYu-2024.
Isolated checkout: F:\anymove_gt_eval_20261101
Original developer/training checkout: F:\anymove (unchanged).
Training status: stopped; completed epoch_013_full.pt retained.
Existing frozen model and both Blender GT sequences are reused as-is.

Actual execution:

    cd /f/anymove_gt_eval_20261101
    git pull --ff-only
    export PYTHON=python.exe
    export MOCAP_CREATURE=assets/dino
    bash tools/eval_dino_view_reliability.sh runs/dino/gt_experiments/seed7_smoke_24
    bash tools/eval_dino_view_reliability.sh runs/dino/gt_experiments/seed11_full_124

The runner rejects overwrite, P1/P2 stacking, unknown Dino rig names or
92-keypoint mismatch, wrong detection shape, and invalid calibration.
The baseline and all 3 candidates are evaluated against the SAME
independently Blender-evaluated 3D joints, using the same original
front/side cameras and Solver (--no-align). The camera layout had already
passed its independent geometry audit to within 0.00006px.

## Seed 7 · 24 frames — Results

3D errors are centimetres measured against independently evaluated
Blender joint world positions; lower is better.

| Variant | Whole-body MAE | Whole-body P95 | Tail MAE | Tail P95 | Feet/Toes MAE | Tail08 Max |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **3.35111** | 9.37482 | **6.41011** | 16.86528 | **3.10201** | **16.98437** |
| tail_front_half | 3.37002 | 9.32037 | 6.69938 | 21.49399 | 3.10125 | 27.94615 |
| toes_side_half | 3.50492 | 9.72327 | 6.41531 | 16.84600 | 3.52373 | 16.99745 |
| combined_half | 3.52140 | 9.74951 | 6.68350 | 21.44874 | 3.52329 | 27.95558 |

All 3 variants worsen whole-body mean error. Toe Side downweight directly
worsens toes; Tail Front downweight directly worsens Tail and Tail08 worst
case, even though whole-body P95 decreases slightly in isolation.

## Seed 11 · 124 frames — Independent confirmation

Same fixed weights, not tuned after Seed 7.

| Variant | Whole-body MAE | Whole-body P95 | Tail MAE | Tail P95 | Feet/Toes MAE | Tail08 Max |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **3.52860** | **8.67428** | **6.53030** | **20.42413** | **3.39332** | **69.79396** |
| tail_front_half | 4.23231 | 10.30244 | 13.68257 | 57.46571 | 3.39316 | 168.14974 |
| toes_side_half | 4.32314 | 11.50290 | 6.52413 | 20.39865 | 5.52394 | 69.67222 |
| combined_half | 5.02743 | 12.89187 | 13.67602 | 57.10969 | 5.52555 | 167.23569 |

The combined case **increases** whole-body mean error approximately
42.5% relative to baseline. Tail Front alone worsens Tail MAE roughly
110%; Toe Side alone worsens Foot/Toe MAE roughly 63%.

In the Tail Front-only case, Tail08 frame 47 GT 3D error went
69.794 -> 112.292 cm; maximum in frame 60 went to **168.150 cm**.
**50 out of 124 frames** have Tail08 GT error more than 20cm worse
than baseline, so this is not merely a single-frame anomaly.

Whole-body 3D velocity error (cm/s): Seed 11 baseline 37.01431,
tail_front_half 43.28553, toes_side_half 46.04695, combined_half 52.32352.
The framewise quality also degrades.

## Interpretation — what is proven, and what is not

**Verified:**
- Both tested seeds exhibit higher whole-body GT MAE for each fixed 0.5
  anatomical view-weight profile.
- Tail Front and Toe Side view weights are not expendable: despite higher
  2D localization residuals they supply distinct geometric information.
- Front supplies horizontal X; Side supplies depth Y; each shares
  vertical Z. Front Tail 2D error occurs in BOTH axes on the audited
  synthetic sequences: Seed 7 mean absolute U/V 6.517/6.392px;
  Seed 11 4.492/5.140px. Side Toe U/V errors are similarly both present.
- Current Solver already downweights Front vertical observations globally
  via front_height_weight=0.25 and applies a robust logarithmic pixel
  residual. Repeating a broad confidence cut reduces even the essential
  axis, potentially allowing geometric / temporal priors to dominate.

**Not demonstrated:**
- That all learned or frame-adaptive view-reliability strategies fail.
- That a third view guarantees improvement on AI-generated multi-camera
  video; this requires actually synchronized camera inputs.
- That improved 2D labels directly guarantee improved 3D Solver output.
- Any natural gait ground-truth accuracy: these are Blender-generated
  procedural motions, not validated walking or AI animation.

## Engineering decision and next experiment

**Reject all three fixed anatomy-only downweight profiles.**
Do NOT activate them in the production AnyMove Solver. Leave Baseline
as default. Preserve the separate NPZs, Motion JSONs and GT audit files
so future code changes can reproduce the exact failing cases.

Priority should shift toward **recovering the missing measurement**:
1. Independently evaluate a Front Tail Joint Reader correction trained on
   true Blender 2D labels and realistic body/self-occlusion conditions,
   without degrading Head/Toes and without overriding the base Reader.
2. Independently construct a synchronized orthographic Top / elevated
   camera for the SAME Blender truth frames and test geometry with
   perfect ground-truth 2D observations as an **oracle upper bound**.
   Oracle coordinates are NOT deployable predictions; do not claim a
   trained three-view Reader exists merely because that upper bound helps.
3. Compare both interventions on all 92 joints, including Root trajectory,
   Tail08 frames 47 and 60, Toe landing, group P95 and high-tail errors.
4. Only then evaluate calibrated per-frame uncertainty or a motion prior;
   require real mesh visibility and independent natural walking sequences.

No P1/P2 network training should be resumed merely to rescue this
fixed-weight experiment. The working original remains untouched.

## Local evidence (not uploaded to GitHub)

In the separate authorized local checkout:

Seed 7:
    runs/dino/gt_experiments/seed7_smoke_24/view_reliability_half_v1/

Seed 11:
    runs/dino/gt_experiments/seed11_full_124/view_reliability_half_v1/

Each contains:
- det_tail_front_half.npz, det_toes_side_half.npz, det_combined_half.npz
- Matching *.view_reliability.json SHA/provenance sidecars
- Matching motion_*.json, each with source_detections_sha256
- full_body_quality/full_body_report.json
- full_body_quality/by_joint.csv
- full_body_quality/by_frame.csv

All tests, profiles and artifacts are opt-in; the authoritative Dino
Reader, train data, rig, video source and iClone import were not altered.
