# AnyMove Human SkinMax V3 — Two-camera azimuth A/B (2026-11-03)

**Branch:** `feature/iclone-general-camera-angles`
**Blender model:** `F:\anymove\runs\human\human_skinmax_multicolor_v3.blend`
**Source SHA-256:** `93ced4096791fbc84712527a3c023e7c3bee381106e15b898cd4295f24c8008b`
**GT:** `runs/human/lora_synth_A/v6_reach_source`
**Status:** geometry/mesh-visibility/visual QA complete; **Reader and H3 inference NOT performed**.

## Model and animation provenance

Human SkinMax V3 contains 55 bones, 30,188 mesh vertices, and 53 distinct bone-associated vertex-color regions. The existing 20 V6 reach-motion sequences include `joints.json` and `skeleton_contract.json`. The latter records the **same source SHA-256 as the verified V3 Blender model**, so the GT comes from the exact model being tested. Replaying sample frames 0/20/53/80/106 in Blender 5 reproduces the reference joint positions to within **0.00028 cm** maximum.

This test reuses existing local V6 motion; no H3/AI video generation, cloud compute, or Reader retraining occurred.

## Configurations

- **A:** 0° / 90° — legacy Front/Side.
- **B:** 315° / 45° — two symmetric frontal oblique angles.
- **C:** 45° / 135° — front/rear oblique angles.

All are 90° apart and use a calibrated level orthographic projection. Same 384×448 px image slots, 2.4328022m vertical span and center Z=0.868858m.

## Test 1: Pure GT geometry (20 reach clips; 2,140 frames; 55 joints/frame)

Identical 2px standard-deviation Gaussian pixel noise in each configuration, with identical seed and noise indexed by frame/joint/view:

| Configuration | Horizontal condition | Noisy 3D MPJPE | P95 |
| --- | ---: | ---: | ---: |
| A 0°/90° | 1.0 | 1.577 cm | 2.808 cm |
| B 315°/45° | 1.0 | 1.577 cm | 2.808 cm |
| C 45°/135° | 1.0 | 1.577 cm | 2.808 cm |

With **noise-free** 2D observations all three recover 3D GT to floating-point precision. Each configuration keeps all GT joints within the frame in this controlled dataset. Geometry alone cannot choose a winner.

JSON: `runs/human/camera_angle_p0_v3/human_gt_triangulation.json`.

## Test 2: V3 mesh-surface visibility (4 reach clips × 4 sample frames = 16 poses)

Blender 5 raycasts against the *evaluated skinned V3 mesh*. For each of 53 dominant-color skin groups, up to 40 surface vertices are sampled with a fixed RNG seed. A color **region** counts as exposed to a camera if ≥8% of its sampled surface vertices have an unobstructed ray. This is a **color-region exposure proxy**: it is NOT true keypoint visibility probability or actual image-reader confidence.

| Configuration | Color regions exposed to both views | Finger regions exposed to both |
| --- | ---: | ---: |
| A 0°/90° | 31.5% | 28.5% |
| B 315°/45° | **64.7%** | **61.7%** |
| C 45°/135° | 58.4% | 55.6% |

Robustness rerun with **independent RNG seed and 80 surface samples per group**:

| Configuration | Both-view color-region exposure | Both-view finger-region exposure |
| --- | ---: | ---: |
| A 0°/90° | 35.7% | 31.9% |
| B 315°/45° | **67.6%** | **69.6%** |
| C 45°/135° | 64.2% | 60.8% |

The ranking of B over A survives the sampling change. Region exposure values depend on the visibility threshold, sample count, render geometry, and motions; they are not a calibrated measure of Reader accuracy.

Reports: `human_v3_surface_visibility.json` and `human_v3_surface_visibility_seed20261104_samples80.json` in `runs/human/camera_angle_p0_v3`.

## Test 3: Projected left/right corresponding-joint separation

23 matching L/R keypoint pairs across the same 20 clips. At an 8px proximity threshold, the fraction of corresponding L/R pairs that are closer than 8px **in at least one camera** is:

- A 0°/90°: **35.38%**
- B 315°/45°: **0%**
- C 45°/135°: **0%**

This is 2D **GT projection separation only**, not detected joint identity quality. It suggests fewer opportunities for near/far left-right overlap in B/C under these poses.

Report: `runs/human/camera_angle_p0_v3/human_v3_gt_left_right_separation.json`.

## Actual V3 Blender renders (no AI generation)

Eevee renders of the **same** V6 forward-left Reach pose (frame 53), using the V3 SkinMax color material, were produced for all six cameras.

Contact sheet: `F:\anymove\runs\human\camera_angle_p0_v3\human_v3_camera_angles_contact.png`.

The original V3 Blend and source motion files were NOT altered.

## Re-run the QA

From the repository root `F:\anymove`:

~~~powershell
python -m tools.eval_human_v3_camera_angles_gt
python -m tools.eval_human_v3_projected_ambiguity

& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b 'runs/human/human_skinmax_multicolor_v3.blend' -P 'tools/validate_human_motion_replay.py'

& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b 'runs/human/human_skinmax_multicolor_v3.blend' -P 'tools/eval_human_v3_camera_visibility_blender.py'

$env:ANYMOVE_VIS_SEED = '20261104'
$env:ANYMOVE_VIS_SAMPLES_PER_REGION = '80'
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b 'runs/human/human_skinmax_multicolor_v3.blend' -P 'tools/eval_human_v3_camera_visibility_blender.py'
Remove-Item Env:ANYMOVE_VIS_SEED,Env:ANYMOVE_VIS_SAMPLES_PER_REGION

& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b 'runs/human/human_skinmax_multicolor_v3.blend' -P 'tools/render_human_v3_camera_comparison.py'
python tools/make_human_v3_angle_contact_sheet.py
~~~

## Interpretation and next gate

The controlled V3 tests support prioritizing **315°/45° for actual Reader trials**, because it retains the best-case 90° triangulation geometry while exposing more of the colored fingers and corresponding L/R joints to both cameras for these arm-reaching motions.

**Do not declare 315°/45° globally best.** This is limited to frontal human Reach clips, a single rig/mesh, finite poses and the chosen surface-visibility proxy. A robust next stage is to render pose-matched multi-angle train/validation images using the V3 colors, train/adapt the Reader, evaluate held-out image keypoint error/occlusion/swap rates, then compare 3D MPJPE and motion jitter. That stage is **not yet done**.
