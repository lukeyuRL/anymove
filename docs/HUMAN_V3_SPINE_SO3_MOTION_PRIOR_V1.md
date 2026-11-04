# AnyMove Human V3 — SO(3) Spine / Root Motion Prior V1

**Project:** AnyMove/iClone — Human V3 Nose 55-bone, 67-marker synthetic humanoid
**Branch:** `feature/iclone-general-camera-angles`
**Date:** 2026-11-04
**Status:** Dataset release passed, experimental Root/Spine solver and Blender QA passed functionally; **Motion-quality acceptance NOT passed**.

## Architecture and explicit Root ownership

The unique scene Root bone `RL_BoneRoot` is a **static skeleton node**, not an animated control:

- Local Euler and pose basis remain exactly zero/identity; no animation F-curves for `RL_BoneRoot`.
- World root translation and orientation are **separate** `trans` and `root_quat` channels; when rendered in Blender, these drive the **Armature Object**, not the bone.
- `root_quat` in `mocap.solve` is relative to the Root Bone rest frame. Blender object world orientation is obtained by conjugation with the root bone rest basis; never copy root_quat directly to a Bone rotation.
- `CC_Base_Pelvis`, `CC_Base_Waist`, Spine01/02 and other bones keep their own meaningful local rotations.
- This contract has been tested in both older Finger Roll / Kick & Wave Blender scenes and a **107-frame genuine 55-degree Pivot Turn**: static root matrix difference exactly 0, no RL_BoneRoot F-curves; per-world-point Blender/Torch FK discrepancy less than 0.01 cm.

## A. Dataset V1 — Continuous rotation ground truth (complete)

Input model `runs/human/human_skinmax_multicolor_v3_nose.blend`, pinned SHA256
`677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc`.

Output (local, untracked, large files):

```text
F:\anymove\runs\human\fullbody_finger_v2_hr_315_45\spine_motion_prior_v1\
```

- 12 distinct action families: pivot turns, walking turns, waist twists, lateral bends, counter-rotation, squat+turn, balance turn, lean/recovery.
- 144 clips × 107 **actual 24 FPS frames** = **15,408 frames**.
- Strict independent split: **96 Train (10,272 frames), 24 Validation (2,568 frames), 24 Test (2,568 frames)**.
- Each frame: 55×3 local Euler, world motion `root_rotation_wxyz`, object `root_translation_m`, 67 world joint XYZ, root-static convention and a **nonphysical foot-contact proxy**.
- Root turning ranges approximately **9.46–71.5 degrees** across the clips; unlike the earlier Reader dataset, Root rotations are not always zero.
- All 144 source clips' `RL_BoneRoot` local Euler components are exactly **zero**, while global root rotation varies in all clips.
- Independent `tools/audit_human_v3_spine_motion_gt.py` release audit: **PASS**, verifies shape, pinning, split, 24FPS, 67 markers and static Root.
- The 5,376 old Reader training images/checkpoints are unchanged and unrelated to this new motion dataset. This V1 is **not** a neural trained Motion Prior.

## B. Train-derived empirical rotation prior

Files:
- `mocap/so3_motion_prior.py`
- `mocap/human_root_initializer.py`
- `tools/calibrate_human_v3_so3_motion_prior.py`
- `tools/build_human_v3_so3_root_research_rig.py`
- `tools/build_human_v3_so3_root_evidence_variants.py`
- `tools/build_human_v3_train_neutral_spine_init.py`

Calibrate 9 spine/pelvis/neck/clavicle bones **from the 96 Train clips only**, using SO(3) relative rotations and rotational acceleration, including Root angular speed and angular acceleration. Include a train-derived local angle activity range, not fixed zero-spine constraints. The Root may turn substantially; there is no absolute zero-heading prior.

The four-anatomical-point Root initializer aligns visible Hip, Waist and L/R Thigh heads with rest using weighted Kabsch and maps world orientation into the root rest local basis. It was independently evaluated on six 107-frame Validation clips with fixed 1px projected observation noise:

| Initialization | Mean global Root orientation error |
|---|---:|
| Legacy zero-orientation fallback | 15.05° |
| Four-point 3D Kabsch | **2.29°** |

The optional SO(3) residual term allows observation-based Root heading to remain a constraint during optimization without freezing the character to face the original direction.

One additional research variant initializes the 9 spine/neck/clavicle local Euler channels from **Train-only median rotations** rather than independent framewise analytic IK, because unobservable local twists cause high-frequency pose allocations.

## C. Pivot Turn Validation ablation — NOT a fresh Test

This result is from **one already-inspected Validation clip**, 107 frames at 24FPS, 315°/45° orthographic projections with deterministic 1px synthetic noise. These numbers must not be advertised as Reader performance on video or independent final acceptance.

| Metric | Baseline 200 iterations | Root+SO(3) 600 iterations | Root+SO(3)+Train neutral init 600 |
|---|---:|---:|---:|
| Whole-body 3D MPJPE | **0.643 cm** | 0.651 cm | 0.719 cm |
| Pelvis mean angular acceleration (°/frame²) | 28.36 | 17.20 | **2.11** |
| Waist mean angular acceleration (°/frame²) | 38.37 | 16.95 | **8.37** |
| Root orientation mean error | 19.72° | **2.09°** | 2.28° |
| Blender GT mesh silhouette IoU | Not rendered in this comparison | **91.49%** | 91.19% |

**Meaning:** Root orientation recovers well; median Spine initialization provides substantially less rotational jerk but slightly worse 3D position and mesh overlap. The remaining waist angular acceleration is still significantly larger than synthetic GT. 600-iteration Root+SO3 is not a production-accepted motion solver, and neutral init is **not automatically promoted**.

The original `RL_BoneRoot` constraint remains enforced in **all** candidates. Neither their animation nor the GT touches the Blender source asset.

## D. Inspect in Blender 5.0

All local artifacts:

```text
F:\anymove\runs\human\fullbody_finger_v2_hr_315_45\solver\spine_so3_train_neutral_init_research\blender_qa\
```

- `spine_validation_pivot_turn_left_s405752_gt_rl_boneroot_static.blend`
- `spine_validation_pivot_turn_left_s405752_root_so3_rl_boneroot_static.blend`
- `spine_validation_pivot_turn_left_s405752_solver_rl_boneroot_static.blend`
- `spine_validation_pivot_turn_left_s405752_GT_RootSO3_TrainNeutral_24fps.mp4` — 107-frame, six-panel synchronized 315°/45° GT / Root+SO3 / Train-neutral comparison.
- `spine_validation_pivot_turn_left_s405752_spine_threeway_contact.jpg` — 7 sampled poses.
- `spine_validation_pivot_turn_left_s405752_visual_ab.json` — image silhouette overlap.

All three Blender .blend files were saved independently and replayed their original FK channels to within **0.01 cm**; Root bone never receives keyed pose channels.

## E. Limitations and next gate

- The 144-clip source is one **stylized procedural human rig**, not natural MoCap. It does not reliably solve real foot-contact or physical locomotion.
- A low 3D marker MPJPE alone does **not** establish low angular jerk, plausible spine rotation or reliable iClone FK transfer.
- Optimize a joint objective based on 3D position, Root geometric orientation, SO(3) rotational derivatives, biomechanical constraints and, eventually, grounded foot contact; do not just hard-limit spine X/Z or low-pass the entire sequence.
- Tune experimental gains on diverse **Validation** action families, not the Test clips. The **24 Test clips remain unused** during calibration and variant comparison. Freeze configuration before a future unbiased Test run.
- Include natural motion data and camera-specific visibility uncertainty before any product-default promotion. Do not map Blender-local Euler directly to iClone FK channels.

**Reproduction**: Generate audited GT with `tools/render_human_v3_spine_motion_gt.py`; audit with `tools/audit_human_v3_spine_motion_gt.py`; calibrate SO(3) with `tools/calibrate_human_v3_so3_motion_prior.py`; run controlled Validation with `tools/evaluate_human_v3_so3_prior_validation.py`. The old Reader and original V3 Nose Blender asset must remain untouched.
