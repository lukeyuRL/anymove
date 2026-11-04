# Human V3 Nose — Multi-family Spine/Root Validation Gate

**Scope:** Only synthetic, audited, clip-disjoint Validation (GT created from a single stylized Blender V3 Nose), 107 source frames at 24 FPS, calibrated 315°/45° cameras, identical deterministic 1px synthetic 2D observation noise. **This is not a natural human motion or iClone quality certification.**

## Frozen modes, inputs and sample

Use four held-out **Validation** motion families and the first seed by ascending seed in each:

1. `walk_turn_right`
2. `waist_twist_right`
3. `lateral_bend_left`
4. `reach_counterrotate`

The source Dataset has 96 Train, 24 Validation and 24 sealed Test clips. Take one of the two Validation clips per family for the first cross-family ablation. Each solver runs exactly **400 optimizer iterations** on **107 synchronized 24FPS 315°/45° observations**. The candidate sets are:

- **baseline:** Train-derived local Euler envelope; legacy Root initialization and no SO(3) motion prior.
- **root_so3:** SO(3) angular velocity/acceleration thresholds calibrated on Train only, Root Kabsch initialization, and observed Root heading evidence.
- **neutral_so3:** root_so3 plus Train-median initialization of pelvis/waist/spine/neck/clavicle local Euler channels.

No variant uses Validation or Test 3D ground truth inside the FK Solver; GT is used only after motion output to score.

## Non-negotiable contract checks

- `RL_BoneRoot` is the unique root bone, with its local Euler output all **zero**. It must never receive Blender pose F-curves.
- Root world motion is carried by the Armature **Object**, independent from RL_BoneRoot pose animation.
- The clip has exactly 107 frames with source frame indices 0..106 at 24 FPS, all 55 bones and 67 markers, finite positions and rotations.
- All three variants must share identical 2D observations, rig rest matrices and named joint ordering.
- The 24 sealed **Test** clips remain unread.

## Pre-specified development decision criteria

A new mode is **eligible for further evaluation**, not production promotion, only when:

1. Average Waist and Pelvis **SO(3) angular acceleration** both fall at least **30%** relative to baseline on the four-family aggregate. Monitor the other local Spine/Neck bones for deterioration.
2. Mean 3D MPJPE regression across the four families is at most **+0.10 cm** relative to baseline, and no family deteriorates by more than **+0.20 cm**.
3. Mean root world-orientation error improves by at least **50%** relative to baseline (a mode may not assume all Root yaw should be zero).
4. No invalid / animated `RL_BoneRoot` is produced. Foot-contact behavior is reported but currently uses a geometry/velocity **proxy**, not physical foot planting GT.
5. Any strong motion-specific failure prohibits declaring a general improvement. Do not use Test to select penalty weights or initialize models.

If no mode satisfies the accuracy/rotation trade-off, **retain the original Solver** and treat orientation-aware FK optimization as a continuing research task. Validation progress and complete per-bone metrics are written to:

```text
F:\anymove\runs\human\fullbody_finger_v2_hr_315_45\solver\spine_so3_research\multifamily_validation\
```

Run:

```powershell
cd F:\anymove
python -m tools.evaluate_human_v3_spine_multifamily --iterations 400 --takes 1
```

**Caveats:** No physical contact constraints, no independent natural motion, no generated AI video, no iClone FK-axis mapping, no new Reader training or inference.
