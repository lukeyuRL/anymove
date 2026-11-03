# AnyMove Human V3 Nose — Foot Chain Geometric Guard (P0)

**Branch:** `feature/iclone-general-camera-angles`
**Scope:** human 67-marker, calibrated orthographic dual-camera Reader; no new H3 video or remote training.
**Source model:** `F:\anymove\runs\human\human_skinmax_multicolor_v3_nose.blend` (original SHA-256 verified)
**Status:** implemented as a pure post-Reader library plus validation and fresh-test evaluation. **Not yet wired into the iClone runtime Solver.**

## Problem confirmed

The prior Reader A/B had a disproportionate problem in the 315°/45° camera pair: three extreme ToeTip localizations exceeded 150 px, including points predicted near the upper body rather than the foot. Some Validation errors moved **both** the ToeBase and the ToeTip about 80 px while their relative distance still looked plausible. This shows that checking only ToeTip→ToeBase is insufficient.

At 448×448 input, the actual ToeTip→ToeBase span is generally only 1–5 px. This is a high-risk target for a single raw heatmap maximum. The failure is *not proof* the 315°/45° camera geometry is worse.

## Implemented two-stage fail-safe

New module: `mocap/toe_tip_guard.py`.

1. Fit per-view, per-foot anatomical projected regression from TRAIN GT only:
   - **Calf→Foot→ToeBase** fallback using a 2×2 displacement map.
   - **Foot→ToeBase→ToeTip** fallback using a second 2×2 map.
2. Set conservative projection segment-length and regression residual bounds **from TRAIN only**. There are no TEST-driven threshold updates.
3. On each frame, test ToeBase relative to Foot and the TRAIN-predicted location. Correct ToeBase only when the Calf/Foot anchor is plausible.
4. Test ToeTip relative to the corrected ToeBase. Correct it only when the Foot/ToeBase anchor is plausible.
5. Return per-marker explicit flags: `replaced` versus `untrusted_no_repair`. If an upstream anchor is invalid, propagate the unreliable flag to the downstream tip.
6. Expose `confidence_scales_from_diagnostics`: uncorrected keypoints receive 1.0, model-inferred replacements default to a **heuristic** 0.25, and unrecoverable observations receive 0.0. This is a *reliability modifier*, NOT a calibrated probability; any Solver integration must validate the scale.

No test-frame GT is used to correct a prediction. Only the inferred keypoint UV and frozen TRAIN anatomical relationships are used.

## Validation — checkpoint and guard frozen before fresh Test

Controlled Train / Validation / prior Test partition: **16 / 4 / 4 clips**, with 67 tracked markers (55 bone heads, 10 fingertips, 2 toe tips). Checkpoints were previously selected on Validation 2D error.

For 315°/45°, Validation on the left toe in the first camera:

| Metric | Raw Reader | Guarded Reader |
| --- | ---: | ---: |
| ToeBase mean 2D error | 8.1801 px | **2.6745 px** |
| ToeTip mean 2D error | 8.2999 px | **2.9324 px** |

Both regressions and bounds are TRAIN-only; this Validation dataset was never used to fit them. `foot_chain_guard_train_val.json` saves the frozen coefficients, thresholds and validation audit.

## Post-hoc diagnosis — original TEST (already examined)

The original Test had been inspected while designing the fix and must **not** be described as an unbiased test of the remedy.

| 315°/45° | Raw Reader | Guarded |
| --- | ---: | ---: |
| ToeTip 3D error | 4.9855 cm | **0.9807 cm** |
| Full 67 marker MPJPE | 1.1882 cm | **1.0686 cm** |

Three ToeTip detections were repaired in the prior TEST. We also confirmed that some baseline ToeBase errors are detectable and repairable. These figures are retrospective debugging evidence only.

## Fresh never-trained / never-tuned test

After fixing the guard logic and checking it using TRAIN/Validation, generated **four new fullbody+finger motion clips** with new seeds and separate image output:

- `fresh_march_open_close_9413`
- `fresh_crouch_fist_9541`
- `fresh_kick_and_wave_9671`
- `fresh_balance_point_9781`

The pretrained Reader weights and guard matrices/thresholds were left unchanged. The fresh dataset has **112 Blender images total**, 56 per angle pair, corresponding to 28 paired frames per configuration.

| Fresh synthetic TEST | 0°/90° | 315°/45° |
| --- | ---: | ---: |
| Raw 67-point MPJPE | 1.5135 cm | 0.8915 cm |
| **Guarded 67-point MPJPE** | 1.5135 cm | **0.8672 cm** |
| Raw ToeTip MPJPE | 0.8995 cm | 1.7704 cm |
| **Guarded ToeTip MPJPE** | 0.8995 cm | **0.9576 cm** |
| Guarded five-region equal-weight score | 0.9801 cm | **0.7703 cm** |
| ToeTip detections replaced | 0 | 1 |
| Unrecoverable toe-chain observations | 0 | 0 |

The 315°/45° branch improved on the new seeds without modifying training weights, while leaving the 0°/90° result unchanged. This remains **small procedural synthetic evidence**, not proof of superiority across real human motion, new characters or H3-generated video.

Local fresh-test report: `F:\anymove\runs\human\fullbody_finger_v3_nose\fresh_test_20261103\fresh_ab_summary.json`.

## Operational use

These tools run on the local PC, requiring no paid services. Execute from `F:\anymove`:

~~~powershell
python -m pytest -q tests/test_human_toe_chain_guard.py

# Already-trained Reader checkpoints are required.
# Fits geometry ONLY from Train, evaluates on Validation:
python -m tools.validate_foot_chain_guard

# For retrospective QA of previously inspected Test:
python -m tools.evaluate_foot_chain_posthoc

# For a new clip-seed Blender test (do not modify the original V3 Nose Blend):
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b --python-exit-code 17 'runs/human/human_skinmax_multicolor_v3_nose.blend' -P tools/render_human_fullbody_fresh_test.py
python -m tools.evaluate_foot_chain_fresh_test
~~~

The input checkpoint SHA-256, model SHA-256 and camera azimuth metadata are validated before applying train-derived corrections. Save both corrected UV values and diagnostic flags when integrating with a Solver.

## Remaining steps (not completed)

**P1 — time-aware Solver:** consume the flags, lower the evidence weight for inferred keypoints and suppress unrecoverable input. Add temporal continuity, rigid-bone FK and true confidence/visibility calibration. The current analysis triangulates independent frames and does **not** show actual full iClone motion behavior.

**P2 — small-part Reader:** train hand/foot crops or a higher-resolution feature/heatmap head; existing 448px images have over 70% of neighboring fingertip pairs within 6px in the baseline, even more in 315°/45°. Small fingertip improvements should be measured on separate new holdouts.

**P3 — wider evidence:** multiple training seeds, additional angles (including 45°/135°), physically verified walk/jump/gait, new humans/rigs and synchronized H3 generated clips. The results above are single-seed, a single stylized human, and procedurally rendered motions. Do not equate them to general AnyMove quality.
