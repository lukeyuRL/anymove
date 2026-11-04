# AnyMove Human V3 Nose — P1 Temporal Guard → FK Solver integration

**Date:** 2026-11-04
**Branch:** `feature/iclone-general-camera-angles`
**Status:** P1 calibrated two-view joint guard, scale-aware human FK contract and actual Reader-to-FK CLI integration implemented/tested. **Not iClone runtime/retarget acceptance.**

## Scope and prerequisites

- **Model:** `runs/human/human_skinmax_multicolor_v3_nose.blend` (SHA-256 `677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc`).
- **Reader:** frozen DINOv2 672×672 / 192×192 67-point checkpoint at `runs/human/fullbody_finger_v2_hr_315_45/reader_v2/full/best.pt`.
- **Cameras:** 315° and 45° level orthographic, two synchronized 672px slots, meters/pixel = 2.432802200317383/672.
- **Rig:** 55 CC bones with 67 observed markers (including ten fingertips and two toe tips); Blender Armature object has **uniform scale 0.01**. Exporter writes world-meter-scaled rest matrices; no destructive Apply Transform on source.
- **P1 calibration:** **Train-only**, 144 motion clips; stores parent indices, 3D segment-length statistics, per-camera projected segment and relative-motion thresholds. It reads no Validation or Test 3D ground truth when fitting the calibration.
- **Runtime:** inference uses Reader keypoint positions and the fixed calibration only. Any 3D GT is used **after** inference exclusively to score predictions.

## What was fixed

1. **Blender→Torch FK joint initialization.** The exported bone tail-marker relationship now favors the immediate child-bone **head** even when Blender Bone chains are disconnected. For non-+Y Character Creator bones, `ik_init` aligns the actual bone-to-child rest vector. Validated against three Blender poses: maximum FK matrix/marker error **0.00033 cm**.
2. **Solver frame-spacing regularization.** The previous acceleration loss treated 5 observations sampled every seven source frames as if they were five contiguous frames. At the initial correct IK state it contributed **5,436** loss units versus an approximately **0.2** observation loss, forcing the solver away from a good pose. Source-frame normalized derivatives now preserve the previous formulation for unit-frame spacing.
3. **Human-specific optimization stability.** The opt-in Human V3 contract uses lower learning rates, gentler acceleration/velocity/prior penalties, fixed small robust data scale, gradient clipping and best-finite-iterate retention. Existing Wolf/Dragon optimization settings are not intentionally changed except when explicit non-unit source frame indices are supplied.
4. **P1 temporal joint association.** Single-camera outliers are rejected using train-derived projected bone ranges and isolated temporal spike checks. Suspect joints receive **heuristic evidence reliability 0.2**; 3D skeleton length violations are marked unreliable rather than automatically overwritten. Input `detections.npz` is never modified.
5. **Layout and metadata.** `mocap.layout` and Blender `common` support creature-specific height `H` (672 for this Human V3 rig). Solver checks camera angles, image size, meter scale, joint order, and carries `source_frame_indices`, source NPZ hash and a detailed temporal diagnostic into the output.
6. **Safety gates.** The synthetic Human rig currently exports broad **unvalidated Euler limits**. The CLI refuses the Human V3 rig unless explicit `--allow-provisional-human-limits` is passed. Produced `motion.json` is **Blender local Euler space**, not iClone FK channel space. Never pass it to `iclone_adapter.motion_writer` as-is.

## Controlled synthetic QA results

### FK initialization and Solver optimization, TRAIN pose only

5 frames, 67 markers, synthetic single-view ThumbTip error injected (215.41 px), then fixed P1 temporal calibration:

| Quantity | Result |
| --- | ---: |
| Blender / Torch independent FK max | **0.00033 cm** |
| Analytical IK initial mean 3D error (clean GT) | **0.234 cm** |
| Guarded bad thumb 2D residual | **5.50 px** |
| Human Solver after 200 iterations, mean 3D error | **0.0473 cm** |
| After 200 iters, reprojection (315°/45°) | **0.116 / 0.083 px** |

This is a **functional numerical QA** using synthetic pose GT, not a claim that generated-video 3D errors are below 0.1 cm.

### Frozen Reader + P1 tracker (previously new 12-seed Blender set, no FK)

- 12 fresh procedural motion clips, 168 paired frames, 336 images.
- Whole-body raw 3D MPJPE **0.5029 cm**, P1 **0.4689 cm**.
- Extreme (>50 cm) outliers **4 → 0**; maximum 3D error **107.18 → 10.68 cm**.
- Temporal displacement error **0.7474 → 0.6743 cm**.
- Inference uses fixed Reader/Train calibration; no Test-directed parameter tuning.

### Real Reader predictions → full FK Solver CLI, additional integration QA

Same 14-frame **finger_independent_roll** procedural clip, frozen Reader 672/192 and **400** Solver iterations each:

| Quantity | Raw Reader → Solver | P1 Guard → Solver |
| --- | ---: | ---: |
| 67-point 3D MPJPE | **1.061 cm** | **1.022 cm** |
| P95 3D error | **4.083 cm** | **4.206 cm** |
| Largest 3D error | 11.08 cm | 11.10 cm |
| 315° reprojection mean | 2.598 px | **2.226 px** |
| 45° reprojection mean | 2.087 px | **1.966 px** |
| Temporal displacement error | 1.285 cm | **1.265 cm** |

8 observations in the 315° view were inferred and downweighted. P95 and maximum errors did **not** improve; do not describe this one-clip check as complete FK Motion quality acceptance. The test clip was in an earlier P1 aggregate QA, so this is **additional integration diagnosis**, not a new blind generalization test.

## Repeatable local commands

Run from `F:\anymove`:

```powershell
python -m tools.calibrate_human_v3_hr_v2_temporal
python -m pytest -q tests/test_human_temporal_skeleton_tracker.py tests/test_solver_sparse_cadence.py
python -m tools.validate_human_v3_hr_v2_solver_fk
python -m tools.smoke_human_v3_hr_v2_p1_solver 200
python -m tools.check_human_v3_p1_solver_main
python -m tools.export_human_v3_hr_v2_reader_detections --clip-index 8
python -m tools.evaluate_human_v3_hr_v2_p1_fk_cli
```

For **explicit research-only** use of `mocap.solve`:

```powershell
$env:MOCAP_CREATURE = 'F:\anymove\runs\human\fullbody_finger_v2_hr_315_45\solver'
$env:MOCAP_CAMERA_AZIMUTHS = '315,45'
$base = 'runs/human/fullbody_finger_v2_hr_315_45'
$clip = 'hrv2_p1_fresh_finger_independent_roll_s208392'
python -m mocap.solve "$base/solver/reader_p1_fresh/$clip.npz" `
    "$base/solver/rigged/rig_info.json" `
    "$base/solver/reader_p1_fresh/motion_reader_p1_guarded.json" `
    --no-align --temporal-calibration "$base/reader_v2/full/temporal_skeleton_train_calibration.json" `
    --iters 400 --allow-provisional-human-limits
```

This `motion.json` contains only 14 **sparsely sampled** source frames. The `source_frame_indices` list preserves original timestamps; it cannot be imported as 14 consecutive 24-FPS frames without retiming/interpolation.

## Remaining blocking gates

1. **Proper human joint limits**, foot planting, and low-confidence FK constraints need clinical-quality kinematic design and fresh evaluation. Current Euler bounds are `[-179°,179°]` placeholders, not production-safe.
2. **Camera-specific 2D confidence/visibility** remains uncalibrated. P1's 0.2 weight is a heuristic reliability modifier, not a probability.
3. Full continuous 24 FPS capture (not 14 sparse samples), calibrated motion retiming and root translation/rotation verification.
4. **iClone host mapping**. The iClone P1 writer requires `anymove.iclone.fk_channels/1` with a live exported skeleton hash and calibrated axes. Current `mocap.solve` output uses Blender local Euler angles and must **NOT** be fed directly to that writer. Match a live iClone Human V3 avatar, confirm bind/rest matrices, establish a per-bone axis mapping and verify a disposable-scene write/readback.
5. Wider evaluation across multiple unseen seeds, natural human MoCap, other body shapes and synchronized AI-generated videos before claiming a generic AnyMove default camera choice.
