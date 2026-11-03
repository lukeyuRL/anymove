# AnyMove iClone — Human V3 Nose Fullbody + Finger Reader Camera A/B

**Date:** 2026-11-03  
**Branch:** `feature/iclone-general-camera-angles`  
**Input Blender:** `runs/human/human_skinmax_multicolor_v3_nose.blend` (SHA-256 `677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc`)  
**Status:** Completed **controlled synthetic Reader comparison**. Not H3 video acceptance, natural gait evaluation, or iClone Motion Solver acceptance.

## 1. Why the preceding Reach-only trial was insufficient

The preceding V6 Reach GT had **zero motion in thigh/calf/foot joint trajectories across all 20 clips**. It measured mostly arm/hand behavior. We therefore generated new, clip-disjoint fullbody motion on the same 55-bone CC skinned Human V3 Nose model without modifying the source Blend. Finger phalanges and toe tips now have explicit moving GT.

## 2. Whole-body and fingertip motion dataset

Eight scripted procedural movement families:

1. March with opening/closing hands
2. High-knee lifts and grasping
3. Lateral steps and finger splay
4. Crouch with fists
5. Lunge and reach
6. Torso turn with stepping
7. Kick with hand wave
8. Single-leg balance and pointing

Each family moves legs, arms, torso/head and all **30 finger phalange bones**. These are **procedural poses**, not verified human gait, motion-capture recordings, or physically accurate foot-contact IK.

Each 107-frame clip (24 FPS) includes ground-truth positions of **67 markers**:
- 55 original CC pose-bone head joints;
- 10 explicit fingertip *tail* endpoints (five per hand);
- 2 toe-tip endpoints.

Independent verification over **24 clips** found actual lower-body movement, hand movement and finger articulation in each clip. All 30 finger bones rotate in every clip (smallest individual local angular sweep ≈0.03479 rad). The Blender GT-to-UV calibration mismatch is less than 0.00003 px.

Source motion GT saved locally at `runs/human/fullbody_finger_v3_nose/dataset/motion_gt`. No model, texture, FBX, Blend or checkpoint assets were uploaded to the public repository.

## 3. Strict, matched two-camera experiment

| Contract | Baseline | Oblique |
| --- | ---: | ---: |
| Calibrated azimuths | 0°/90° | 315°/45° |
| Blender frame size per camera | 448×448 | 448×448 |
| Original 3D motion clips | Identical | Identical |
| Entire dataset (two camera images per sampled frame) | 576 images | 576 images |
| Clip-disjoint Train / Validation / Test | 16 / 4 / 4 | 16 / 4 / 4 |
| Effective dynamic images (Train / Validation / Test) | 384 / 48 / 48 | 384 / 48 / 48 |
| Reader architecture | DINOv2 registers base + 67 heatmaps | Same |
| Frozen pretrained backbone | Yes | Yes |
| Trainable parameters | Heatmap head only | Same |
| Training | 6 epochs, batch 4, matched seed 20261104 | Same |
| Best epoch selected by validation 2D mean | Epoch 4 | Epoch 2 |

**Static rest endpoints at frames 0 and 106 were removed from Reader Train / Validation / Test.** This prevents identical neutral rest poses appearing across clip splits. The Test images never participated in checkpoint selection. Models were trained only with local RTX 3070; no paid service or H3 video generation was used.

Validation 2D means at best epoch: baseline **1.7473 px**, oblique **1.3640 px**.

## 4. Unseen Test 3D results

Raw 3D triangulation of the Reader's 2D predictions, compared with independent Blender 3D GT. No FK, temporal association, Visibility head, Motion Prior or motion-generation inference is included.

| Keypoint group | 0°/90° 3D error | 315°/45° 3D error |
| --- | ---: | ---: |
| **All 67 points** | 1.5382 cm | **1.1882 cm** |
| 30 finger phalange head joints | 1.6571 cm | **1.1212 cm** |
| 10 fingertip endpoints | 2.6332 cm | **1.2380 cm** |
| 8 leg bone head joints | 1.2374 cm | **0.8298 cm** |
| 6 arm/hand bone head joints | **1.2614 cm** | 1.5702 cm |
| 2 toe-tip endpoints | **0.8312 cm** | 4.9855 cm |
| 4 torso markers (partial grouping) | 0.6731 cm | **0.5999 cm** |

Mean 2D errors: all 67, **2.3180 vs 1.7647 px**; ten fingertips, **3.6223 vs 1.7933 px** (baseline vs oblique).

Under equal fixed per-joint weighting, the new 315°/45° configuration has a **22.75% lower all-marker 3D MPJPE**. This is not proof of a general whole-human advantage.

## 5. More honest equal-anatomy-region score

40 of the 67 tracked points describe fingers. The canonical per-marker MPJPE **overweights finger tracking** if the user wants a whole-body decision. Split all 67 markers into five disjoint regions and give each anatomical region 20% weight:

| Disjoint region | Markers | 0°/90° | 315°/45° |
| --- | ---: | ---: | ---: |
| Fingers / fingertips | 40 | 1.9011 cm | **1.1504 cm** |
| Legs / feet / toes | 10 | **1.1562 cm** | 1.6609 cm |
| Arms / hands / clavicles | 8 | **1.1842 cm** | 1.3141 cm |
| Torso / root | 6 | **0.7599 cm** | 0.8335 cm |
| Head / neck | 3 | **0.4730 cm** | 0.4896 cm |
| **Five-region equal-weight score** | 5 regions | 1.0949 cm | 1.0897 cm |

Only **0.0052 cm** separates the five-region scores, an insignificant difference at this experiment's scope. An equal-region score is a deliberately chosen secondary metric, not a universal standardized benchmark.

**Interpretation:** 315°/45° improves fingers and the non-toe leg joints in this synthetic test, but is not proven superior for the overall human skeleton, especially with the occasional toe-tip failure.

## 6. Diagnosed toe-tip failure mode

Independent diagnostic on the *already selected* models:

- 0°/90°: zero toe-tip 2D detections exceeded 10 px;
- 315°/45°: **three left-toe detections** exceeded 150 px;
- two oblique left-toe predictions landed near upper-body image coordinates instead of near the foot; the largest was ~257 px away.

Those outliers dominate the 315°/45° toe-tip mean and expose a catastrophic-tail risk. Do not select a winner based only on the lower mean overall MPJPE. This is a post-hoc **diagnosis**, not a model hyperparameter chosen on Test.

## 7. Finger pixel-resolution limit

At native **448×448** per-camera input, adjacent fingertip GT projections were:

| | 0°/90° | 315°/45° |
| --- | ---: | ---: |
| Median adjacent fingertip separation | 4.58 px | 4.31 px |
| Adjacent fingertips less than 6 px apart | 70.7% | 80.6% |

This is a real spatial-resolution constraint. Before making product-quality claims, test a higher-resolution / crop-aware Reader, explicit per-finger visibility and uncertainty, and multiple independent training seeds.

## 8. Local artifacts

All outputs are under `F:\anymove\runs\human\fullbody_finger_v3_nose`:

- `dataset/`: identical Blender GT and independent A/B camera samples;
- `dataset/fullbody_motion_qa.json`: every motion clip's leg/finger/hand travel audit;
- `fullbody_finger_marker_qa.json`: 67-marker geometry and fingertip pixel scale;
- `reader_models_strict/baseline_0_90/best.pt`: 0°/90° checkpoint;
- `reader_models_strict/symmetric_315_45/best.pt`: 315°/45° checkpoint;
- `reader_models_strict/heldout_camera_ab_comparison.json`: canonical Reader A/B;
- `reader_models_strict/anatomy_balanced_ab.json`: disjoint region equal weighting;
- `reader_models_strict/symmetric_315_45/toe_tip_diagnostic.json`: bad toe-tip predictions;
- `reader_models_strict/human_v3_nose_fullbody_finger_reader_ab_contact.png`: heldout overlay contact, green=GT, red=Reader.

Models/images stay local and were not added to GitHub.

## 9. Reproduction (runs locally, no H3)

~~~powershell
cd F:\anymove
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b --python-exit-code 17 'runs/human/human_skinmax_multicolor_v3_nose.blend' -P tools/render_human_v3_fullbody_finger_ab.py
python -m tools.audit_human_v3_fullbody_finger_dataset
python -m tools.train_human_v3_fullbody_finger_reader_ab --epochs 6 --batch 4
python -m tools.eval_human_v3_fullbody_finger_reader_ab
python -m tools.diagnose_human_v3_fullbody_toe_tips
python -m tools.analyze_human_v3_fullbody_anatomy_balance
python tools/make_human_v3_fullbody_finger_reader_contact.py
~~~

## 10. Remaining engineering gates

- Fix **catastrophic toe-tip outliers** using training-only or validation-only error analysis, calibrated uncertainty, surface occlusion flags and temporal/bone-length constraints. Use **new heldout clips** after making those changes; do not optimize on the already viewed four TEST clips.
- Increase finger spatial resolution (e.g. a higher-resolution Reader or hand crops), then compare with the same workload and splits.
- Add diverse motion sources including natural walk and verified foot-ground mechanics. Current scripted motions are not natural human locomotion.
- Run multiple training seeds and additional camera configurations (especially 45°/135°) before choosing a general default.
- Test final synchronized AI-generated video and iClone FK solver **only as a separate approved stage**.
