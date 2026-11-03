# Human V3 Nose Reader HR V2 — 672/192 heldout evaluation

**Date:** 2026-11-04
**Branch:** `feature/iclone-general-camera-angles`
**Status:** Completed all six training epochs and the previously unseen synthetic heldout evaluation. **Engineering smoke/quantitative benchmark pass; production motion-quality acceptance NOT passed because of catastrophic outliers.**

## Source and experimental contract

- Source asset: `runs/human/human_skinmax_multicolor_v3_nose.blend`, SHA-256 `677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc`.
- Source Dataset V2: `runs/human/fullbody_finger_v2_hr_315_45/dataset`, audit `PASS`. The 315°/45° pair comprises 2,688 synchronized stereo poses, 5,376 native 672px images, 192 motion clips, 67 markers including all 30 finger phalanges, ten finger tips, and two toe tips.
- Train: 144 clips / 4,032 images. Validation: 24 clips / 672 images. Test: 24 distinct clips / 672 images, i.e. 336 paired frames.
- Backbone: locally cached pretrained DINOv2 with registers base, frozen. Native Reader resolution 672×672, 192×192 heatmaps, separate neck + heatmap output head.
- Batch 16, 6 epochs, single seed 20261104. RTX 3070, PyTorch peak allocated GPU memory approximately 2.28 GiB. Checkpoint selected by lowest **Validation 2D mean px**, never Test.
- Checkpoint: `runs/human/fullbody_finger_v2_hr_315_45/reader_v2/full/best.pt`. Best epoch is zero-based 5 (sixth epoch).
- Existing 448px Reader is **not** a direct A/B benchmark: it was trained/tested on a different, smaller procedural dataset. Old 2D px and new 2D px must not be compared unnormalized.

## Epoch learning history (Validation; 2D image pixels)

| Epoch (1-based) | Train heatmap CE | All 67 mean px |
| ---: | ---: | ---: |
| 1 | 2.08124 | 1.7043 |
| 2 | 1.07159 | 1.4127 |
| 3 | 0.91049 | 1.3070 |
| 4 | 0.81117 | 1.2182 |
| 5 | 0.74550 | 1.1724 |
| 6 | 0.69380 | **1.0929** |

The sixth epoch is the checkpoint selected before seeing any Test labels or evaluation results.

## Independent heldout Reader performance

Inference on 24 distinct procedural motion clips, 672 images (336 calibrated 315°/45° pairs), direct two-view 3D triangulation of Reader predictions against Blender 3D ground truth, no FK Solver, no Temporal Joint Tracking, no Foot Guard.

| Body group | 2D mean px | 3D MPJPE cm | 3D P95 cm | 3D max cm | Temporal-displacement error cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 67 points | **1.0770** | **0.4916** | 1.2337 | **90.077** | 0.7219 |
| 40 fingers + fingertips | 1.2849 | 0.5894 | 1.4634 | 90.077 | 0.8998 |
| 10 finger-tip endpoints only | 1.9659 | 0.9252 | 2.4169 | 90.077 | 1.4400 |
| 10 legs/feet/toes | 0.8116 | 0.3587 | 0.6176 | **52.283** | 0.5057 |
| 2 toe tips only | 0.9900 | 0.4204 | 0.6373 | **52.283** | 0.6301 |
| 8 arms/hands/clavicles | 0.6956 | 0.3252 | 0.6422 | 16.248 | 0.3918 |
| 6 torso/root | 0.8991 | 0.4061 | 1.2438 | 9.057 | 0.5544 |
| 3 head/neck | 0.5624 | 0.2453 | 0.4855 | 0.727 | 0.2870 |

The equal weighting of five disjoint anatomical regions yields **0.3850 cm** mean 3D error. This is a custom secondary metric, not a standardized score.

## Severe-tail diagnosis — posthoc, no model tuning

Outlier analysis was done **after** checkpoint selection and Test evaluation. It must not be used to set fresh-training thresholds.

| Error threshold | Validation, 22,512 marker observations | Test, 22,512 marker observations |
| --- | ---: | ---: |
| 3D error > 2 cm | 382 | 387 |
| 3D error > 5 cm | 49 | 62 |
| 3D error > 10 cm | 16 | 12 |
| 3D error > 20 cm | 10 | 6 |
| 3D error > 50 cm | 6 | 4 |

Most Test >5 cm outliers involve a **single grossly incorrect camera prediction**: 46 errors are larger in the 315° camera, versus 16 in the 45° camera.

Selected previously unseen Test examples:

- `CC_Base_L_ThumbTip`, finger roll frame 56: **90.08 cm** 3D error, with **253.7 px** error on the 315° view but **4.4 px** error on the 45° view.
- `CC_Base_R_Mid3`, crouch frame 77: **89.23 cm** 3D, **247.6 px** error on 315° view.
- `CC_Base_R_Thumb1`, finger roll frame 56: **78.49 cm** 3D, **219.9 px** error on 45° view.
- `CC_Base_R_ToeTip`, squat frame 56: **52.28 cm** 3D, **258.1 px** error on 315° view.

Some wrong detections have **high raw heatmap peak logits**, so uncalibrated heatmap maximum alone is not a trustworthy joint visibility/confidence metric. Visual overlays confirm cross-hand, cross-body and foot/torso confusion rather than a uniform camera-calibration offset. The data projection QA itself passed with a maximum 3D GT / 2D marker difference of ~0.000069 px.

**Acceptance finding:** the low mean MPJPE demonstrates good performance on most synthetic heldout poses, but rare 50–90 cm spikes are unacceptable for direct iClone Motion import without bone-geometry and temporal uncertainty gating.

## Local artifacts and reproduction

- `reader_v2/full/training_report.json` — six training epochs, Validation data selection
- `reader_v2/full/best.pt` — selected Reader checkpoint; remains local due size
- `reader_v2/full/heldout_test.json` — complete independent Test metrics
- `reader_v2/full/outlier_diagnostic_posthoc.json` — named joint/view outliers for Validation and Test
- `reader_v2/full/outliers_contact_posthoc.jpg` — test overlay; **green = 3D GT projected point, red circle = Reader prediction, yellow line = disagreement**
- `reader_v2/full/eval_test.log` — inference run log

Commands from `F:\anymove`:

~~~powershell
python -m tools.eval_human_v3_hr_v2_reader
python -m tools.diagnose_human_v3_hr_v2_outliers
python -m tools.render_human_v3_hr_v2_outlier_contact
~~~

## Next engineering stage

1. **Train-only geometric joint association:** verify fingertip/phalange predictions relative to wrist and finger parent bones; flag large deviations on *each* camera, not just in final 3D.
2. **Dual-view evidence gating:** use reprojection and FK bone-length constraints to downweight or mask faulty 2D joint observations. A camera angle alone does not resolve a grossly wrong 2D keypoint.
3. **Temporal joint tracking:** preserve identity through frames and inpaint rejected observations with constrained FK / motion prior; do **not** simply output the previous frame's pose indefinitely.
4. **Hand/foot crop Reader:** native high-resolution supervised hand/foot training, with crop proposals predicted from image or reliable parent joints **at inference**. GT oracle crops are allowed *for Train supervision only*.
5. **New unbiased acceptance:** create entirely new procedural seeds/motion families **after freezing the guard and crop model**, measure outlier rate, equal-region MPJPE, and temporal jitter. Later repeat on natural motion and synchronized H3 videos.

Nothing in this report establishes that 315°/45° outperforms 0°/90° at 672px because a matched 0°/90° Dataset V2/Reader V2 experiment has not yet been run.
