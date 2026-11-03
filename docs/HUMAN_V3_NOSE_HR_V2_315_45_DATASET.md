# Human V3 Nose: HR Dataset V2, 315° / 45°

**Branch:** `feature/iclone-general-camera-angles`
**Source:** `runs/human/human_skinmax_multicolor_v3_nose.blend`
**Source SHA-256:** `677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc`
**Output:** `F:\anymove\runs\human\fullbody_finger_v2_hr_315_45\dataset`
**Release status (2026-11-04): `PASS` — all 5,376 native PNG images, GT, calibration and independent audit finished. No Reader training started.**

## Independent release verification

- All 12 family batches returned success; 192 clips and 2,688 matched 315°/45° pairs were confirmed.
- **5,376 RGBA PNGs** at 672×672 verified by PNG CRC and SHA-256, totaling approximately **1.569 GiB** of image files.
- All 67 marker UVs per view are consistent with 3D GT and calibrated Blender cameras; the **measured maximum absolute projection difference is 0.0000689 px** (acceptance threshold 0.0025 px).
- Counts confirmed: **4,032 Train / 672 Validation / 672 Test images**, and **2,688 per camera** across all splits.
- **4,032 Train-only GT hand/foot crop-hint entries** were generated; these boxes are not valid as inference-time predictions.
- Surface visibility labels across all 360,192 marker observations: **86,358 surface-visible, 142,760 surface-occluded, 114,946 uncertain, 16,128 unknown**. These are geometry proxy labels, not calibrated model probabilities.
- Audited release manifest is `dataset/manifest.json`; the authoritative check is `dataset/audit.json`.
- A 12-family visual review contact sheet is at `F:\anymove\runs\human\fullbody_finger_v2_hr_315_45\qa_contact_12_motion_families_315_45.jpg`.

## Data volume rationale

No fixed number of Blender training renders can be proved statistically optimal without a data-size learning curve. This experiment targets **2,688 images per physical camera**, a balanced design with >22% headroom over 2,200 images per camera:

| Item | Count |
| --- | ---: |
| Native image size | 672×672 RGBA |
| Camera azimuths | 315°, 45° |
| Motion families | 12 |
| Clips per family | 16 |
| Train clips / Validation clips / Test clips | 144 / 24 / 24 |
| Non-rest poses per clip | 14 |
| Stereo frame pairs | 2,688 |
| Frames per physical camera, all splits | **2,688** |
| Total image files | **5,376** |
| Train / Validation / Test image files, both cameras | 4,032 / 672 / 672 |

**Important:** Train alone has **2,016 images per camera**; 2,688 includes Validation/Test. More is not automatically better when repeated procedural motions add only near-duplicate poses. Future learning-curve training is needed to determine whether this count is actually sufficient.

## Motion and label contracts

The V3 Nose model has 55 pose-bone joints. GT includes **10 fingertip and 2 toe endpoints**, making **67 total 3D tracking markers**. All 30 finger phalange bones and both legs move in the procedural action families. Four new motion families extend the first eight: independent finger rolling, heel raise/turn, cross-step/grip, and squat/open palm. This is procedural motion, **not physically verified human gait or natural human MoCap**.

The camera slot `front` means 315°, and `side` means 45°; these are *legacy data-slot names*, not literal front or side camera positions. Both are level orthographic cameras with synchronized poses and calibrated 3D-to-2D projection.

Each rendered view stores image UV positions and a **surface visibility proxy** based on ten deterministically sampled dominant-skin-weight vertices per tracked surface region, checked with Blender BVH ray casts against the deformed mesh. This is *not calibrated 2D detector confidence*. The three unsupported surface regions (RL_BoneRoot, CC_Base_Pelvis, CC_Base_NeckTwist01) are explicitly marked `unknown`, never fabricated as visible.

A separate **Train-only** index of GT-derived left/right hand and foot crop boxes is generated for future supervised high-resolution training. Oracle crop boxes are **not used for unseen Validation/Test inference**.

## Finished release contract

The final independent audit must find 192 unique motion clips, exactly 2,688 matching two-camera frame pairs, exactly 5,376 valid 672×672 RGBA PNGs, 67 finite 3D/2D labels per view, independent Train/Validation/Test clip identities, sufficient true leg and finger movement, and matched calibrated camera projections.

When the audit passes, these artifacts live inside `dataset`:

- `manifest.json` — all image rows, angles, projection matrices, markers and motion splits
- `audit.json` — release counts and visibility-proxy classes
- `image_sha256.json` — SHA-256 hashes of all PNGs after integrity verification
- `train_only_gt_crop_hints.json` — Train-only hand/foot supervised crop targets
- `motion_gt/*.json` — GT poses, 107 frames per clip
- `family_00_manifest.json` to `family_11_manifest.json` — resumable batches
- `train/{front,side}/*.png`, `validation/{front,side}/*.png`, `test/{front,side}/*.png`

## Reproduce offline

From `F:\anymove`, generate one family using Blender 5 without modifying the original Blend:

~~~powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b --python-exit-code 17 "runs/human/human_skinmax_multicolor_v3_nose.blend" -P "tools/render_human_v3_hr_v2_315_45.py" -- --family-index 0
~~~

Repeat indices 0 through 11, then run the independent final audit (which verifies PNG CRCs and hashes):

~~~powershell
python -m tools.audit_human_v3_hr_v2_315_45
~~~

**Reader training has deliberately not been started** because this stage only requested a completed 315°/45° Dataset. Retain the original 448px Reader checkpoints as baselines; a 672px/192-heatmap Reader must be separately validated and trained after dataset release.
