# AnyMove P3 �X Four-View Reader + Mesh-Surface Visibility / Uncertainty

Date: 2026-11-01  
Branch: `feature/dino-motion-quality-p123` at `F:\\anymove`  
Status: **Four-view training pipeline PASS; candidate quality NOT PROMOTED**  
Scope: Dino and Dragon, existing Blender static-pose datasets, local RTX 3070, Python 3.14.3. No new H3 videos, no full 2,200-pose retraining, no production iClone changes, no Commit/Push.

## 1. Why this phase

P2 completed a generic 2/3/4-camera FK Solver. Geometric experiments using Blender-derived 2D observations with controlled corruption improved 3D reconstruction on both creatures, while feeding the existing Front/Side-trained Reader into Back/Opposite cameras worsened 3D quality. This is **model/view generalization and uncertainty calibration**, not proof of N-camera geometry being wrong.

P3 is a contained first test of a new four-view Reader checkpoint and actual Mesh Surface Part Visibility supervision.

## 2. Exact dataset provenance / train-val leakage protection

Data is REPLAYED from the original stored pose metadata, not generated with H3 or from a guessed 3D Motion:

- Dino source: `runs/dino/dataset_hr_v1`; 67 Bones / 92 Keypoints.
- Dragon source: `runs/dragon/dataset_hr`; 42 Bones / 50 Keypoints.
- Four-view Blender script: `blender/render_four_view_visibility.py`; validated Pose Replay/Camera Geometry/Depth+Skinning Part-ID.
- Output: `runs/{creature}/dataset_fourview_p3_smoke_v1`.
- **Train poses**: indices 1�V24; **Validation poses**: indices 25, 50, 75, 100. All four views from a pose are always in the *same split*. In particular, no `25_front` may enter Train while `25_back` enters Validation.
- 24 Train �� 4 = 96 view images, 4 held-out validation poses �� 4 = 16 images: **112 images per Creature**.
- Source images rendered at `scale=0.5` for smoke; Reader consumes 672��672 letterboxed model input and heatmap 192��192.
- Source metadata SHA256, `rig_info.json` SHA256, exact 3D pose replay, 2D camera reprojection, Mesh Part-ID and depth/camera symmetry already independently verified at P1.
- Original 2,200-pose Reader checkpoints are **NOT** modified. Existing Reader checkpoints have already seen some of the original training pose identities for Front/Side; validation IDs multiples of 25 reflect the original val split, but independent test sequences are still required to establish generalization.

## 3. Implemented modules

| File | Function |
|---|---|
| `mocap/fourview_data.py` | Creature-generic 4-view dataset, exact pose-disjoint split, RGBA/gray-background augmentation, 672 letterbox, UV+visible/occluded/unknown labels and validation error/visibility metrics |
| `mocap/train_four_view.py` | Original Reader frozen-backbone warm-start, train decoder + new Quality Head, heldout four-view metrics, model checkpoint and independent baseline evaluation |
| `tools/build_four_view_observations.py` | Support P3 version-2 mesh-surface Quality checkpoint; export view-indexed UV/Peak/Visibility/Error with explicit uncalibrated semantics; store GT separately |
| `mocap/multiview.py` | Validate P3 optional Quality Head fields; reject any actual Mesh GT label in Solver packet |
| `tools/eval_four_view_reader_p3.py` | Automatic pass/fail based on 2D, independent Blender 3D FK, Visibility AUC/ECE, expected error calibration and 2-view regression |
| `tests/test_iclone_fourview_reader_p3.py` | Real dataset integrity, train/val leakage, unknown-label handling, original backbone immutability, dedicated experimental checkpoint and no production promotion |

New P3 checkpoint has `quality_spec.version=2`, `target=mesh_surface_part_visibility`, `visibility_semantics=mesh_surface_part_visibility_v1_uncalibrated`. The old `mocap.detect.load` does not accept this version; the new 4-view builder explicitly loads it in experimental mode. This fail-closed boundary prevents accidental use in production.

### Visibility and Uncertainty losses

The source label is **visibility of the owning skinned Mesh Surface Part** in each camera, NOT a ray to an embedded bone center and not an H3-measured occlusion.

- Skinning Region Part-ID + Depth Buffer from Blender P1 determines per-view **1 visible / 0 occluded / -1 unknown**.
- `-1` and offscreen coordinates are excluded from visibility supervised loss. Occluded examples receive 3x BCE weight because of class imbalance.
- Reader's second Quality Head channel regresses `log1p(heatmap 2D localization error)`, using detached predicted UV and Blender Ground Truth keypoint position.
- Loss = existing Reader heatmap MSE �� 1000 + 0.2 �� (visibility weighted BCE + expected-error SmoothL1).
- Backbone is frozen; neck/upsampling/heatmap head and new `quality_out` train with separate learning rates, AdamW, CUDA FP16 AMP. Batch=2, 3 epochs, base decoder LR 0.0002; quality LR 0.0004.
- Inference Quality Scores have explicit metadata and are **NOT** automatically trusted as `P(visible)` or inserted into FK Solver weights.

## 4. Actual 2D results: same four independent held-out poses per Creature

All 2D errors below are in **unscaled camera Layout pixels** (original rendered smoke images have half resolution). Before/after numbers can be compared within these paired experiments only.

### Dino, 92 Keypoints

| View | Original Reader mean | P3 Reader mean | Delta |
|---|---:|---:|---:|
| Front | 20.43 px | 22.39 px | **+1.96 px worse** |
| Side | 22.42 px | 17.29 px | -5.13 px |
| Back | 52.18 px | 43.40 px | -8.79 px |
| Opposite Side | 36.77 px | 21.56 px | -15.21 px |
| Overall | **32.95 px** | **26.16 px** | -6.80 px |

### Dragon, 50 Keypoints

| View | Original Reader mean | P3 Reader mean | Delta |
|---|---:|---:|---:|
| Front | 25.72 px | 25.61 px | -0.11 px |
| Side | 27.29 px | 24.55 px | -2.74 px |
| Back | 67.49 px | 65.01 px | -2.49 px |
| Opposite Side | 41.78 px | 37.44 px | -4.34 px |
| Overall | **40.57 px** | **38.15 px** | -2.42 px |

**Averaged localization improves, but Front Dino regresses; Dragon Back remains poor.** This only evaluates static Workbench Blender images, not synchronized H3 videos.

## 5. Full P2 N-camera FK on genuinely held-out Blender 3D GT

The same P2 N-view Solver uses either Front+Side or all four camera predictions from the SAME checkpoint. Mean Euclidean 3D Joint Error in **mm** against **separate Blender world GT** (4 validation poses):

| Creature | Original 2V | Original 4V | Fine-tuned 2V | Fine-tuned 4V |
|---|---:|---:|---:|---:|
| Dino | **487.37** | 737.21 | 568.16 | 663.04 |
| Dragon | **153.33** | 215.60 | 188.10 | 205.16 |

The fine-tuned Reader improves four-view results vs original four-view by 74.17 mm for Dino and 10.44 mm for Dragon, **but four-view is still worse than original two-view** on both. The fine-tuned model also makes two-view 3D results worse.

**Do NOT promote to production, do NOT replace existing Reader checkpoint, do NOT send these outputs to iClone as validated motion.**

The original Reader is trained on Front/Side with a different render domain; P3 smoke uses neutral Workbench stills and tiny pose coverage. These results do not prove something about H3 4-view, real motion dynamics, or novel Creature species.

## 6. Visibility / Localization Uncertainty validation

Measured ONLY against held-out **Blender Mesh Surface Part** labels, not real-H3 visibility.

| Creature | Front AUC | Side AUC | Back AUC | Opposite Side AUC |
|---|---:|---:|---:|---:|
| Dino | 0.684 | 0.787 | 0.767 | 0.873 |
| Dragon | 0.686 | 0.687 | 0.591 | 0.834 |

The model is not calibrated:

- Dino Brier: Front 0.176, Side 0.155, Back 0.154, Opposite 0.133; ECE about 0.13�V0.24.
- Dragon Brier: Front 0.135, Side 0.148, Back 0.117, Opposite 0.107; ECE about 0.11�V0.19.
- Predicted-vs-observed localization mean-error calibration gap (Layout px): Dino Front 10.87, Side 6.00, Back **30.67**, Opposite 7.84; Dragon Front 12.30, Side 13.24, Back **50.21**, Opposite 25.60.

The model does not learn a stable `P(visible)` appropriate for 3D weighted fusion across viewpoints; it cannot yet be enabled in `mocap.solve_multiview`.

## 7. Artifact location and reproducible commands

```powershell
cd F:\\anymove

# Uses existing P3 training data. Reader original checkpoint remains untouched.
python -m mocap.train_four_view `
    runs/dino/dataset_fourview_p3_smoke_v1 `
    assets/dino/rigged/rig_info.json `
    --init-reader runs/dino/reader_hr_v1.pt `
    --out runs/dino/NEW_P3_EXPERIMENT.pt `
    --epochs 3 --batch 2 --workers 0 `
    --lr 0.0002 --quality-strength 0.2

# Readout of one already trained checkpoint on 4 heldout poses.
python -m mocap.train_four_view `
    runs/dino/dataset_fourview_p3_smoke_v1 `
    assets/dino/rigged/rig_info.json `
    --eval-checkpoint runs/dino/reader_fourview_p3_smoke_v1.pt `
    --report runs/dino/NEW_P3_REEVALUATION.json `
    --batch 2

# Independent 2D + 3D + reliability acceptance gate
python tools/eval_four_view_reader_p3.py `
    --creature dino --repo F:\\anymove `
    --out runs/dino/NEW_P3_ACCEPTANCE.json
```

Actual experiment files:

- `runs/dino/reader_fourview_p3_smoke_v1.pt`
- `runs/dragon/reader_fourview_p3_smoke_v1.pt`
- `runs/{creature}/reader_fourview_p3_smoke_v1.validation.json`
- `runs/{creature}/fourview_reader_p3_acceptance_v1.json`
- `runs/{creature}/dataset_fourview_p3_smoke_v1/p3_{baseline,trained}_heldout_views.npz`
- `runs/{creature}/dataset_fourview_p3_smoke_v1/p3_{baseline,trained}_fk_3d_eval.json`

P3's 3D Ground Truth is only in the distinct `*.ground_truth.npz`, not in train, Reader inference or Solver packets.

## 8. Acceptance / next prioritized work

Automated `tools/eval_four_view_reader_p3.py` gates:

1. Four-view mean 2D heldout improves: **PASS** for Dino and Dragon.
2. Fine-tuned 2-view 3D must not regress relative to original Reader: **FAIL** for both.
3. Fine-tuned 4-view 3D must beat fine-tuned 2-view 3D: **FAIL** for both.
4. All-view Mesh Visibility AUC >=0.8 and ECE <=0.1: **FAIL** for both.
5. All-view expected localization-error mean calibration gap <=5 Layout px: **FAIL** for both.

Thresholds are operational screening choices, not physically guaranteed theory.

**P3 implementation and smoke training: COMPLETE. Quality promotion: FAIL / disabled by default.**

Before any next quality claim, render a meaningfully larger pose-disjoint four-view train/val/test corpus from existing stored poses, with balanced viewpoint coverage and render-domain augmentation; compare frozen vs partial-finetuned Backbone, real independent 3D joint ground truth, calibrated per-view uncertainty and heldout known surface visibility. Only after that, test fully synchronized continuous Blender motion sequences. H3 generation still requires explicit separate approval.

## 2026-11-01 correction: Dino's limb colours were removed by the P1/P3 renderer

A user visual review detected a **material-domain bug**: P1 `render_four_view_visibility.py` used Blender Workbench `color_type="SINGLE"`, which turns the entire Dino into gray, unlike the original Cycles Reader training images.

### Verified cause

- Original `runs/dino/dataset_hr_v1/00001_front.png` has **blue left leg and red right leg**, plus smaller yellow/green toe colour accents.
- Original `assets/dino/rigged/rigged.blend` contains **NO `legpaint` vertex colour attribute**, but its Principled BSDF Base Color is linked to original image textures, including `Tyrannosaurus_albedo.jpg.001` and `Dinosaur__Diffuse.png`.
- `assets/dino/creature.json` has `paint: {}`: this source Dino's colour identity is **already in the image textures**, not in a newly generated procedural vertex-colour mask. Do not claim the `paint` configuration provides these colours.
- The P1 code used gray Workbench to test *only geometry*, but P3 unwittingly reused those rendered images to train a Reader originally trained on material-textured/limb-coloured images. This is an important additional domain shift and makes the earlier P3 smoke-quality conclusion **confounded**.
- The lower Part-ID visualization uses dominant Skinning Weight and is **not the same as the original diffuse texture's semantic paint**. Dino has 67 bones but only 54 Mesh Vertex Groups. Precisely 13 bones lack Skinning groups, including **8 terminal toe bones**, 4 finger-tip bones, and the last tail bone. Those bone IDs have no independently skin-owned surface and their `part_visibility` correctly remains UNKNOWN, rather than faking a coloured body part.

### Corrective implementation

- Updated `blender/render_four_view_visibility.py`: default `--appearance source-material`, using `common.studio(scene,samples=16)`, the same Cycles material lighting/render pipeline as the original Reader dataset. Optional `--appearance neutral-geometry` keeps old Workbench for geometry-only tests. No existing images overwritten.
- Recorded `appearance_mode`, `source_material_colors_preserved`, `render_samples` in new manifests.
- Updated `mocap/fourview_data.py`: track and reject mixed per-pose appearance modes.
- Updated `mocap/train_four_view.py`: refuse gray or legacy untagged training/evaluation data by default; `--allow-neutral-geometry` is explicit for reproducing old **gray-domain experiments only**.
- Updated `tools/audit_four_view_visibility.py`: support **full-resolution** 880px side images in the static contact sheet.
- Added a real rendered-image regression in `tests/test_iclone_fourview_reader_p3.py` to verify blue/red limbs are present when source materials are enabled.

### Evidence (existing pose 00001, no new pose/AI video)

| Appearance | Foreground high-saturation pixel share |
|---|---:|
| Original Cycles Front Reader image (400��720) | 51.03% |
| Old P3 gray Workbench Front (200��360) | **0.00%** |
| Corrected Cycles source-material Front (200��360) | **51.99%** |

Test directories:

```text
runs/dino/four_view_material_fix_test_v1/        # 0.5 scale, 4 static colour images
runs/dino/four_view_material_fix_fullres_v1/     # scale 1.0, 4 static colour images
```

The **full-resolution** corrected render was also independently audited: 4/4 cameras PASS, per-view Blender Raycast depth p95 **0.000002m**, Workbench/Cycles silhouette vs software depth IoU **0.963�V0.972**, camera��Blender GT projection max **0.0028px**. Exact 3D Pose is unchanged.

### Consequence for earlier P3 experiment

The old `runs/dino/dataset_fourview_p3_smoke_v1` and `runs/dragon/dataset_fourview_p3_smoke_v1` were **gray appearance experiments**. Their measured 2D/3D failures remain genuine measurements **of that gray-domain experiment** but must not be used to conclude that a properly material-coloured four-view Reader cannot improve motion.

**DO NOT promote old P3 smoke checkpoints.** For a valid repeat: rerender four views in NEW dirs, use `--appearance source-material --scale 1.0` (to protect tiny toe pixels) and preserve pose-disjoint train/val splitting; retrain both Creature Readers and verify 2D and independent 3D GT. No full rerender or retraining was carried out by this specific visual-bug correction.


## Material-correct full-resolution rerun COMPLETED (2026-11-01)

The earlier gray Workbench 0.5-scale P3 results above have been superseded **for evaluation of textured material-correct Reader training**, but remain archived as their own ablation.

**New authoritative run:** `docs/FOUR_VIEW_P3_MATERIAL_FULLRES_RERUN.md` with 28 pose-disjoint four-view samples per creature, proper Cycles original-material at `scale=1`, train/val 24/4 poses, separate Dino and Dragon high-res checkpoints and an independent 3D FK comparison.

- Correct source-material Dino original Reader 2V: **82.706mm**, 4V: **89.583mm**; new four-view Reader 2V **85.655mm**, 4V **84.003mm**; hybrid (original Front/Side + new Back/Opposite) 4V **81.159mm**.
- Correct source-material Dragon original Reader 2V: **61.510mm**, 4V **52.238mm**; new four-view Reader 2V **58.872mm**, 4V **49.699mm**; hybrid 4V **49.164mm**.
- The 4-view mini-finetuning improves average 2D errors but calibration and original-view nonregression are not universally satisfied.
- **No production promotion**: only 4 heldout independent static poses, no real synchronized motion, visibility/uncertainty uncalibrated. No new H3 video or iClone changes.
