# AnyMove P3 �X Source-material full-resolution four-view rerun (2026-11-01)

**Repo:** `F:\\anymove`, branch `feature/dino-motion-quality-p123`  
**Status:** Source-material dataset + separate checkpoint training + independent FK audit COMPLETE. Production Reader PROMOTION **NOT APPROVED**.  
**User-visible cause:** Older four-view P3 used Blender Workbench `SINGLE` gray appearance and erased original red/blue leg and toe texture information. This rerun repairs that material domain and the 0.5��1.0 image scale. Both change simultaneously, so it is **not a single-variable colour ablation**.

## 1. Inputs and provenance

Uses existing pose metadata and 3D GT, **no new AI/H3 videos or motion sequences**:

| Creature | Original pose metadata | Original Reader checkpoint | 3D keypoints |
|---|---|---|---|
| Dino | `runs/dino/dataset_hr_v1/` | `runs/dino/reader_hr_v1.pt` | 92 |
| Dragon | `runs/dragon/dataset_hr/` | `runs/dragon/reader_hr.pt` | 50 |

New four-view Blender renderer uses `--appearance source-material --scale 1 --samples 16`, i.e. source texture/material and Cycles; preserves original red/blue leg/toe colours rather than gray Workbench. Rerendered each of **24 Train poses (IDs 1..24)** and **four held-out Validation poses (25, 50, 75, 100)**. A source pose's four camera views all stay in the same split. **112 still images per Creature**, with Front, Side, Back, Opposite Side camera geometry plus ground-truth 3D joints and per-View Mesh Skinning Surface Part Visibility.

New source material datasets:

```text
runs/dino/dataset_fourview_p3_material_fullres_v1/
runs/dragon/dataset_fourview_p3_material_fullres_v1/
```

Independent integrity check `tools/audit_p3_source_material_dataset.py` passes both 28-pose datasets:

- 112 RGBA images per Creature, with matching Depth and Part-ID NPZ, verified pose source SHA256 and rig SHA256.
- Zero mixing between `source-material` and `neutral-geometry`.
- Dino mean high-saturation *foreground* share: Front **45.63%**, Side **32.15%**, Back **48.96%**, Opposite Side **32.89%**; the prior gray Workbench experiment had 0% in the Front.
- Dragon mean high-saturation foreground share: Front **32.89%**, Side **31.78%**, Back **35.18%**, Opposite Side **32.72%**.
- Blender independent raycast vs software Depth P95 <= **4��10??m** for both datasets.
- Max original Blender 3D pose replay deviation: Dino **0.170mm**, Dragon **0.040mm**.
- 13 Dino bones have no independent Skinning Vertex Groups, including 8 terminal toes, 4 fingers and one final tail bone; their surface-part visibility must remain `unknown`, not hard-coded as occluded or visible.

### Controlling the experiment

- Reader: pretrained high-resolution `672��672 / heatmap 192`, **frozen DINOv2 Backbone**; Decoder and new visibility/expected-error Quality Head fine-tuned only.
- `3 Epochs`, microbatch `2`, Windows Python `3.14.3`, RTX `3070 8GB`, same LR `0.0002`, quality weight `0.2`, seed `20261101` as previous gray smoke, so results are directly comparable to a **same-dataset original Reader**, with prior gray P3 reserved as domain-shift evidence.
- Baseline Reader and new Reader are evaluated on identical **four held-out Blender 3D poses**. No withheld 3D GT is present in the optimization input observation NPZ.
- P2 N-camera FK Solver is identical for all branches: same `240` optimization steps, same Rig joint limits (unvalidated default limits remain a caveat), same camera projection. `2V`=Front+Side, `4V`=Front+Side+Back+Opposite.
- Reader output is color-dependent, **Mesh Surface Part-ID/Depth is not an RGB segmentation model**, and the Quality Head scores are *uncalibrated*. The Solver never reads the GT Mesh visibility oracle or uses uncalibrated Quality predictions as weights.

## 2. Mean 2D localization error on original Camera Layout pixels (px)

| Creature | View | Original Reader | Source-material P3 Reader | Delta (new-original) |
|---|---|---:|---:|---:|
| Dino | Front | 4.121 | **3.807** | -0.314 |
| Dino | Side | **2.661** | 2.956 | +0.295 |
| Dino | Back | 16.223 | **11.951** | -4.273 |
| Dino | Opposite Side | 11.031 | **7.926** | -3.105 |
| Dino | **Four-view mean** | 8.509 | **6.660** | -1.849 |
| Dragon | Front | **5.898** | 6.734 | +0.836 |
| Dragon | Side | **6.450** | 6.807 | +0.357 |
| Dragon | Back | 33.795 | **29.133** | -4.662 |
| Dragon | Opposite Side | **13.945** | 14.176 | +0.231 |
| Dragon | **Four-view mean** | 15.022 | **14.212** | -0.810 |

**Interpretation:** textured 4-view Reader mini-finetuning helps the newly added views, especially Dino, but regular Front/Side may regress, more noticeably for Dragon. This is already a meaningful improvement over the gray P3 smoke, but the latter differed in both appearance and resolution and is NOT a valid one-factor ablation.

## 3. Independent true 3D FK errors, four held-out poses (mm)

Every row is an actual 3D joint-position mean Euclidean error against separately stored Blender Ground Truth. The same 2D Reader observations are used for both 2V and 4V branches; training and Solver never see the withheld 3D targets.

| Creature | Reader arrangement | 2V | 4V |
|---|---|---:|---:|
| Dino | Original Reader on all views | **82.706** | 89.583 |
| Dino | P3 fine-tuned Reader on all views | 85.655 | 84.003 |
| Dino | **Hybrid**: Original Front+Side, New Back+Opposite | **82.706** | **81.159** |
| Dragon | Original Reader on all views | 61.510 | 52.238 |
| Dragon | P3 fine-tuned Reader on all views | 58.872 | 49.699 |
| Dragon | **Hybrid**: Original Front+Side, New Back+Opposite | **61.510** | **49.164** |

**Key evidence:**

- Dino Hybrid 4V reduces original 2V error by **1.547mm** (82.706��81.159), small and insufficient to claim stable superiority.
- Dragon Hybrid 4V reduces original 2V error by **12.346mm** (61.510��49.164), promising on these heldout still poses.
- Hybrid copies **exactly** the original Front+Side UV/Confidence/valid and **exactly** the adapted Back+Opposite observation arrays. Camera matrices, keypoint names and frame IDs match; the two-view Hybrid control yields identical world joints to original Reader two-view run.
- The hybrid packet contains **no 3D GT, Mesh Visibility oracle or uncalibrated Quality scores**. Only independent posthoc `eval_multiview_gt.py` accesses 3D GT. Data audit confirms source hashes.
- New Renderer appearance provides a dramatic domain shift correction; do not extrapolate improvement to independently AI-synthesized four-view H3 videos until synchronized camera/appearance validation.

### Critical scope restriction

These are **four independent static poses** per Creature, not a synchronized 124-frame motion sequence or a novel Creature; current N-camera Solver has no evidence of motion plausibility, temporal tracking, real-world camera synchronization or consistent H3 multiview generation. Do NOT infer a general 4-view Motion quality win from 1.5mm on four Dino poses.

## 4. Mesh Surface Visibility and expected-error calibration

Independent validation against heldout *Mesh Surface Part* labels, excluding UNKNOWN:

| Creature | View | AUC | Brier | ECE | Predicted-vs-real localization gap (px) |
|---|---|---:|---:|---:|---:|
| Dino | Front | 0.829 | 0.146 | 0.233 | 0.71 |
| Dino | Side | 0.823 | 0.134 | 0.203 | 0.90 |
| Dino | Back | 0.786 | 0.143 | 0.124 | **7.26** |
| Dino | Opposite Side | 0.884 | 0.118 | 0.203 | 3.37 |
| Dragon | Front | 0.721 | 0.160 | 0.200 | 3.21 |
| Dragon | Side | 0.827 | 0.155 | 0.187 | 4.04 |
| Dragon | Back | 0.713 | 0.156 | 0.260 | **25.80** |
| Dragon | Opposite Side | 0.803 | 0.147 | 0.270 | 10.87 |

Visibility quality is **not calibrated**, and the Back predicted localization errors are systematically unreliable, especially Dragon. No automated quality weighting is enabled in the production or N-camera Solver.

`tools/eval_p3_source_material.py` verifies original checkpoint hashes, pose split and GT separation, then fails production promotion if any required original-view regression, 3D baseline, visibility calibration or uncertainty quality condition fails. Both creatures currently have `production_promotion_allowed=false`.

## 5. Files on the machine

Implementation:

```text
blender/render_four_view_visibility.py          # corrected source-material Cycles renderer
mocap/fourview_data.py                          # source-material mixing protection
mocap/train_four_view.py                        # original Reader warmstart and 4-view training
tools/audit_p3_source_material_dataset.py       # source/data/material integrity
tools/build_four_view_observations.py            # original / new P3 Reader inference
tools/eval_multiview_gt.py                      # independent 3D GT / FK audit
tools/eval_p3_source_material.py                # same-dataset 2D/3D/uncertainty acceptance
tools/blend_fourview_reader_p3.py               # optional 2-view original + 2-view adapted A/B
tests/test_iclone_p3_source_material_fullres.py
tests/test_iclone_p3_hybrid_reader.py
```

New experimental model weights and acceptance:

```text
runs/dino/reader_fourview_p3_material_fullres_v1.pt
runs/dragon/reader_fourview_p3_material_fullres_v1.pt
runs/dino/fourview_p3_material_fullres_acceptance_v1.json
runs/dragon/fourview_p3_material_fullres_acceptance_v1.json
```

Per-pose image labels, 2D observation packets, 2V/4V FK outputs and independent 3D evaluation reports live under `runs/{creature}/dataset_fourview_p3_material_fullres_v1/`.

Datasets/checkpoints from the earlier **gray** runs remain unchanged and tagged as legacy ablations. Original Reader checkpoints are unchanged and their SHA256 hashes are verified by the acceptance tool.

## 6. Verified tests and release decision

- P3 fullres dataset and checkpoint regression: **4/4** tests pass.
- P3 inherited Reader tests: **9/9** pass.
- P2 camera / selected-view isolation tests: **9+5** pass.
- P1 geometry labels / artifact tests: **7+3** pass.
- Hybrid exact source/fusion & independent GT audit: **4/4** pass.
- **41/41 targeted tests pass**. `python -m compileall` and `git diff --check` pass.
- No paid H3 videos generated, no iClone import output modified, no production Reader overwritten, no Git Commit or Push performed.

**Recommendation:** keep production Front/Side Reader intact. Continue testing Hybrid four-view as the most plausible risk-controlled path, but do **NOT** deploy before (a) a much larger pose-disjoint four-view dataset, (b) visibility/uncertainty per-view calibration, and (c) synchronized motion sequence and iClone import checks. The current success is limited to P3 static smoke.
