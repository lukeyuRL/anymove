# Dino SkinMax41 vs Original Material �X Controlled Reader A/B Smoke

Date: 2026-11-02. Repo F:\anymove, branch feature/dino-motion-quality-p123.

**Status: Complete, A/B smoke negative. Do NOT promote 41-colour Reader or replace production checkpoint.**

## Goal and controlled protocol

Does 41-class anatomy-aware paint applied to each Dino Vertex according to its maximum positive Skinning Weight improve 2D Keypoint localization and 3D FK reconstruction, especially toes, forearms/hands and tail?

- A: Original Dino source albedo and legacy blue/red limb colours.
- B: Same source, rig, geometry and four orthographic views, except an additional 41-class view-aware SkinMax colour layer.
- Train 24 poses (IDs 1�V24), validation four poses (IDs 25,50,75,100).
- New evaluation 12 static poses (IDs 11001�V11012, Blender Seed 20261221), not used in paint optimization, Reader training or checkpoint selection.
- Original palette optimization used 32 separate P4 TRAIN poses and earlier diagnostic renders 9001,9002,9007. New evaluation is disjoint.
- Both Cycles renders use source-material, 16 samples, image scale=1.0 and identical camera matrices, motion-free stills.
- Both Readers use original pretrained runs/dino/reader_hr_v1.pt, exact checkpoint SHA256 62e1c1da8d7614c0ecab6cf7a2343d8fb3e30e0690ff3f0f9104898de94645c7.
- Reader config 672x672/192x192 heatmap; frozen DINOv2 backbone; decoder+mesh-surface Quality Head updated, five epochs, batch two, AdamW LR 0.0002, Quality weight 0.2, Seed 74111, four views and identical train/val split.
- Both FK runs use existing P2 N-camera solver, identical rig/priors, 240 optimization iterations. Solver does not read GT. 3D GT only read in independent evaluator.

The dataset contract verified all paired 28 train/val plus 12 new-evaluation poses: identical rig and source metadata hashes, cameras, depth, Part-ID and projected 2D GT. Actual RGB differences were detected in all four cameras.

The 41 colour classes are chosen from 4,909 of 12,232 source Mesh vertices, where the winning Skinning Bone belongs to 22 hind-leg, 12 forelimb and 7 tail regions. The other 7,323 vertices preserve the source albedo. Thirteen Bones with zero winning skin-weight vertices have NO distinct colours. Vertex assignment uses argmax(SkinWeight) with deterministic group tie-breaks. Colours interpolate at Mesh faces.

## Matched training Validation (2D px)

| View | A Original | B SkinMax41 |
|---|---:|---:|
| Front | **4.034** | 16.225 |
| Side | **2.781** | 5.743 |
| Back | **10.813** | 26.966 |
| Opposite Side | **7.815** | 9.937 |
| All view mean | **6.361** | 14.718 |

Before this five-epoch fine-tuning, the same original Reader checkpoint measured 8.509 px on original-material A images and 37.035 px on B painted images. The colour layer thus makes a strong out-of-distribution input for the original reader. B improves during its five epochs, but does not catch A.

## New-seed 12-pose 2D GT test

Toe Keypoint mean / P95 error in original Camera Layout px:

| Camera | A Original | B SkinMax41 |
|---|---|---|
| Front | **2.333 / 5.952** | 4.212 / 12.979 |
| Side | **2.240 / 5.715** | 5.098 / 13.993 |
| Back | **5.722 / 21.489** | 11.846 / 53.699 |
| Opposite Side | **6.062 / 16.788** | 8.875 / 25.652 |

The full JSON independently reports Body, Hindlegs, all Toes, Toes with SkinMax colour, Toes without dominant weight, Forelimb/Hands, all Tail, Tail with colour and no-colour terminal Tail.

## New-seed 12-pose 3D FK GT test (mm)

| Metric | A Original | B SkinMax41 |
|---|---:|---:|
| Two-view Mean Joint Error | **86.344** | 150.999 |
| Four-view Mean Joint Error | **67.177** | 139.740 |
| Four-view Joint Error P95 | **184.225** | 376.373 |
| Four-view Toe Mean Error | **57.429** | 113.226 |
| Four-view Toe with winning SkinMax Bone | **43.830** | 91.937 |
| Four-view Hindlegs Mean Error | **59.503** | 92.280 |
| Four-view Forelimb/Hands Mean Error | **62.482** | 200.507 |
| Four-view Tail Mean Error | **110.257** | 254.048 |

Paired 4-view B minus A mean error = +72.563 mm, so B is worse. Zero of 12 poses improved. Pose-level paired bootstrap 95% interval: +46.042 to +102.835 mm.

Paired 2-view B minus A mean = +64.655 mm; one of 12 poses improved; bootstrap 95% interval +34.348 to +94.475 mm.

Both branches individually benefit from adding more Camera observations compared to their own two-view results, but B remains worse. A four-view from 86.344 two-view to 67.177 four-view, while B from 150.999 two-view to 139.740 four-view.

Independent tools/eval_multiview_gt.py audits for A and B both passed camera reprojection, GT source, FK roundtrip and joint-order checks.

## Technical interpretation and limits

**The 41-colour overlay hurts detection and FK in this specific 24-pose, five-epoch, frozen-DINOv2-backbone training regime. It should not be promoted to production.**

This is NOT proof that all colour coding is detrimental. The original pretrained Reader expects existing albedo patterns and original blue/red leg identity. The B images are initially strongly out-of-distribution; 24 training poses with a frozen Backbone may be insufficient. Also 41 hues may induce excessive artificial high-saturation boundaries and remove useful source texture gradients. These explanations are hypotheses: this A/B did not isolate individual mechanisms.

The new 12 poses are procedurally generated independent static Blender states, NOT continuous motion or H3 video. No claims of temporal tracking or H3 generation generalization are supported.

Recommendations for a next, separately controlled A/B: preserve the effective original colour identity and most source textures, apply more subtle 20�V60% alpha colour coding to specific tiny Toe/Hand/Tail regions, mixed-colour training (original plus treatment data) with much broader pose coverage, or a small adapted Backbone. Any future changes require matched Reader A/B, new independent Seed and separate 2D Toe/Hand/Tail P95 and 3D FK evidence. Do not continue training only B more epochs and call it a matched A/B.

## Local artifacts

Main: F:\anymove\runs\dino\skinmax_camera_aware_v1\

- Original A train/val source: F:\anymove\runs\dino\dataset_fourview_p3_material_fullres_v1\
- Colour B train/val source: ab_train_val_color41_v1\
- New source 3D GT: ab_blind_new_pose_v1\
- New test A images: ab_blind_A_original_v1\
- New test B images: ab_blind_B_color41_v1\
- Original A trained model: reader_A_original_5ep.pt
- 41-colour B trained model: reader_B_color41_5ep.pt
- Both model validation records: corresponding .validation.json files
- Paired-source pose/camera/GT data check: ab_dataset_contract_v1.json
- Grouped 2D and 3D A/B, paired bootstrap: ab_result_v1.json
- Independently checked FK and GT: ab_fk_gt_A_audit_v1.json and ab_fk_gt_B_audit_v1.json
- Source JSON palette: optimized_palette_constrained_v4.json
- Evaluation script: F:\anymove\tools\eval_dino_color41_ab.py
- Regression tests: F:\anymove\tests\test_iclone_dino_color41_ab.py

No H3 video, no model overwrites, no changes to production iClone output, no Git Commit/Push.

## Quality gate

A/B promotion to production: **FAIL**.

All 4 Dino A/B tests and all 5 camera-aware colour/geometry regression tests passed. Code compile and git diff --check passed.
