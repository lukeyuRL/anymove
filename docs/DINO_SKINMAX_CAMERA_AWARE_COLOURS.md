# AnyMove Dino �X Camera-aware SkinMax colour coding (2026-11-02)

## Decision

**Do not impose a 20-colour limit.** Assign one immutable colour per *actual, nonempty* dominant-Skinning-Bone region useful to motion tracking, including tail and forearms/hands. Exclude zero-winning-weight Bones altogether. Preserve natural original head, spine, torso, eyes and mouth texture to avoid unnecessary synthetic appearance shift.

The final test palette has **41 unique colours** selected from the actual Dino Mesh:

| Category | Classes | Reason |
|---|---:|---|
| Both hind legs, thighs, calves, feet and toes | 22 | Tiny toe parts and left/right identity |
| Both forelimbs, arms, hands, finger bases | 12 | Arm motion and finger identity |
| Tail01 through Tail07 | 7 | Motion/curvature ambiguity along the articulated tail |
| **Total** | **41** | Classes derive from effective Skin Weight, not palette size |

The original body mesh contains **12,232 vertices**. **4,909 vertices** have a winning Skinning Bone in these 41 categories; **7,323 vertices** remain their original material. **13 Bones do not win any Vertex** (8 terminal toes, 4 terminal arm/finger bones, Tail08); these do not consume colours. The 13 effective body/head/eye/mouth/spine/neck Bones are intentionally unpainted, not deleted. Every Mesh Vertex has some Skinning influence.

The rule is exactly `winner(vertex) = argmax(weight of each positive skin group)`; in a tie, the lowest Vertex Group ID wins. Colour is stored as a **Blender POINT-domain FLOAT_COLOR** attribute `AnyMoveDinoViewAware`; non-selected vertices have alpha zero and retain the original albedo. All material slots use independent copied Materials with alpha mix over source Base Color. Adjacent vertex colours necessarily interpolate inside Mesh triangles; this is NOT face-wise hard segmentation. No Skinning Weights, rig bones or geometry are modified.

## Why camera-aware instead of a round-robin 41-colour sequence

One globally unique colour per Bone does **not** guarantee visually distinct neighbours. Front and Side are *orthographic* views; even different Bones can project into the same image region. Their painted surfaces cannot both be visible at one pixel: the front surface occludes the rear, which no colour coding can recover.

The actual training-pose conflict graph was generated from **32 P4 TRAIN poses**, not selected from blind tests, over all four camera depth/Part-ID NPZ maps. At a 10-pixel proximity threshold:

- Front: **356** distinct co-projected near part pairs over the training sample
- Side: **248**
- Back: **317**
- Opposite Side: **243**

These are counts of pairs observed near one another at least once, **not 356 overlaps in one frame**, not 356 mistakes and not only Skinning/mesh-adjacent pairs.

Graph weights combine (a) physical mesh-edge adjacency of `argmax(Skin Weight)`, (b) rig parent/child and siblings, (c) left/right counterpart regions, (d) tail curling proximity and (e) **actual 2D proximity from Part-ID**. Front/Side receive **2��** the weight of the opposite views. Colour optimization is a seeded, repeatable Lab-space heuristic, not a mathematically globally optimal arrangement or a measured improvement in Reader inference.

Original appearance anchors preserved: **left thigh blue `#2979FF`, right thigh red `#E53935`, left upper arm cyan `#00F0FF`, right upper arm yellow `#FFC850`, and distal Tail07 magenta `#FF28F0`**.

The final camera-aware palette is stored at:

`runs/dino/skinmax_camera_aware_v1/optimized_palette_constrained_v4.json`

The full 41 Bone��hex mappings, selected Vertex counts, parent topology and projection conflicts are in the JSON source reports.

## Actual validation, not theoretical guarantees

After vertex RGB assignment, three rendered Blender test poses (`9001,9002,9007`) were used for qualitative lighting diagnostics and **also to refine v4 colours**. Therefore these three poses must **not** be called statistically independent blind inference tests of the final palette; create new held-out poses for subsequent Reader A/B.

The P4 TRAIN-pose *theoretical* projection constraints under DeltaE76 <35: Front **1** near pair, Side **0**. Some Back/Opposite still require care. This threshold is a heuristic, not a psychophysical calibration.

Actual rendered colour medians measured in eroded frontmost Part-ID interiors on 3 poses, using a more conservative **DeltaE76 <25** threshold among pairs whose visible masks lie within 5 pixels:

| View | observed near pairs | pairs below DeltaE76 25 |
|---|---:|---:|
| Front | 172 | **0** |
| Side | 153 | **2** |
| Back | 141 | **3** |
| Opposite Side | 142 | **5** |

This is an approximation: surface Part-ID is by dominant *triangle* Skinning, while the actual material uses *vertex* argmax weights and interpolates; Blender's Cycles lighting further alters perceived colours. The two remaining observed Side low-contrast cases are **Left MidToe01 / Right Toe** and **Left Calvicle / Right Calvicle**. Do not claim the palette removes all visual confusion. A missing/occluded surface cannot be resolved by colours.

Structural validation PASS:

- **3 poses �� 4 cameras = 12 render stills**, Camera/Depth/Part-ID reprojection and Blender independent raycast PASS.
- All 12 NPZ `part_id`, depth and joint UV arrays are **numerically unchanged** against the original uncoloured source for the same 3 poses (no geometry modification).
- 41-class `argmax(weight)` colours survive saving and reopening the separate Blender file, with precisely 4,909 coloured vertices and 7,323 vertices using original source textures.
- Original `assets/dino/rigged/rigged.blend` has SHA256 `2b39bcf6ca0fe7ef3feb7601fe2b2d4923fe2351288582c66b8ae74963466256` and is unchanged.
- 5/5 targeted tests passed; no H3 video generation, no model retraining, no update to production iClone/Reader or motion.json, no Commit/Push.

## Inspect and reproduce locally

**Editable Rig copy, textures packed:**
`F:\anymove\runs\dino\skinmax_camera_aware_v1\dino_viewaware_skinmax_v4.blend`

**Original versus colour-coded Front/Side comparisons:**
`F:\anymove\runs\dino\skinmax_camera_aware_v1\test_render_v4\09001_front_ab_compare.png`
`F:\anymove\runs\dino\skinmax_camera_aware_v1\test_render_v4\09001_side_ab_compare.png`

**Audits:**
`runs/dino/skinmax_camera_aware_v1/camera_conflicts.json`
`runs/dino/skinmax_camera_aware_v1/rendered_camera_colour_audit_v4.json`
`runs/dino/skinmax_camera_aware_v1/test_render_v4/viewaware_render_geom_audit_v4.json`

**Code:**
`blender/analyze_dino_skinmax_camera.py`
`tools/build_dino_camera_conflicts.py`
`tools/optimize_dino_camera_palette.py`
`tools/refine_dino_camera_constrained.py`
`blender/dino_skinmax_camera.py`
`blender/render_four_view_visibility.py --viewaware-palette <palette JSON>`
`blender/save_dino_viewaware_skinmax.py`
`blender/verify_dino_viewaware_skinmax.py`
`tools/compare_dino_viewaware_camera_palette.py`
`tests/test_iclone_dino_skinmax_camera.py`

## Before using it for a new Reader

**Appearance/camera contrast is tested, model accuracy is NOT.** To establish whether it actually improves Dino 2D toe P95 or 3D FK reconstruction, use a matched A/B training experiment with equal data, exact pose splits, equal Reader capacity and hyperparameters, original versus SkinMax colours only. The newly created colour-coded Reader should then be evaluated on **new seed** Blender poses that were never used for colour optimization or model selection, plus on H3 synthetic video if explicitly approved. Also test augmented trials where the colour code is partially absent or drifts, because H3/AI video generators might not preserve 41 stable anatomy-specific colours.

No automated conclusion should be made that 41 colours are universally better than 20 or 8 until such a model A/B is run. The class count is free to change based on genuine Skinning region coverage and projection conflicts, not a target number.
