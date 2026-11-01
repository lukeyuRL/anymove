# Dino Occlusion-aware Motion Reconstruction — Experimental P2/P3 A/B

Branch: feature/dino-motion-quality-p123.
Status: opt-in diagnostics and ground-truth benchmark only.
**Do not turn on by default in iClone, CC, or AI video capture.**

## Contract: 2D observation omission is not animation-key deletion

For each frame \`t\`, camera \`v\`, joint \`j\`, we store an explicit
\`visible_mask[t,j,v]\` boolean. This is an **observation-selection decision**,
not a claim that every omitted keypoint is anatomically hidden.

If mask is false:
- ignore that view's measured 2D keypoint in Solver reprojection loss;
- interpolate only for the *IK / root initialization*, never as a new
  measured 2D observation; fallback to Rig REST projection if a joint is
  unanchored for every frame in a view;
- use the other view, joint limits, global kinematic and temporal
  regularization; missing X/Y may be underdetermined;
- write **all T frames** into the same backward-compatible \`motion.json\`.
  No Blender/iClone per-frame FK key is removed.

Explicit masking is implemented in
\`mocap/occlusion_observations.py\` and optional branch of
\`mocap/solve.py\`; without masks, the original Solver still runs.
All outputs are in a NEW experiment subfolder with individual input
SHA-256 references; no live training/model weights altered.

Masked commands require \`--no-align\`, since synchronously sampled
masks cannot be silently retimed. Explicit P1/P2 tracking multipliers,
legacy hind association and old Dragon visibility weighting cannot
be stacked onto this experiment.

The additional \`--masked-temporal-strength 2\` option applies an
**extra acceleration regularizer only around masked joints**. It is NOT
a trained biomechanical Motion Prior; do not claim temporal correctness
without independently verified continuous motion ground truth.

## Provenance and the geometry trap

\`blender/export_dino_mesh_visibility.py\` loads the **actual animated
evaluated Blender Dino Mesh** and independently checks all evaluated
world-joint coordinates against the existing Blender \`joints.json\`.
It uses Blender BVHTree line-of-sight raycasting against the deformed
triangle mesh at every frame.

Crucial: Bones and their joint endpoints are usually *inside*
skin. A naive camera-to-joint ray hits skin even when the pose is
visually estimable. The current experimental proxy therefore marks
a geometric obstruction only if the camera ray crosses **two or more**
mesh surfaces before the joint. To avoid overinterpreting a virtual
bone outside any skin, it requires proximity <=0.5 metres to a mesh
surface. Points lacking this support remain **Unknown**, not magically
visible or occluded.

This is **not real anatomical visibility GT**. The proxy is useful
to test a failure mode, not to supervise a production Visibility Head.
For this Dino, Tail08 has no direct skin Vertex Group and its tip is
~1.05 m from the nearest deformed skin. The geometric occlusion proxy
therefore leaves Tail08 Unknown, including the known catastrophic
frame 47 case where the Front 2D Reader error is ~71.5 px.
That is localization ambiguity/foreshortening, not confirmed mesh
occlusion.

## Observation selection variants

\`tools/make_dino_occlusion_ablations.py\` builds new NPZ files with
EXACT original Reader UV, confidence peaks, FPS and camera calibration;
only new bool \`front_visible_mask/side_visible_mask\` fields differ.

| Mode | Criterion for dropping ONE view observation | Deploy? |
|---|---|---|
| mesh | actual skinned Blender mesh ray surface-crossing proxy with support | No: synthetic oracle-like Blender-only signal, **not** validated anatomy |
| p2_error | P2 predicted localization error >= 10 layout px | No: uncalibrated for real video |
| mesh_p2 | union of geometric proxy and P2 high error | No |
| oracle_gt | independent Blender 2D positional GT error >= 10 px | **Never**: GT leakage upper-bound diagnosis |
| *_temporal | same mask plus extra masked-joint acceleration penalty | No: not a learned prior |

No approach deletes or creates Blender/iClone Keyframes.
Every A/B is scored against the SAME independent
\`gt_render/joints.json\` 3D world positions using
\`tools.evaluate_dino_full_body\`.

## Reproduce on authorized Windows computer

Use isolated clone and already-created Ground Truth baseline
\`F:\anymove_gt_eval_20261101\` (source \`F:\anymove\` stays untouched):

\`\`\`bash
cd /f/anymove_gt_eval_20261101
git pull --ff-only
export PYTHON=python.exe
export BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
export MOCAP_CREATURE=assets/dino
export DINO_OCCLUSION_ERROR_THRESHOLD_PX=10
export DINO_OCCLUSION_TEMPORAL_STRENGTH=2

bash tools/eval_dino_occlusion.sh \
  runs/dino/gt_experiments/seed7_smoke_24 \
  occlusion_mask_v1 \
  mesh,p2_error,p2_error_temporal,oracle_gt \
  runs/dino/reader_hr_quality_p2_eval_snapshot.pt
\`\`\`

Use a NEW tag if outputs already exist. For independently held-out
Seed 11 124-frame evaluation replace the first GT source with
\`runs/dino/gt_experiments/seed11_full_124\` and choose a NEW tag.
Once a mask is evaluated, inspect
\`ground_truth_comparison/full_body_report.json\`, \`by_joint.csv\`,
\`by_frame.csv\`, not the Solver's own per-run reprojection. Reprojection
for masked variants can include initialization-interpolated UV, so it is
**not** directly comparable to baseline.

Optional \`tools/audit_dino_occlusion_sources.py\` compares how often
mesh/P2 masks select high-2D-error observations against independent
Blender 2D GT; it does NOT label true anatomical self-occlusion.
\`tools/review_dino_masked_observations.py\` visualizes 2D observations
removed from Solver, while all animation frames remain present.

## Early grounded findings — Seed 7 / 24 frames

Anatomical group counts: Head/Neck 17, Torso/Root 5, Forelimbs 21,
Hindlimbs 6, Feet/Toes 34, Tail 9; none unclassified.

Blender mesh proxy: Front 303 / 2208 joint-frames, Side 598 / 2208
were marked blocked. Front flagged detections had mean actual
2D localization error **4.35 px**, compared with **1.20 px** for
supported/unflagged observations. But the mask also deleted entire
24-frame Side-view trajectories for several toe/hand joints and all
view observations for mouth in some frames. Therefore it is not safe
to interpret each ray crossing as a trustworthy observation-rejection
decision.

Against Blender 3D GT, Baseline MAE **3.351 cm** and the initial
mesh-geometry mask-only Solver MAE **6.492 cm** (substantially worse).
**Production blocker**: reject this geometry proxy as an automatic
mask until per-joint visibility supervision is independently validated.

P2 Reader error threshold 10 px produced Front 48 and Side 1 masked
joint-frames. In Front it selected 41/48 genuine >=10 px errors
(precision 85.4%) and 41/57 of all such errors (recall 71.9%).
Despite selecting bad 2D points, the no-extra-temporal P2 hard-mask
Solver scored full-body **3.356 cm** vs Baseline **3.351 cm**;
Tail **6.512 cm** vs Baseline **6.410 cm** (slightly worse).
A correct missing-observation decision alone does not reconstruct
the unique 3D X coordinate lost when Front is discarded.

**These are controlled synthetic results, not natural Dino AI video
visibility accuracy or real-world Motion Quality.** Do not infer that
a high synthetic-cutout AUC validates real self-occlusion labels.
