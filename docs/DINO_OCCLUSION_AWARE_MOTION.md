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


## Completed Seed 7 / 24-frame missing-observation A/B results

**Actually run** in isolated authorized Windows clone with Blender 5.0,
local Dino rig, 24fps Seed 7 joint GT, the same stable P2 Reader snapshot.
All 92 joints and all 24 animation frames are retained for each variant.
Nothing in the source F:\anymove project or Reader training is modified.

### True 3D world-position errors (Blender GT)

| Variant | Full body mean (cm) | Full body P95 (cm) | Tail mean (cm) | Head/neck mean (cm) |
|---|---:|---:|---:|---:|
| Baseline | **3.35111** | 9.37482 | **6.41011** | 4.02197 |
| Mesh crossing mask | 6.49171 | 18.10352 | 7.96891 | 14.94065 |
| P2 predicted error >=10px hard omission | 3.35614 | **9.26248** | 6.51207 | 4.01389 |
| P2 hard omission + local temporal acceleration ×2 | 3.43522 | 9.50044 | 7.33965 | 4.01683 |
| Blender GT error>=10px oracle omission | 3.36494 | 9.23340 | 6.69437 | **3.97410** |

All scores use the **identical independent Blender ground truth**, not
the masked Solver's own reprojection, which may include imputed UV.

**Conclusion**: Neither mesh proxy, calibrated-ish P2 error threshold,
extra local acceleration penalty, nor even a 2D-ground-truth Oracle
improved overall mean 3D reconstruction. P2 hard masking did slightly
lower global P95, but Tail worsened. The Oracle is an unattainable
GT-leakage diagnostic and still worsened overall/Tail mean, so the
primary problem now is 3D information recovery under absent camera-axis
observations, not merely improving occlusion classification.

### Actual observation-selection precision vs known large 2D error

All "precision/recall" values here use *actual 2D position error >=10px*
as a diagnostic target, **NOT true mesh self-occlusion**:

| View, mask | Excluded | Actually >=10px | Correct exclusions | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Front, Blender mesh proxy | 303 | 57 | 24 | 7.92% | 42.11% |
| Side, Blender mesh proxy | 598 | 2 | 1 | 0.17% | 50% |
| Front, P2 error proxy | 48 | 57 | 41 | **85.42%** | **71.93%** |
| Side, P2 error proxy | 1 | 2 | 0 | 0% | 0% |

The mesh proxy has terrible *high-localization-error selection
precision*, especially in Side. In particular the Side geometry proxy
rejected **329 foot/toe observations** despite **zero** corresponding
>=10px position errors. This explains why a naively correct "ray
intersects skin" heuristic can destroy reliable motion evidence.
Do NOT relabel this as an anatomical visibility metric.

### Tail08 geometry and missing-X diagnosis

The Tail08 bone tip lacks a matching skinned Vertex Group and is
~1.05m away from the nearest Dino mesh surface, so a mesh-based
per-joint occlusion tag is **Unknown** for it, not "visible" or
"masked." It was never identified as blocked by the mesh gate.

In independent Seed 11 124-frame GT, Frame 47 Tail08:
- Baseline 3D error decomposes into X -66.77 cm, Y +9.08 cm,
  Z -18.18 cm (69.79 cm norm).
- P2 soft-weighted X error worsened to -73.63 cm (76.33cm norm).
- Front uniquely provides direct X information under current camera
  geometry. Removing its observation without another way to estimate
  X (synchronized Top/elevated camera, verified dynamic prior) cannot
  be assumed to reconstruct the true Tail08 lateral trajectory.

### Explicit opt-in and leakage safety gates

A masked detection NPZ requires:
- a matching SHA-256-bound \`*.occlusion_input.json\` file;
- \`--experimental-masked-observations --no-align\` in the Solver;
- GT oracle additionally requires \`--allow-gt-oracle\`, **NEVER**
  in production AI-video captures.

The experiment runner sets these flags only in its isolated QA modes.
The ordinary Reader/Solver/iClone path is unchanged. Source
detections/motions are never overwritten; no animation key gets deleted.

### Files created and QA

Under
\`F:\anymove_gt_eval_20261101\runs\dino\gt_experiments\seed7_smoke_24\occlusion_mask_v1\`:

- \`mesh_geometry.npz\` / \`mesh_geometry.mesh_geometry.json\`
  (deformed Blender raycast and per-joint skin support, provenance SHA)
- \`det_{mesh,p2_error,p2_error_temporal,oracle_gt}.npz\`
  (input Reader UV/peaks unchanged, boolean observation masks)
- \`motion_*.json\` / \`motion_*.occlusion.json\`
  (all 24 frames with masked observation provenance)
- \`mask_review_mesh.mp4\` and \`mask_review_p2.mp4\`
  (green kept 2D observations; red X omitted; purple ring skin UNKNOWN)
- \`source_selection_report.json\` (post-hoc 2D GT selection audit)
- \`ground_truth_comparison/full_body_report.json\`,
  \`by_joint.csv\`, \`by_frame.csv\` (ground-truth 3D comparison)

**Next acceptance gate:** do not optimize thresholds against this one
smoke Seed. First add a credible source of unique X-axis constraints
and test independently on Seed 11 + natural Dino motion. A mere
smoother curve or missing keyframe interpolation cannot substitute
for a trained Motion Prior or synchronized additional camera.


## Independent Seed 11 / 124-frame source-selection validation

**Completed on the authorized user's computer** with the same frozen Dino
Reader and independent Blender animation Seed 11, 124 frames, 24 FPS.
No additional 3D mask solve was run because the Seed 7 full-body
mask/Oracle A/B already failed its initial acceptance gate.
This run evaluates only source-selection consistency, not Motion quality.

Blender evaluated mesh geometry is saved in
\`runs/dino/gt_experiments/seed11_full_124/occlusion_sources_v1/\`,
with \`mesh_geometry.npz\`, SHA-bound geometry sidecar,
\`source_selection_report.json\` and the complete Blender raycast log.

Post-hoc high 2D error >=10 layout px selection audit:

| View / predictor | Flagged | True >=10px errors | True positives | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Front mesh crossing proxy | 1654 | 193 | 88 | **5.32%** | 45.60% |
| Front P2 expected error | 166 | 193 | 142 | **85.54%** | **73.58%** |
| Side mesh crossing proxy | 3147 | 18 | 13 | **0.41%** | 72.22% |
| Side P2 expected error | 29 | 18 | 0 | **0%** | 0% |

The very low Side high-error selection precision demonstrates that
mesh obstruction is a **different target variable** than localization
error. It MUST NOT be mistaken for anatomical visibility classification
accuracy; a keypoint can lie behind surface layers yet still be read
correctly by the network. Neither target has actual labeled,
human-verified Dino mesh self-occlusion GT in these experiments.

Special cases:
- Tail08 nearest mesh distance 1.05369m, **Unknown all 124 frames**;
  geometric proxy never marks it occluded in either view.
- Tail07 has 30 Front geometrically blocked frames, none in Side.
- LeftToe03 has 95 Side geometric blocked frames even though only
  **8 total >=10px Side errors** exist across *all 34 toe joints*.
- Front Tail GT error remains the dominant high-error region. P2
  flagged 160 Front Tail joint-frames, 142 actually >=10px (precision
  88.75%; recall 74.74% of 190 GT high-error Tail cases).

Safety validation executed (not just unit-tested): 
\`mocap.solve\` refused a mask-bearing NPZ with no
\`--experimental-masked-observations\`, and independently refused a
\`oracle_gt\` mask lacking explicit \`--allow-gt-oracle\`;
neither command wrote a motion file. Regressions and Video QA outputs
remain in the isolated clone.

**Go/no-go:** P1 guard OFF, mesh occlusion hard mask OFF,
P2 hard mask OFF. Keep P2 error output for diagnostics and future
joint-specific robust weighting; next substantive work is independent
dynamic 3D constraints (synchronized Top or learned motion prior) rather
than deleting additional 2D/animation observations.
