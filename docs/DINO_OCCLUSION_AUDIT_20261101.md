# Dino Occlusion-aware Reconstruction — Local A/B 結果與下一階段決策

Status: **受控 Blender 實測完成；目前所有 Occlusion Mask 都維持 opt-in，未通過啟用驗收**。

Last verified: 2026-11-01, desktop `LukeYu-2024`, isolated clone
`F:\anymove_gt_eval_20261101`.
Working tree `F:\anymove`, original Reader / P2 checkpoints, Blender Dino mesh,
original `motion.json`, and all 2,200 static-pose samples were not overwritten.

## 1. Precisely what was tested

- Controlled Blender ground truth: **Seed 7, 24 frames, 24fps, 92 keypoints, 67 bones**.
- Same frozen 672/192 original Reader, same raw front+side video and known Blender evaluated 3D joint positions.
- Independent Blender ray intersections for Mesh mask. A joint is *blocked-proxy* if its ray crosses **at least two** mesh surfaces AND the joint is within 0.5m of its closest skinned surface.
- P2 head snapshot: `85d51371b8667eb7a2fd465cc85bfb3e28a6398285b78611e66f854dde289458`. P2 *expected error* threshold = 10 layout px, **not** synthetic-cutout visibility probability.
- New `mocap/occlusion_observations.py` and opt-in `mocap.solve`: masked per-view 2D data term receives zero weight, missing pixel UV only interpolated to initialize IK; no animation frames or final FK Keys deleted. No P1/P2 tracking weights stacked with missing masks.
- `p2_error_temporal`: same mask plus 2× additional local acceleration penalty around missing observations. This is *not* a learned Motion Prior.
- `oracle_gt`: mask only 2D measurements whose error against Blender 2D GT is >=10px. **This arm leaks GT and cannot be deployed.** It tests this one rule's performance given ideal erroneous-measurement selection; it is **not** the optimal possible 3D reconstruction Oracle.

Source: local experiment
`runs/dino/gt_experiments/seed7_smoke_24/occlusion_mask_v1/ground_truth_comparison/`.
Files `full_body_report.json`, `by_joint.csv`, `by_frame.csv`. Geometry metadata
`mesh_geometry.mesh_geometry.json`; raw weight/mask provenance sidecars and solver logs also stored there.
The local results are NOT uploaded to GitHub.

## 2. Ground truth 3D results

MAE / P95 / Max are distance to **independent Blender 3D truth** in centimetres,
not self-reprojection or distance between two Solver outputs.

| Experimental arm | Whole-body MAE (cm) | P95 (cm) | Tail MAE (cm) | Tail P95 (cm) | Root GT MAE (cm) |
|---|---:|---:|---:|---:|---:|
| Original Baseline | **3.35111** | 9.37482 | **6.41011** | **16.86528** | 2.02491 |
| Mesh-proxy Mask | **6.49171** | 18.10352 | 7.96891 | 26.47182 | 3.16420 |
| P2 Error Mask (>=10px) | 3.35614 | **9.26248** | 6.51207 | 19.37590 | 1.98387 |
| P2 Error Mask + temporal 2× | 3.43522 | 9.50044 | 7.33965 | 23.83384 | **1.95354** |
| Oracle GT Error Mask (>=10px; truth leakage) | 3.36494 | 9.23340 | 6.69437 | 19.87374 | 1.91307 |

**Every tested hard-masking variant has worse global MAE than Baseline.**
Mesh-proxy badly fails; P2 high-error rejection and GT oracle selection are
near baseline but **do not recover missing Tail X**. Extra acceleration
smoothing degrades Tail further. P95 and Root may shift favorably for some
arms; that does not negate failure of the stated full-body/Tail accuracy goal.

## 3. Why the visibility premise failed

A Blender bone joint is usually *inside* skin, so even an accurately localized
bone can have one or multiple mesh intersections from a camera. Ray-cast
intersection count is only a geometric **proxy**, not human-annotated
keypoint visibility or a calibrated P(visible).

At Seed 7 (24 × 92 = **2,208 joint-frames per view**):

| 2D diagnostic | Front mesh mask | Front P2 error mask | Side mesh mask | Side P2 error mask |
|---|---:|---:|---:|---:|
| Observations dropped | 303 | 48 | 598 | 1 |
| Dropped with true 2D error >=10px | 24 | 41 | 1 | 0 |
| Drop precision for >=10px errors | **7.9%** | **85.4%** | **0.2%** | **0%** |

The poor mesh proxy precision explains why deleting these observations
strongly destroys Tongue/Head, Feet/Toes and other FK estimates.
For example Tongue02 mean 3D GT error increases from baseline ~8.60cm to
Mesh Mask **55.48cm**.

### Virtual bone tip that cannot be treated as an image landmark

Tail08 terminal keypoint is approx. **1.053m from the nearest evaluated
Dino mesh surface**, with no direct matching skinned Vertex Group. It is
a kinematic endpoint extending outside the skinned silhouette, NOT a
directly observable skin landmark. The geometry support flag is UNKNOWN
for all 24 frames, so geometry-based masks rightly refuse to claim
it was visible or occluded. Simply adding an extra view **does not make
an off-mesh virtual endpoint directly visible**; a surface landmark and
rig kinematics are required.

Other distant/internal rig endpoints include Hip, early Spine and Neck
joints. Camera visibility, bone endpoint observability and prediction
error are **different variables**; do not merge them.

## 4. What actually generalizes across tested synthetic seeds

For the same P2 threshold, with 10px *actual 2D Reader error* as the
diagnostic target (these are not labeled true mesh occlusions):

| View / measurement | Seed 7 (24f) | Seed 11 (124f) |
|---|---:|---:|
| Front P2 error mask precision | 85.4% | **85.5%** |
| Front P2 error mask recall | 71.9% | **73.6%** |
| Front true >=10px outlier observations | 57 | 193 |
| Front P2 mask observations | 48 | 166 |
| Side P2 mask correctly selected >=10px outliers | 0 | 0 |

This supports **P2 as a Front outlier detector**, particularly for the
Tail, on the tested Blender-generated domain. It does NOT prove real AI
video generalization. Side P2 hard gating is actively unpromising here.

Because these findings were inspected across **both** seeds, a new held-out
continuous-motion seed (not Seed 7 or 11) is required to validate any
new tail-only masking rule and avoid a selection-bias claim.

## 5. The next engineering milestone — observability-aware 3D reconstruction

**Do not keep tuning a global Visibility Mask or simple acceleration weight.**
Make the following work conditional on 3D GT acceptance:

1. **Explicit observation semantics.** For each Keypoint store
   `observed_skin_landmark | virtual_fk_endpoint | internal_bone`;
   plus `visible | occluded | unsupported | unknown` per view/frame.
   Derive virtual-tail FK endpoints from visible skin/tail curvature and
   pose/bone constraints; do not expect DINOv2 to see a point >1m outside
   the mesh. Do not mistake a renderer's 2+ mesh intersections for GT
   occlusion.
2. **P2 predicted localization-error handling.** A candidate
   `Front/Tail-only` inlier gate (Side unchanged) is motivated by high
   Front P2 precision and zero Side precision; DO NOT promote it without
   a NEW independent GT seed. Retain soft-weight and robust-loss
   baselines. Avoid using measured Blender GT at inference.
3. **Proper missing-view inverse kinematics.** Solve 3D with
   independently calibrated per-camera projection residuals,
   always-visible observations and bone constraints; use a pose
   initializer or prior that does not assume an unobserved virtual
   endpoint's current 2D coordinate is available. A simple interpolation
   of missing 2D images is only a temporary initializer and not 3D truth.
4. **Real motion prior before hard deletion.** Train/test on >=2 distinct,
   independently verified continuous Dino animations with exact rig/FPS
   (do not treat independent 2,200 still images as a time series).
   Use learned bone/axis angular dynamics and a robust 2D measurement
   likelihood. Evaluate Tail08, Tongue03, Toe03 plus Root at omitted
   single frames vs contiguous 3/6/12-frame gaps; emphasize 3D
   MAE/P95, velocity and contact errors, not visual smoothness.
5. **Top vs Elevated camera geometry:** Only after the above, use the
   **same** Blender GT animation to render synchronous third views and
   test observable landmarks and tails. The extra camera can constrain
   true X/Y/Z motion but cannot create an image of a virtual bone endpoint.
6. **Keep motion keys.** No front/side observation -> zero data term
   for that view/frame; the final output should still contain the same
   full-frame Root and all FK pose keys for iClone. The downstream
   animation-curve simplifier (if any) is a separate problem.

**Promotion gate:** 3D Tail + full-body GT error and extreme-case
Root/Tail08 metrics must improve against untouched Baseline on both
a new held-out synthetic sequence AND a verified continuous real/Blender
action. The code/CI passing alone is insufficient.

### Literature anchors

- [Anipose, Cell Reports 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8498918/):
  robust observation-fitting, temporal/spatial constraints; specifically
  cautions that long missing tracks may be hallucinated by temporal
  regularization and that only two cameras can limit recovery.
  This does **not** prove AnyMove's particular prior will succeed.
- [CoTracker3 (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.html):
  stronger point tracking and occlusion handling than simple LK is a
  possible later candidate, but tracked image pixels are NOT a valid
  substitute for invisible/virtual FK bone endpoints.

## Reproduce

```bash
cd /f/anymove_gt_eval_20261101
export PYTHON=python.exe
export BLENDER="/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"
export MOCAP_CREATURE=assets/dino

# Already run; use a NEW tag on re-execution to preserve old reports
bash tools/eval_dino_occlusion.sh \
  runs/dino/gt_experiments/seed7_smoke_24 \
  occlusion_mask_new_tag \
  mesh,p2_error,p2_error_temporal,oracle_gt \
  runs/dino/reader_hr_quality_p2_eval_snapshot.pt
```

The code is in the opt-in branch `feature/dino-motion-quality-p123`,
currently retained as Draft PR #2.

## Local Motion Prior prerequisite audit

Read-only Blender inspection on the user's local Dino assets confirms:

- `assets/dino/rigged/rigged.blend`: one Armature, **no active Action, no Actions in the .blend, no NLA tracks**.
- `assets/dino/rigged/Dinosaur_FullbodyTemplate_fix.fbx`: imported in a separate non-saving Blender process; **no Actions, no active Action and no NLA tracks**.
- Only existing `runs/dino/clips/test_walk` was found; this is solved/AI video output and **not independently verified motion truth**.
- The Seed 7/11 Blender GT sequences created by `mocap.exam.make` are synthetic, deterministic, rig-valid test motions, but **not** independently verified Dino locomotion with contact annotations. The 2,200 still Reader poses are not a real time series.

**Implication:** A physically meaningful or learned Dino Motion Prior cannot be claimed trained from available independent continuous-motion ground truth. Next first-party task is to establish at least two genuinely distinct, verified rigged continuous animation sources (walk/turn and tail articulations), validate bone names/FPS/contacts, then use separate training and held-out sequences to assess missing-view reconstruction. Do not promote a prior trained only on this AI-Solver's own output. Synthetic exam motion may support *infrastructure smoke*, not claims of natural gait fidelity.
