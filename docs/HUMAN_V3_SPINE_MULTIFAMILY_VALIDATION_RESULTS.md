# Human V3 Nose — Four-family Spine SO(3) Validation Results

**Date:** 2026-11-04
**Branch:** `feature/iclone-general-camera-angles`
**Status:** **NO CANDIDATE PASSED. Keep current baseline.** Two Spine variants reduce angular acceleration and Root heading error but degrade 3D positions outside predefined tolerances.

## Pre-registered validation contract

The thresholds and four action families were recorded in `docs/HUMAN_V3_SPINE_MULTIFAMILY_VALIDATION_GATE.md` before the run. Exact run:

```powershell
cd F:\anymove
python -m tools.evaluate_human_v3_spine_multifamily --iterations 400 --takes 1
python -m tools.summarize_human_v3_spine_multifamily
python -m tools.audit_human_v3_spine_observability
```

- **Dataset:** Human V3 Nose Spine Motion GT V1, 144 clips ×107 frames, 24 FPS, source SHA `677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc`.
- **Train:** 96 clips. **Validation:** 24. **Test:** 24 sealed, **none read** by this evaluation.
- **Sample used:** one preselected Validation clip from each of `walk_turn_right`, `waist_twist_right`, `lateral_bend_left`, `reach_counterrotate`, total **4 clips ×107 frames = 428 frames**.
- **Observation contract:** Same orthographic 315°/45° 672×672 2D projections from procedural Blender 3D GT, identical deterministic independent 1-pixel Gaussian noise, full 67-point 2D observations, same 55-bone rig matrices, 400 iterations per candidate.
- **Three variants:** `baseline` (Train-derived Euler limits, old Root initializer), `root_so3` (SO(3) angular prior + 3D hip Kabsch Root + Root observation evidence), `neutral_so3` (same + Train median Spine local Euler initialization).
- **RL_BoneRoot is fixed** in all three outputs: local Euler = 0, Root turn belongs to Armature Object. None of these files is iClone FK channel output.
- The source is procedural GT, **not** real Reader errors, real AI-generated video, or natural human motion.

## Cross-family matched 400-iteration results

| Mean metric | Baseline | Root + SO(3) | + Train neutral Spine initialization |
|---|---:|---:|---:|
| Whole-body 3D MPJPE | **0.502 cm** | 0.716 cm | 0.870 cm |
| Whole-body 3D P95 | **1.390 cm** | 2.446 cm | 3.503 cm |
| Root mean orientation error | 9.78° | **2.35°** | 2.73° |
| Waist angular acceleration (°/source-frame²) | 40.74 | 23.26 | **9.92** |
| Pelvis angular acceleration (°/source-frame²) | 33.38 | 21.79 | **1.92** |
| Seven-bone trunk angular acceleration (°/source-frame²) | 22.56 | 10.97 | **4.00** |
| Same-bone source GT acceleration | 0.087 | 0.087 | 0.087 |
| Proxy contact foot horizontal speed | 0.862 cm/frame | 0.864 | 0.877 |

Relative to baseline, `root_so3` lowers Waist angular acceleration **42.9%**, Pelvis **34.7%**, and Root heading error **76.0%**, but increases MPJPE **+0.214 cm**. `neutral_so3` lowers Waist **75.6%** and Pelvis **94.2%**, but increases MPJPE **+0.367 cm**.

**Both fail** predeclared criteria: max mean MPJPE regression allowed `+0.10 cm`, max single-family allowed `+0.20 cm`. `root_so3` worst-family regression was `+0.264 cm`, `neutral_so3` `+0.443 cm`. All four action families showed position error regressions.

### Per-family 3D MPJPE (cm)

| Motion | Baseline | Root + SO(3) | + Train neutral |
|---|---:|---:|---:|
| Walk Turn Right | **0.554** | 0.699 | 0.855 |
| Waist Twist Right | **0.491** | 0.755 | 0.933 |
| Lateral Bend Left | **0.481** | 0.721 | 0.849 |
| Reach Counter-Rotation | **0.483** | 0.689 | 0.841 |

Current Train-derived joint limits were independently checked against the **96 Train and 24 Validation clips**: no local Euler source rotations were outside them. Therefore the accuracy regression is **not** explained by joint limits clipping the newly generated motion examples.

## Follow-up structural observability analysis

Using 12 distinct frames from these four Validation Solver clips, finite-difference the 268-dimensional 2D observation vector (67 joints ×2 views ×2 components) against **15** parameters: Root XYZ, Pelvis XYZ, Waist XYZ, Spine01 XYZ and Spine02 XYZ.

With all 67 points assumed visible and exact orthographic geometry, the smallest 2D Jacobian singular values were approximately **1.65×10⁻⁶ to 1×10⁻⁵ px/radian** (numerically near-singular), with condition numbers up to **1.72×10⁹**. Across all 12 sampled frames the weakest single-parameter sensitivity list contained `CC_Base_Pelvis.Y` and `CC_Base_Spine01.X`. The pseudo-inverse also gives relatively high uncertainty for Root.Z and Pelvis.Y, consistent with weak overall Root/Pelvis orientation observability; the precise pairwise correlation was not independently measured.

**Interpretation:** the image positions are highly informative for anatomical *positions*, but particular distributed Root/Pelvis/Waist/Spine orientations are not uniquely identifiable. Local reallocation of rotation can preserve joint locations while producing very different per-bone angular trajectories. The idealized independent 1px pixel-noise covariance gives some single-axis uncertainty estimates up to roughly **14°**; due to the rank deficiency, a pseudo-inverse may understate true uncertainty. This numerical conditioning is a local diagnostic, not a calibrated probability, and it **does not prove** that all motion frames are unobservable.

## Why this matters and recommended development

**Do not promote either SO(3) variant to the default Solver.** The result is repeatable across four procedural motion families and demonstrates a consistent accuracy vs rotation-stability trade-off. Do not open the 24 sealed Test clips to choose weights.

Next engineering direction should address the *source of rotational evidence and parameter identifiability*, not only a larger scalar smoothing penalty:

1. Introduce a **separate trunk orientation observation** (e.g. learned surface cues/torso frame from texture, skin landmarks or image features) with explicit uncertainty; validate independent geometric consistency across 315°/45° views.
2. Use a **low-dimensional hierarchical twist/swing allocation**, possibly root and pelvis orientation first, then Waist/Spine with correlated constraints, rather than allocating all local Euler XYZ independently for every frame.
3. Design a constrained second-stage optimizer: preserve a bounded 2D reprojection and 3D joint error budget while minimizing SO(3) local angular acceleration **inside the weakly observable subspace**. Avoid a global zero-yaw Root prior.
4. Continue enforcing **RL_BoneRoot static**, Root/Armature Object motion ownership, exact 24FPS source timestamps and matching FK rest basis.
5. Calibrate against more diverse Validation and real motion examples; use the 24 sealed Test only after a candidate meets the upfront development criteria.

## Reproducible local reports and artifacts

```text
F:\anymove\runs\human\fullbody_finger_v2_hr_315_45\solver\spine_so3_research\multifamily_validation\
```

- `multifamily_1takes_400iter_validation.json` — all per-clip, per-bone scores and raw/optimized FK metadata
- `multifamily_4families_decision_report.json` — pre-registered gate results
- `spine_jacobian_observability.json` — 15-column Jacobian and SVD for 12 Validation frames
- Individual `*.motion.json` and diagnostic `*.log` for each of the 12 matched Solver runs

Dataset and evaluation remain local. This repository commit contains code and documentation only, not Blender weights, generated videos, large NPZ inputs or 3D motion outputs.
