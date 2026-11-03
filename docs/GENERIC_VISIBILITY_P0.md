# Generic Visibility P0 �X Dragon / Dino (Creature-Independent Two-View Reliability)

**Date:** 2026-11-01  
**Branch:** `feature/dino-motion-quality-p123`  
**Implementation:** P0 code integrated and opt-in; **not accepted as true mesh occlusion learning**.

## Scope

Replace the hardcoded Dragon-only experimental Front hind-leg reliability proxy with a common **rig topology-driven** reliability interface for two-view Dragon, Dino and compatible new Creature rigs.

This milestone is **not four-view**, is not a trained visibility model, and is not a calibrated P(visible). It does not recover 3D depth from a single Front view. It detects conservative **observation reliability risk** when:

- A bone's observed orthographic 2D head-to-tail segment exceeds its known 3D rest length (physical impossibility, allowing a scale-aware tolerance);
- Front/Side shared-height observations disagree **and** one view has stronger proximity to anatomically distant keypoints than the other. If overlap is similar in both views, abstain rather than guessing the faulty view.

All bones, keypoint indices and rest positions come from `rig_info.json`. The implementation does **not** depend on names like `knee_L` or `CC_Dinosaur_L_Calf`; it uses the rig's parent graph and numeric index mapping.

## Implementation

| File | Description |
|---|---|
| `mocap/visibility_generic.py` | Two-view creature-generic reliability proxy, Front/Side individual (T,J) multipliers, rigorous checks and JSON audit |
| `mocap/solve.py` | Optional side- and front-specific weights in 2-view triangulation, confidence loss and root initialisation |
| `mocap/visibility.py` | Preserved Dragon-only original for historical comparison |
| `tests/test_iclone_visibility_generic.py` | New behavior / real Dino + Dragon / invalid inputs tests |
| `tools/audit_generic_visibility_p0.py` | Compare default and generic variants against the SAME raw Reader observations, without confusing differences for 3D truth |

Flags:

- `--visibility-aware` / `--generic-visibility`: new creature-generic two-view mode, **opt-in**.
- `--visibility-legacy-dragon`: preserves previous Dragon `front_hind_weights` behavior, **opt-in**.
- No flag: completely unchanged default Solver behavior.
- Do not mix with `--temporal-association` unless separately validated; current guard rejects combinations.

`--visibility-aware` is **intentionally redefined** relative to older experimental Dragon-only code. The result schema identifies the new algorithm version; old runs must retain their prior provenance.

### Weight formula

Let `w_v(t,j)` be a bounded reliability multiplier for View `v`.

```text
w_v = max(0.45, 1 - 0.55 * risk_v)
risk_v = max(
    orthographic_bone_overprojection_risk_v,
    shared_height_disagreement * view_overlap * view_overlap_advantage
)
```

The implementation uses `min(w_bone,w_overlap)`, which is equivalent to the maximum risk, and 5-frame median filtering on risk. Overlap partners must come from anatomically different branches; same or adjacent bone chains are excluded.

The original Reader peak confidence and tracked weights are **preserved**. The new multiplier influences the Front and/or Side reprojection loss and initial triangulation and is not interpreted as an independent probability. No raw `det.npz` is overwritten.

### Camera and rig contract

`mocap.layout_guard.verify_layout_for_rig` must verify the Creature camera fields before generic visibility is enabled. Both views must be synchronized, orthographic, have identical vertical direction and metres-per-pixel and retain an exact ordered keypoint-to-Rig mapping.

The generic **Creature** interface is not yet generic **N-Camera**. The planned 4-view Solver will need per-view camera matrices and multi-view consistency rather than this Front/Side shared-height heuristic.

## Reproduce on the authorized Windows machine

```powershell
cd F:\\anymove

# Dino
$env:MOCAP_CREATURE='assets/dino'
python -m mocap.solve `
  runs/dino/clips/test_walk/det.npz `
  assets/dino/rigged/rig_info.json `
  runs/dino/clips/test_walk/visibility_generic_p0/motion.json `
  --visibility-aware

# Dragon
$env:MOCAP_CREATURE='assets/dragon'
python -m mocap.solve `
  runs/dragon/clips/hr_roar/det.npz `
  assets/dragon/rigged/rig_info.json `
  runs/dragon/clips/hr_roar/visibility_generic_p0/motion.json `
  --visibility-aware

# Code regressions
python -m unittest discover -s tests -p "test_iclone_visibility_generic.py" -v
python -m unittest discover -s tests -p "test_iclone_visibility.py" -v
python -m unittest discover -s tests -p "test_iclone_solver_camera_layout_guard.py" -v
```

Opt-in runs write a new `motion.json` and `motion.visibility.json`. The latter reports the algorithm, parameters, affected joint/frame counts, per-affected-joint Front/Side multipliers, source detection SHA-256 and resulting reprojection. None of these numbers are proof of actual occlusion.

## Results �X Dino existing 124-frame H3 clip

| Metric | Default | Generic P0 | Delta |
|---|---:|---:|---:|
| Front reprojection against the SAME raw Reader | 4.46175 px | 4.48725 px | +0.02550 px |
| Side reprojection against the SAME raw Reader | 4.48184 px | 4.48737 px | +0.00553 px |

- Original Dino: 67 Bones, 92 Keypoints, 124 Frames, 24fps.
- Front risk-weighted observations: 1,584 / 11,408, mean reliability weight 0.982432.
- Side risk-weighted observations: 1,188 / 11,408, mean weight 0.980550.
- Bone-head motion differences vs default: mean ~0.418 cm, P95 ~1.512 cm, max ~15.541 cm.
- Root motion differences vs default: mean ~0.351 cm, max ~0.646 cm.
- 2D reprojection **slightly worsens** for this Dino clip. That is NOT a statement about unknown 3D motion accuracy, nor a reason to promote this heuristic.

## Acceptance and next steps

P0 acceptance means **both rigs can run the same, versioned, geometry-based opt-in module safely without renamed joints, broken Root/FK, changing original observation arrays, or silently inheriting Wolf camera settings**.

P0 explicitly does **not** require improved reprojection on every clip, because visibility weights may correctly reject a wrong Reader observation; conversely, lower reprojection to the Reader can also overfit its mistakes.

Do not enable generic risk weights in production without independent known-3D-ground-truth tests, mesh part visibility annotations, temporal identity checks, foot contacts and iClone import QA.

Future stages:

1. Blender Mesh Surface Part-ID and Depth Buffer labels (true occlusion data with unknown/unlabelled cases);
2. Train and calibrate an independent visibility and localization uncertainty head on the Creature reader;
3. Migrate camera and observation storage to `views[view_id]` and build an N-view solver for 2/3/4 views; verify synchronisation on Blender GT before attempting 4-view AI video.


## Results �X Dragon existing 124-frame H3 clip

| Metric | Default | Generic P0 | Delta |
|---|---:|---:|---:|
| Front reprojection against the SAME raw Reader | 4.52532 px | 4.48715 px | -0.03817 px |
| Side reprojection against the SAME raw Reader | 5.26708 px | 5.29786 px | +0.03078 px |

- Original Dragon: 42 Bones, 50 Keypoints, 124 Frames, 24fps.
- Front risk-weighted observations: 893 / 6,200, mean reliability weight 0.981242.
- Side risk-weighted observations: 1,406 / 6,200, mean weight 0.940910.
- Joint motion differences vs default: mean ~0.329 cm, P95 ~1.319 cm, max ~20.829 cm.
- Root motion differences vs default: mean ~0.123 cm, max ~0.441 cm.
- **Mixed reprojection**, not demonstrated 3D quality gain.
- Full Solve return code 0; `motion.json`, `motion.visibility.json`, `comparison.json` available at
  `runs/dragon/clips/hr_roar/visibility_generic_p0/`.

## Final P0 verification

- Both full 1,500-iteration 124-frame opt-in solves succeeded, with separate output paths and untouched original `det.npz` and `motion.json`.
- Dragon and Dino results share the schema `anymove.generic_two_view_reliability_proxy/1`.
- The original Dragon-only helper in `mocap/visibility.py` is unchanged and its 10 legacy tests passed.
- `tests/test_iclone_visibility_generic.py`: 7 tests passed including actual local Dino/Dragon rigs and synthetic view-exclusive corruption.
- `tests/test_iclone_solver_camera_layout_guard.py`: 4 tests passed.
- Profile `comparison.json` explicitly flags that a two-view Solver's common-Reader reprojection does **not** prove true 3D or occlusion accuracy.

### Final decision

**P0 FEATURE/COMPATIBILITY: PASS. P0 MODEL-QUALITY PROMOTION: NOT ESTABLISHED.**

The correct next gate is a mesh-surface visibility truth audit and independent 3D motion comparison. Do not infer that an error-gated visibility proxy predicts actual occlusion, and do not apply it to single-view inputs because it relies on cross-view shared-height observations.


## Post-export FK numerical round-trip (2026-11-01)

Recomputed `Rig.forward(trans, root_quat, euler)` against the stored `joints` in the experimental `motion.json`:

- Dragon: 124 frames / 50 keypoints, maximum 3D joint difference **0.000094 m**.
- Dino: 124 frames / 92 keypoints, maximum 3D joint difference **0.000154 m**.
- Both are within the expected rounding effects of persisted `motion.json` fields.
- Dragon experimental root maximum displacement from initial: 0.080 m.
- Dino experimental root maximum displacement from initial: 0.511 m.
- These are FK and numerical consistency checks, NOT 3D accuracy / naturalness tests.

Additional regression suites: `test_iclone_p2_weighted_reader.py` 5/5 and `test_iclone_association_safety.py` 11/11 passed. Overall targeted checks: 37/37.
