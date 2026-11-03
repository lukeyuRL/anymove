# P2 �X AnyMove N-Camera Reader Contract and Differentiable FK Solver

Date: 2026-11-01  
Branch: `feature/dino-motion-quality-p123` (`F:\\anymove`)  
Implementation: **P2 data contract + multi-camera FK Solver + controlled validation PASS**.  
Quality promotion for pretrained 4-view Reader: **FAIL** (worse than 2 views).  
No H3 generation, no Reader retraining, no production `motion.json` changes, no Commit/Push.

## Scope

The production 2-view path (`mocap.detect`, `mocap.solve`, iClone Adapter) remains unchanged. The P2 implementation is an *experimental parallel pipeline* accepting **2, 3, or 4 views** and arbitrarily named rig/keypoints. It does not need to know a creature's bone names and does not copy a BVH Euler convention into an incompatible rig basis.

- `mocap/multiview.py`: versioned, validated observation NPZ contract and orthographic camera matrix projection / weighted triangulation.
- `tools/build_four_view_observations.py`: convert prior P1 four-view Blender stills to one view-indexed observation packet, from either actual Reader inference or controlled Blender-truth projections with artificial 2D corruption.
- `mocap/solve_multiview.py`: 2/3/4-view weighted robust 2D reprojection and exact imported-rig FK, with root translation and axis-angle root rotation, bounded Euler rotations and optional temporal penalty.
- `tools/eval_multiview_gt.py`: separate post-optimization evaluation against independent Blender 3D Ground Truth, never loaded by the solver.
- `tests/test_iclone_multiview.py`, `tests/test_iclone_multiview_solver.py`: geometry, rank, provenance, GT separation, selected-view leakage and real fixtures.

Data contract:

```text
view_ids            [V]              e.g. front,side,back,opposite_side
names               [J]              exact rig_info keypoint order
uv                  [V,T,J,2]        original view-local pixels
confidence          [V,T,J]          RAW Reader peaks (not P(visible))
valid               [V,T,J]          Boolean in-bounds / observation flag
world_to_camera     [V,4,4]          rigid camera world-to-local matrix
image_size_px       [V,2]            width,height
meters_per_px       [V]              orthographic camera scale
frame_ids           [T]              sample/frame IDs
fps                 scalar           video or capture fps
independent_stills  bool scalar      True for P1 independent pose samples
source_kind         scalar string    blender_gt_projection / synthetic_noisy / reader_inference
```

**No 3D GT** or ground-truth surface-visibility oracle is inserted in the Solver NPZ. A separate `*.ground_truth.npz` contains world GT joints and per-view visibility labels for evaluation. The experimental solver does not read the GT file or its path. `check_packet` fails closed if GT is mistakenly included.

The two opposite orthographic cameras have the same rank-2 camera axes; Front+Back alone is **not a valid 3D triangulation**, and the relative singular-value test rejects it. A missing Keypoint due to out-of-frame Reader predictions is initialized from the known Rig Rest Pose plus the observed root translation, **not copied from GT**. It is then constrained by FK and valid 2D view observations.

Existing Reader checkpoints have seen only Front/Side training views and possibly colorized/texture-rendered images, whereas this P1 smoke is a gray Workbench rendering. Therefore the Back/Opposite Side input is **out-of-distribution**, and performance must be evaluated, not assumed.

## P1 static test dataset

Existing static Blender poses, no motion clips and no H3:

- Dino: `runs/dino/four_view_visibility_p1_validation/` (two independently sampled poses; 67 Bones/92 Keypoints).
- Dragon: `runs/dragon/four_view_visibility_p1_validation/` (two independently sampled poses; 42 Bones/50 Keypoints).
- 4 orthographic camera matrices per pose and independent 3D GT (verified previously by full scene raycast and camera reprojection).
- `T=2` means **two unrelated static poses**, NOT adjacent video frames. The temporal loss is therefore disabled by design.
- Compare `front,side` and `front,side,back,opposite_side` using the same first two view observations and same 320-step optimizer budget.
- Both experiments share identical Rig geometry and FK constraints. The starting triangulation uses only each branch's selected 2D observations.

### A. Controlled synthetic noisy 2D observations

Inputs are 2D projections of Blender 3D GT with deterministic perturbations: normal visible-part sigma = 2 px, hidden-part sigma = 12 px and unknown-part sigma = 3 px; Reader error patterns are NOT claimed to match this model. These labels are used only to **simulate measurement noise**, not as a visibility oracle inside Solver.

| Creature | Direct triangulation 2 views | Direct triangulation 4 views | FK Solver 2 views | FK Solver 4 views |
|---|---:|---:|---:|---:|
| Dino | 229.10 mm | 218.95 mm | 118.63 mm | **91.99 mm** |
| Dragon | 56.93 mm | 46.86 mm | 28.12 mm | **23.77 mm** |

**Conclusion:** The N-view geometry and FK code can use extra valid views to reduce 3D error. This is a controlled unit/system validation only. It does not show actual four-view Reader/H3 generalization.

### B. Actual pretrained 2-view Reader applied to all four Workbench views

Reader 2D pixel error compared with independently stored Blender 2D GT:

| Creature | Front | Side | Back (OOD) | Opposite Side (OOD) |
|---|---:|---:|---:|---:|
| Dino | 5.55 px | 9.76 px | 29.45 px | 18.43 px |
| Dragon | 7.10 px | 5.60 px | 25.09 px | 25.72 px |

N-camera FK Solver results against the **same independent Blender 3D GT**:

| Creature | FK 2 views | FK 4 views | Four-minus-two |
|---|---:|---:|---:|
| Dino | **343.41 mm** | 619.90 mm | +276.49 mm |
| Dragon | **61.88 mm** | 97.25 mm | +35.38 mm |

**Conclusion:** Four-view Reader quality currently FAILS. Adding two poor OOD observation streams hurts 3D accuracy, despite correct multiview geometry. Do not use four-view predictions or weights in production yet. The existing Reader architecture *can* be called once per view, but its checkpoint has NOT been trained for Back/Opposite angles.

The gray Workbench domain shift affects the whole test. These 3D error values must not be compared directly with the earlier H3-video reprojection experiments, nor generalized to natural Creature motion or all poses.

**Important unit distinction:** mm denotes mean Euclidean world 3D joint error against Blender GT; px denotes mean 2D Euclidean error against Blender GT or input observations (explicitly distinguished in per-run JSON). Lower 2D reprojection against the Reader's own predictions is NOT proof of 3D quality.

## Reproduce on local Windows environment

All commands run from `F:\\anymove`, Python 3.14.3, RTX 3070 with CUDA:

```powershell
cd F:\\anymove

# Package synthetic corruption from P1 Blender Ground Truth
python tools/build_four_view_observations.py `
    --source runs/dino/four_view_visibility_p1_validation `
    --rig assets/dino/rigged/rig_info.json `
    --mode synthetic_noisy --indices 0,1 --seed 1729 `
    --noise-px 2 --occluded-noise-px 12 `
    --out runs/dino/four_view_visibility_p1_validation/NEW_SYNTH_PACKET.npz

# Use the already-trained Dino Reader on four existing Workbench stills.
python tools/build_four_view_observations.py `
    --source runs/dino/four_view_visibility_p1_validation `
    --rig assets/dino/rigged/rig_info.json `
    --mode reader --checkpoint runs/dino/reader_hr_v1.pt `
    --indices 0,1 `
    --out runs/dino/four_view_visibility_p1_validation/NEW_READER_PACKET.npz

# Fit exactly the same packet with 2 or 4 selected cameras.
python -m mocap.solve_multiview `
    runs/dino/four_view_visibility_p1_validation/p2_reader_ood.npz `
    assets/dino/rigged/rig_info.json `
    --views front,side --iterations 320 `
    --out-prefix runs/dino/four_view_visibility_p1_validation/NEW_2V_FIT

python -m mocap.solve_multiview `
    runs/dino/four_view_visibility_p1_validation/p2_reader_ood.npz `
    assets/dino/rigged/rig_info.json `
    --views front,side,back,opposite_side --iterations 320 `
    --out-prefix runs/dino/four_view_visibility_p1_validation/NEW_4V_FIT
```

`--views front,side,back` is also supported for three cameras. Two-view runs cannot access unselected Back/Opposite observations; a test deliberately corrupts those values and confirms identical 2-view optimized output.

Evaluate fitted outputs using the **separate** `tools/eval_multiview_gt.py` command. Keep `--ground-truth` out of Solver commands entirely.

Actual P2 result JSON files exist under each Creature's `four_view_visibility_p1_validation` directory:

```text
p2_synth_2v4v_3d_eval_v1.json
p2_reader_2v4v_3d_eval_v1.json
p2_synth_fit_2view_v1.npz / .json
p2_synth_fit_4view_v1.npz / .json
p2_reader_fit_2view_v1.npz / .json
p2_reader_fit_4view_v1.npz / .json
```

## Validation contract and limitations

- Exact 3D projection from saved P1 camera matrices roundtrips within source metadata precision. Numerical Torch and NumPy projections agree.
- Degenerate Front+Back rank test fails safely, regardless of nearly-zero floating-point singular values.
- Surface-part visibility in `*.ground_truth.npz` is used only for the controlled noise generator and independent audit, **never** as model-predicted `P(visible)`.
- Root pose initialized from selected-view triangulation and known Rig Rest Pose; missing joints are seeded from Rig (not GT).
- Full FK rigid bone lengths are preserved; optimizer output is validated by replaying `mocap.solve.Rig.forward` and comparing world joints in a separate evaluator.
- Invalid/mixed camera matrices, duplicate view names, nonfinite observations, and accidental 3D GT leakage fail closed.
- The 2/4-view results are based on **only two poses per creature**, no independent continuous motion sequences. No statistically defensible motion-accuracy or temporal-visibility claim.
- The Optimizer is not a drop-in replacement for `motion.json`. Independent still poses are saved as NPZ + JSON, not importable animation clips.
- The current four-view strategy uses extra orthographic opposite views (more visibility, geometric axes partly redundant). It does not guarantee two additional independent depth dimensions.

## Next engineering decision

**P2 architecture acceptance PASS. Four-view Reader-based quality acceptance FAIL.**

Before integrating four-view motion into iClone:
1. Train/finetune the high-res Reader on **all four correct camera angles**, with independent held-out Blender poses. Preserve Dragon, Dino and other Creature pose-index contracts.
2. Calibrate per-view 2D localization uncertainty against independent Blender GT, and avoid blending OOD camera predictions merely because the camera exists. Unknown/occluded mesh-part surface labels must not be used as if independently inferred at inference.
3. On new held-out synchronized **continuous** Creature motion, compare 2/3/4-view 3D joints, Root trajectory, joint-limit violations, temporal jitter, contact stability, and rendered video consistency.
4. Only after proving generalization and synchronization for Blender GT, discuss H3 four-view prompts. **Do not generate paid H3 videos without explicit approval.**


## Optional Reader Visibility/Uncertainty extension (separate from training labels)

The P2 observation packet now accepts optional per-camera Reader predictions:

```text
visibility_score     [V,T,J]  floats [0,1]
expected_error_px    [V,T,J]  nonnegative floats
quality_semantics    scalar   "synthetic_cutout_only_uncalibrated"
```

These fields are accepted **only together** and require the exact uncalibrated semantics. They are **NOT** used as Solver weights, since the existing checkpoint predicts artificial cutout visibility and synthetic localization-error heuristics, not validated real mesh-surface occlusion probabilities. Ground-truth `joint_surface_part_visible` is explicitly rejected in the Solver packet and remains in `*.ground_truth.npz` for independent testing only.

Dino quality-head smoke was run against the already-trained `runs/dino/reader_hr_quality_p2_headonly.pt` checkpoint, producing:

`runs/dino/four_view_visibility_p1_validation/p2_reader_quality_ood.npz`

The quality fields have shape `(4,2,92)`; their declared semantics is `synthetic_cutout_only_uncalibrated`. Mean predicted localization error by view (px): Front **7.28**, Side **4.50**, Back **9.63**, Opposite Side **7.63**. These estimates are not ground-truth errors and are not necessarily calibrated, especially on the new OOD views. A regression test changes every quality prediction to extremes and verifies that the experimental FK Solver is identical.

**Input provenance note:** New packet manifests store the hash of `rig_info.json`; new Solver outputs include a SHA256 of their input NPZ. Earlier P2 smoke outputs predate these added fields and remain archived in their original versioned result files. The evaluation tool checks source packet path/view/frame consistency even for old smoke results.

