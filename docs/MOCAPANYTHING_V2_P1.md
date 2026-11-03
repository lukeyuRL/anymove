# MoCapAnything V2 �� AnyMove P1 Front-only Solver: 124-frame Controlled Ablation

**Date:** 2026-11-01  
**Status:** Implementation PASS; candidate **FAIL** against Front-only no-V2 baseline and held-out Side  
**Safety:** No new generated video, no new Creature dataset, no iClone production-motion modifications, no new Motion-Prior training.

## 1. Purpose and scientific boundary

Test the claim that pretrained MoCapAnything V2 contributes a useful creature-agnostic 3D motion prior to AnyMove, using an existing H3 video and the same 67-bone Dino.

The prototype uses BVH-native differentiable forward kinematics, **not** an unvalidated direct copy of BVH Euler angles into AnyMove's bone-local Euler channels.

Two equal-budget candidate optimizers:

- **Rest-only**: reference rest local rotations + a bounded learned correction; no V2 motion.
- **V2-conditioned**: pretrained V2 local rotations + an identically bounded learned correction.
- Both fix the same initial world Root track from the existing AnyMove Front Reader and known Camera Layout. Depth / Y stays fixed. Both use only Front 2D observations and identical objectives, step budgets, initial zero corrections and learning rate.
- The previous full AnyMove Front+Side Solver is a reference, *not a fair single-view training baseline*.
- Side Reader observations are carried only in the evaluation payload. Neither its values nor its confidence weights are passed to or read by `optimize`; `eval_world` uses them only after optimizer outputs are frozen. The poisoned-Side test verifies this separation.

Loss used for each:

```text
L = confidence_weighted_charbonnier_robust_front_reprojection
  + 12 * mean_squared_local_rotation_correction
  + 0.035 * mean_squared_root_relative_joint_acceleration_in_pixels
  + 3 * mean_squared_correction_angular_velocity
  + 3 * soft_penalty_for_correction_above_20deg
```

Correction Euler components are bounded within ��48�X via `tanh`; this is an experimental safety bound, **not** a validated anatomical limit. Real Dino imported rotation limits are draft ��20�X/axis. FK bone lengths are structurally fixed.

## 2. Reproducible assets and commands

```powershell
cd F:\\anymove

python tools/mocapanything_v2_front_solver.py --smoke

python tools/mocapanything_v2_front_solver.py `
  --root-bvh F:\\mocapanything_v2_test\\Dino_V2_front_root_124f.bvh `
  --detections runs\\dino\\clips\\test_walk\\det.npz `
  --rig-info assets\\dino\\rigged\\rig_info.json `
  --reference-bvh F:\\mocapanything_v2\\zoo\\characters_face_zplus\\Dino\\Dino_ffs.bvh `
  --v2-repo F:\\mocapanything_v2 `
  --steps 160 --lr 0.012 --prior 12 --temporal 0.035 `
  --out-dir F:\\mocapanything_v2_test\\p1_front_only_160

python tools/eval_mocapanything_v2_p1.py

python -m unittest discover -s tests -p "test_mocapanything_v2_front_solver.py" -v
python -m unittest discover -s tests -p "test_mocapanything_v2_root_bridge.py" -v
```

The experiment was run on Python 3.14.3 / CUDA RTX 3070 8GB. FK evaluation is `124 frames �� 67 bones`, with 51 non-leaf joints receiving optimizable correction; leaf-bone rotations are frozen because their heads do not constrain their rotations.

## 3. Results (124 frames; identical 67 matched bone-head observations)

| Method | Front mean (px) | Side held-out mean (px) | Side hind legs & toes (px) |
|---|---:|---:|---:|
| Original V2 with Front Reader Root | 14.2924 | 29.0853 | 49.296 |
| Rest-only Front Solver, 160 steps | **2.0477** | **22.8831** | **39.965** |
| V2-conditioned Front Solver, 160 steps | 2.2368 | 23.3301 | 40.947 |
| Production AnyMove two-view Solver | 4.6065 | 4.2423 | 5.939 |

**V2 conditioning loses 0.1891 px Front and 0.4470 px Side to the no-V2 Front-only baseline.** Both exhibit large Side errors even after extremely low Front error, consistent with ambiguous and wrong out-of-plane joint locations. This is evidence against using the pretrained V2 as a guaranteed helpful prior on this one Dino H3 example, not a universal statement about V2 performance.

The initial V2-only root correction was already tested at P0. Additional side-hip root Y estimation improved its Side error only ~29.09 �� 28.80 px, so missing root depth does not explain most of the gap.

### Kinematic / temporal audit

- Both P1 BVHs export at **24 FPS**, preserve **124 frames / 67 bones**, and pass numerical re-import. Maximum FK positions roundtrip error ~2.5 �� 10?? m.
- Maximum bone-length deviation from Rest is ~3 �� 10?? m (BVH precision limit).
- Root track max displacement from first frame ~0.427m (Front Reader-derived); this is **not** independent V2 Root prediction.
- For corrected / Rest / V2-conditioned runs respectively, p95 **root-relative joint acceleration per frame2**: **0.01413 / 0.04312 / 0.04870 m**. Fitting Front can increase motion dynamics/jitter; no conclusion about naturalness without real ground truth.
- Fraction of local rotation-axis values exceeding **draft** 20�X: original V2 0.048%, Rest-only 0.867%, V2-conditioned 0.766%. These imported 20�X limits are not validated creature anatomy, so the numbers are audit signals only.
- Foot-ground contact heuristic found 0 reliable contact frame-pairs in either optimized P1 result; **Foot Sliding cannot be established or ruled out** from this heuristic.

## 4. Separation tests & validity

1. Reference skeleton aligned by name rather than index, with a rest-pose max error ~2.7e-6m. Differentiable Torch FK vs BVH's NumPy FK max difference **2.52e-12m** before optimization.
2. The Root Bridge preserves the original input video, its 2D Reader detections and all V2 local rotations; corrected Root X/Z comes from original Front hip detections only.
3. `test_side_poison_has_zero_effect_on_solver_output` replaces all Side UVs with 1e7 while leaving Front unchanged. The optimized 3D output and loss remain bit-for-bit identical. Side affects evaluation only.
4. Both branches use the same optimizer/loss/bone selection/Root. BVH export passed hierarchy/frames/FPS/FK verification. P1 unit tests: **4/4 passed**; P0 Root Bridge unit tests: **4/4 passed**.
5. Side is a held-out **2D Reader detection**, not independent 3D ground truth. Original H3 may not contain synchronized true multi-camera 3D motion. The reference two-view Solver was itself optimized to the Reader detections.

## 5. Files written

```text
F:\\anymove\\tools\\mocapanything_v2_front_solver.py
F:\\anymove\\tools\\eval_mocapanything_v2_p1.py
F:\\anymove\\tests\\test_mocapanything_v2_front_solver.py
F:\\anymove\\docs\\MOCAPANYTHING_V2_P1.md
F:\\mocapanything_v2_test\\p1_front_only_160\\Dino_P1_rest_front_only.bvh
F:\\mocapanything_v2_test\\p1_front_only_160\\Dino_P1_v2_front_only.bvh
F:\\mocapanything_v2_test\\p1_front_only_160\\p1_report.json
F:\\mocapanything_v2_test\\p1_front_only_160\\p1_quality_diagnostics.json
```

## 6. Decision / next stage

**Do not replace existing AnyMove dual-view Solver and do not enable the V2 prior by default.** P1 demonstrates a clear Front�VSide generalization gap and no advantage from V2 conditioning on this example.

Better next research decisions, in order:

1. **Controlled depth-aware source:** create multi-view 3D GT from *existing* rig/synthetic assets (no new H3 generation) to isolate pose reconstruction vs H3 view inconsistency. Preserve video synchronization and camera matrices; avoid calling programmatic animation natural MoCap.
2. **Pose-specific visibility and uncertainty:** use the existing AnyMove trained Reader confidence and potential explicit occlusion masks to avoid treating hidden hind limbs as visible. Confidence peaks are uncalibrated; evaluate error calibration before altering weights.
3. **Global motion correctness:** compare frame-to-frame root depth / body yaw with independent view data; current single Front view cannot identify all depth degrees of freedom.
4. If still pursuing V2, treat its rotations as an optional **low-weight candidate prior** with a reject gate based on held-out evaluations, but do not tune against this same held-out Side clip and then claim an unbiased validation. Collect independent held-out sequences or physically derived GT for later hyperparameter selection.

**Current milestone: P1 prototype + honest negative result + audit PASS. Model-quality acceptance FAIL.**
