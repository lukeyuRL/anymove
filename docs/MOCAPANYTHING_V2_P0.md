# MoCapAnything V2 �� AnyMove Dino �X P0 Root + Reprojection Audit

Date: 2026-11-01  
Status: **Root bridge implemented and verified; V2 motion quality NOT accepted**  
Scope: existing 5-second H3 split-screen Dino clip, no AI video generation, no retraining.

## 1. Reproducible inputs

- AnyMove: `F:\\anymove`; branch `feature/dino-motion-quality-p123`.
- V2 experimental fork: `F:\\mocapanything_v2`, upstream commit `c99fb22`.
- H3 input: `runs/dino/videos/test_walk/s1/h3max.mp4` (1344��768, 124 frames, 24fps).
- Dino FBX: `assets/dino/rigged/Dinosaur_FullbodyTemplate_fix.fbx` (67 bones).
- Existing AnyMove Reader: `runs/dino/clips/test_walk/det.npz` (124��92 2D landmarks, front+side).
- Existing AnyMove full solver reference: `runs/dino/clips/test_walk/motion.json`.
- V2 original motion: `F:\\mocapanything_v2\\outputs\\anymove_dino_p0\\h3front_124f\\Dino#H3Front\\Dino_Dino_rot6d_pred.bvh`.
- V2 test environment: Python 3.14.3, CUDA PyTorch, RTX 3070 8GB.
- V2 video input was **Front only**. Side observations were NOT supplied to V2.

## 2. Verified technical facts

1. The bone names are identical as sets, but the V2 BVH and AnyMove `rig_info.json` arrays use different order. Join by **name**, never numeric bone index.
2. V2 predicts 3D position in **root-relative normalized coordinates**. Its direct Pose output is NOT an absolute world-space root trajectory.
3. Rest-based rigid orientation registration has RMS error ~`0.0000013m`, reference scale ~`1.0`; the approximate BVH��AnyMove rotation is `(x,y,z) -> (x,-z,y)`. The V2 template reference and exported motion encode a different static root offset; use the **source motion's actual root FK**, not template root OFFSET, when exporting a translation.
4. Existing V2 BVH had **constant root position**. The correction reconstructs world X/Z from the **existing trained AnyMove Front Hip Reader** and keeps Y fixed at the rig's reference depth.
5. The resulting BVH preserves 124 frames, 67 joint names, 24fps, and all local rotations exactly. Roundtrip root FK target max error `7.6��10??m`; Front Hip reprojection max error `3��10??px`.
6. **Root recovery is not V2-only, nor creature-agnostic yet**: it currently relies on the Dino-trained AnyMove Reader. Single-front-view depth is not observable.

## 3. Results �X mean Euclidean 2D reprojection, same 67 matched bone-head landmarks

| Candidate | Front mean px | Side mean px | Training / observations used |
|---|---:|---:|---|
| AnyMove baseline Solver | **4.61** | **4.24** | Front+Side Reader + both-view Solver |
| V2 BVH FK with Front-derived Root | **14.29** | **29.09** | V2 Front video; Front Hip track for root; Side held out |
| V2 raw normalized Position with Front-derived Root | 24.54 | 38.43 | Same input and root correction |

All scores are recomputed against the **original** `det.npz` keypoints, not against the optimized solver's internal coordinates. These are not 3D ground-truth errors, and the AnyMove two-view Solver is **not** an apples-to-apples single-view model.

### Regional errors (mean pixel distance)

| Body region | AnyMove Front | V2 Front | AnyMove Side | V2 Side |
|---|---:|---:|---:|---:|
| Body/root (3 joints) | 4.05 | 2.50 | 2.03 | 7.58 |
| Hind legs / toes (30 joints) | 4.10 | 15.16 | 5.94 | **49.30** |
| Forelimbs (16 joints) | 2.97 | 9.82 | 1.77 | 14.52 |
| Head/neck (10 joints) | 2.37 | **21.21** | 2.18 | 12.34 |
| Tail (8 joints) | 12.81 | 15.78 | 6.23 | 11.43 |

The most severe V2 Side discrepancies are hind toes (>55px mean for several joints). Consequently the main blocker is **local joint pose / out-of-plane articulation**, NOT root depth alone.

A deliberate **non-held-out** diagnostic lets the Side Hip Reader set root Y. V2 Side error then changes only **29.09 �� 28.80 px**. This leaks the test view and must NEVER be quoted as a single-view result.

### Root trajectory & smoothing audit

- Front-derived XZ root displacement from initial: max **0.427m**.
- Raw root XZ total path: **3.848m**, p95 speed **1.851m/s** (includes likely video/Reader jitter).
- 9-frame Savitzky�VGolay smoothing: path 3.306m; p95 speed 1.143m/s, at +0.566px mean Front Hip residual.
- Smoothing was analyzed as an ablation only; no smoothed root exported or production default changed.

## 4. Files on this machine

| Item | Path |
|---|---|
| Root bridge (reusable CLI) | `F:\\anymove\\tools\\mocapanything_v2_root_bridge.py` |
| Bridge unit tests | `F:\\anymove\\tests\\test_mocapanything_v2_root_bridge.py` |
| Corrected 124-frame BVH | `F:\\mocapanything_v2_test\\Dino_V2_front_root_124f.bvh` |
| Root verification JSON | `F:\\mocapanything_v2_test\\Dino_V2_front_root_124f.report.json` |
| Reprojection comparison JSON | `F:\\mocapanything_v2_test\\reprojection_baseline.json` |
| Regional/temporal ablation JSON | `F:\\mocapanything_v2_test\\V2_front_root_ablation.json` |
| Three-frame STATIC comparison image | `F:\\mocapanything_v2_test\\V2_vs_AnyMove_reprojection_3frames.png` |

The static contact sheet has yellow rings (Reader observations), green dots (AnyMove), and magenta crosses (V2). It is a single PNG made from existing H3 frames; **no video was generated**.

## 5. Run commands (from F:\\anymove)

```powershell
python tools/mocapanything_v2_root_bridge.py `
  --v2-repo F:\\mocapanything_v2 `
  --rig-info assets/dino/rigged/rig_info.json `
  --reference-bvh F:\\mocapanything_v2/zoo/characters_face_zplus/Dino/Dino_ffs.bvh `
  --source-bvh 'F:\\mocapanything_v2/outputs/anymove_dino_p0/h3front_124f/Dino#H3Front/Dino_Dino_rot6d_pred.bvh' `
  --detections runs/dino/clips/test_walk/det.npz `
  --out F:\\mocapanything_v2_test/Dino_V2_front_root_124f.bvh

python -m unittest discover -s tests -p test_mocapanything_v2_root_bridge.py -v
python F:\\mocapanything_v2_test/analyze_v2_errors.py
```

Verification: bridge PASS, **4/4 tests PASS**. No iClone retarget, independent 3D GT, silhouette score, or high-quality natural-motion test has been performed.

## 6. Next experimental priority

**P1 �X V2-conditioned AnyMove Solver, not direct V2 export.** Treat V2 FK motion as an initial *regularizer*, never as authoritative truth. The first controlled experiment must optimize only the Front 2D Reader residual plus bone-length/FK and temporal constraints, and use Side as an entirely held-out 2D diagnostic:

```text
loss = robust_front_reprojection
     + lambda_prior * rotation_distance_to_V2
     + lambda_temporal * joint_acceleration
     + lambda_joint * rotation_limit_penalty
```

All branches must share the **same 124 input frames, same Dino Rig, same Front Reader observations, and same video calibration**. Compare direct V2 FK, V2 + Front-only Solver, and a Front-only AnyMove baseline. Report full-clip / per-bone Front, held-out Side, joint limits, foot-contact drift, temporal jitter, and any regressions.

Only after a successful held-out-side test, evaluate full Front+Side fusion and iClone integration. Do not use the full two-view solution as a disguised single-view prior. Stop if solver improvement is solely front-view overfitting or if the V2 prior materially worsens the existing AnyMove pipeline.

## 7. Decision

**Root issue fixed; V2 motion quality currently FAILS as a replacement for AnyMove.**

Root translation can now be exported reproducibly and constrained to Front observations. The next blocker is depth-ambiguous limb articulation and poor generalization of V2 local pose to the H3 Dino clip. Adding more training or generating more H3 footage is not justified by the current evidence.
