# Dino Motion Quality P1–P3 — opt-in implementation and acceptance

Branch: `feature/dino-motion-quality-p123` (based on `iClone_High_Res`).
Scope: improve the **existing** two-view AnyMove Reader → joint detections →
3D FK Solver → `motion.json` → Blender/iClone. Do not change camera layout,
the iClone import contract, Dragon's reference animation, or the original
`reader_hr_v1.pt`. Do **not** enable Dragon's hind-leg temporal association on
the Dino skeleton: it is specialized to named Dragon joints.

## 0. Dataset availability and safety

The Dino meshes, `assets/dino/creature.json`, `assets/dino/rigged/rig_info.json`,
`runs/dino/dataset_hr_v1/`, and `runs/dino/reader_hr_v1.pt` are **local /
external assets**; they are NOT present in the GitHub checkout. In particular,
the GitHub tree cannot establish that an updated dataset was actually generated,
uploaded, or used to train a model. Inspect and audit the actual source directory
before running any job. The earlier Dino report cited **92 keypoints** but this
is not a substitute for re-reading `rig_info.json`; joint count is not bone
count.

On the machine holding the *latest* Dino still-pose dataset:

```bash
cd /workspace/anymove   # or F:/anymove on Windows Git Bash
git switch feature/dino-motion-quality-p123
git pull --ff-only
python -m tools.audit_dino_dataset \
  --dataset runs/dino/dataset_hr_v1 \
  --rig assets/dino/rigged/rig_info.json \
  --creature assets/dino/creature.json --min-poses 25
```

This CPU-only audit checks both RGBA PNG headers and dimensions for each pose,
joint count and finite 2D annotations, unique sample indices, train/validation
presence, blur profile, and a deterministic framing signature that binds the
**exact rig bones/keypoints**, camera layout, and yaw setting. A valid result
means the *static Reader dataset* is internally consistent. It does **not**
certify real self-occlusion labels or continuous animation training samples.
If your dataset is named differently, provide its actual path rather than
copying older data over `dataset_hr_v1`.

## P1 — Temporal Joint Tracking (Reader + LK optical flow)

The existing `mocap.track` tracks independently per view via pyramidal LK and
forward/backward error and produces a new NPZ with original Reader detections
preserved. In this branch, flow extrapolation is **also rejected if the
previous fused joint was more than 7 layout pixels away from the raw LK seed**.
Otherwise applying optical flow measured on a misplaced raw Dino joint to an
already-corrected joint can create spurious drifts. This is an engineering
safety threshold, not a Dino-trained value. Do not equate LK optical flow to
anatomical identity, especially during occlusion.

```bash
export MOCAP_CREATURE=assets/dino
python -m mocap.track VIDEO.mp4 \
  runs/dino/clips/CLIP/det.npz \
  runs/dino/clips/CLIP/det_tracked_p1.npz
python -m mocap.solve \
  runs/dino/clips/CLIP/det_tracked_p1.npz \
  assets/dino/rigged/rig_info.json \
  runs/dino/clips/CLIP/motion_tracked_p1.json
```

Run the **identical video**, frames, rig, calibration and baseline Reader in
both arms. Compare 2D tracking vs known per-frame Blender 2D truth and solve
against known 3D animation if available. Verify identity switches, joint
velocity, paw sliding, reprojection, occlusion recovery, Root trajectory.
Default behavior remains unchanged. Do not promote on reprojection alone.

## P2 — experimental Visibility / Uncertainty Reader

The existing quality head predicts (a) *synthetic cutout* visibility and (b)
uncalibrated predicted 2D localization error. These ARE NOT true Blender mesh
self-occlusion probabilities, Bayesian uncertainty or per-joint calibrated
weights. This branch:

1. Adds `--init-reader` (or `MOCAP_INIT_READER`) to warm-start from a
   verified, compatible **existing Dino inference Reader**. Joint names/order,
   Reader/heatmap resolution and quality-head absence must match. The original
   checkpoint is never overwritten; SHA-256 is recorded for compatible resume.
2. Trains both quality outputs at the Reader's own predicted heatmap position,
   matching the inference sampling location (previously supervision was
   sampled at the labeled point).
3. Computes held-out **Front and Side** error-reliability diagnostics:
   predicted-vs-observed 2D pixel error quantile bins, calibration gap,
   Pearson association and risk among the lowest predicted-error 50%.
   These use static synthetic pose labels and are *diagnostics*, not an
   in-the-loop calibration fit or validation of real video occlusion.

Example after dataset audit (choose a **new** output filename and backup
directory for this experiment):

```bash
export MOCAP_CREATURE=assets/dino
export MOCAP_INIT_READER=runs/dino/reader_hr_v1.pt
export MOCAP_RESUME_DIR=runs/dino/resume_quality_p2_v2
MOCAP_TRAIN_QUALITY=1 bash tools/train.sh dino 20 2 672 192 \
  runs/dino/dataset_hr_v1 runs/dino/reader_hr_quality_p2_v2.pt 0
```

The physical batch, memory and throughput are *not yet GPU-verified*. Reduce
batch or enable `MOCAP_LOW_VRAM=1` if needed (that enables checkpointing and
gradient accumulation); do not change settings mid-resume. The model checkpoint
holds `quality_validation`. No Solver weighting from the quality predictions
is enabled until out-of-distribution real-video tests show reliability;
`front_conf` and `side_conf` retain their original semantics.

True mesh visibility needs a **surface/body-part renderer** or
depth/instance-part visibility labels, with hidden joints distinguished from
visible surface landmarks. A ray cast to a skeleton joint's internal position
would incorrectly classify most joints as hidden.

## P3 — Motion Prior / Motion Denoising

The existing `mocap.motion_prior` learns per-bone/per-axis **angular
acceleration** variance from at least 2 distinct, independently verified,
continuous animations at matching FPS and exact bone order; it is a statistical
MAP prior, **not** a diffusion model. The 2,200 independent Dino training
poses are not eligible, even if sorted into a sequence.

This branch optimizes the exact same second-order MAP objective via
pentadiagonal Cholesky in O(frames) per Euler axis instead of dense matrix
solves, unwraps angular phase to avoid false `+/-pi` acceleration spikes,
rejects duplicate motion-file contents used as fake independent examples,
and preserves Root Translation/Quaternion plus existing rig joint limits.
Output must be a **new** Motion JSON with recomputed FK and reprojection.

Only after reviewing independent trusted Dino animations:

```json
{
  "version": 1,
  "sequences": [
    {"path": "dino_walk_verified.json", "source_kind": "verified_blender_animation", "verified": true},
    {"path": "dino_turn_verified.json", "source_kind": "verified_blender_animation", "verified": true}
  ]
}
```

Paths are **illustrative**. Neither the motion files nor a trained Dino
prior are supplied in this Git checkout. These sources must be known
continuous *3D animations* rather than the same AI-Solver prediction evaluated
as truth.

```bash
python -m mocap.motion_prior train \
  runs/dino/verified_motion/manifest.json \
  runs/dino/prior_verified_v1.json

export MOCAP_CREATURE=assets/dino
python -m mocap.denoise_motion \
  runs/dino/prior_verified_v1.json \
  runs/dino/clips/CLIP/motion.json \
  assets/dino/rigged/rig_info.json \
  runs/dino/clips/CLIP/det.npz \
  runs/dino/clips/CLIP/motion_denoised_p3.json --strength 1.0
```

Do not interpret smoother joint angles as more anatomically correct motion.
The existing stage does not learn new semantic motion, infer ground
contacts, remove a left/right mislabeled detection, or denoise Root Motion.

## Promotion / acceptance gates

| Layer | Objective evidence | Veto condition |
|---|---|---|
| Dataset | audited Rig signature, both PNGs, 2D labels, split, provenance | wrong/missing local Dino rig or mismatched signature |
| P1 | held-out Dino 2D trajectory displacement, ID/occlusion recovery, same-GT 3D joint MAE/P95 | high-confidence wrong-joint drift, severe seam or new foot sliding |
| P2 | Front/Side held-out 2D error and calibration bins + real-video generalization | uncalibrated prediction treated as P(visible) or as Solver weight |
| P3 | clean independent Dino motion training; 3D pose MAE/P95, temporal acceleration, paw/contact drift, Root continuity | training from same solved clip, new FK mismatch, iClone Root regression |
| Integration | same-rig Blender and iClone motion import; exact Bone order/units/FPS | changed iClone adapter contracts or incompatible new skeleton |

First run the CPU regression suite; a GitHub PR to `iClone_High_Res` also
triggers the existing workflow. Neither suite is a substitute for an actual
Dino GPU checkpoint, real video, or Blender/iClone render verification.

```bash
python -m compileall -q mocap blender tools iclone_adapter tests
python -m unittest discover -s tests -p 'test_iclone_*.py' -v
```

Release policy: all P1/P2/P3 variants opt-in, artifact filenames distinct,
raw `det.npz`, `reader_hr_v1.pt` and `motion.json` never replaced until
the stated held-out gates succeed.
