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

### P1 first real Dino clip: 124-frame preliminary results (reported 2026-11-01)

The user-provided comparison for the *same 124-frame video*, using the
existing `compare_visibility.py`, reported mean inter-solution joint delta
**0.9728 cm**, P95 **3.7854 cm**, Root mean delta **1.2753 cm** and Root max
delta **4.7151 cm**. The baseline's Front/Side per-run reprojection was
4.46174/4.48182 px; the tracked variant's was 4.46021/4.48385 px.
These values are observations supplied in chat, not independently re-read
from the local Dino files by CI.

**Critical interpretation:** the original comparison printed
`same-detection solver ablation`, although `mocap.track` generates an
independent source NPZ and can change BOTH UV and Solver weights. Moreover
both per-run reprojection figures use different observations. They do not
establish that tracking improved 3D accuracy, nor even which P1 effect
caused the Root displacement. `paw_agreement.score=null` is not zero;
the available score was simply not evaluable.

The updated `tools/compare_visibility.py` now inspects
`source_detections_sha256` instead of assuming identical input data.

#### Next run: common RAW Reader reference and Root dynamics

Use the actual same-capture `det.npz` and its P1 tracked counterpart, with
both corresponding motion files, after `git pull --ff-only`. The tool
verifies source SHA-256, Rig camera layout, joint order/FPS, original
Reader UV/peaks, tracked weights and original video calibration, and
reports changes including Root velocity and acceleration.

```bash
export MOCAP_CREATURE=assets/dino
CLIP=runs/dino/clips/REPLACE_WITH_CAPTURE_NAME
python.exe -m tools.diagnose_dino_tracking \
  "$CLIP/det.npz" "$CLIP/det_tracked_p1.npz" \
  "$CLIP/motion.json" "$CLIP/motion_tracked_p1.json" \
  assets/dino/rigged/rig_info.json \
  --out "$CLIP/p1_dino_diagnostics.json"
```

**Windows Git Bash:** run these commands from the repository root (`cd /f/anymove`)
and keep `MOCAP_CREATURE=assets/dino` set. The `python.exe -m tools...`
form avoids `ModuleNotFoundError: No module named 'mocap'` caused by direct
execution putting `tools/` rather than the repo root on Python's import path.
Direct invocation is also supported by these two scripts after the CLI fix.

The report's `shared_raw_reader_reprojection_px` evaluates BOTH motions
against the SAME raw Reader joint observations, which is a fairer
*common-reference* comparison. The original Reader can itself be wrong;
lower error against it does **not** imply lower 3D ground-truth error.
Inspect `worst_joint_mean_deltas_cm`, `max_root_difference_frame`,
`root_dynamics`, and the P1 tracking counts (especially `flow_only`,
`outlier_flagged`, and `flow_seed_rejected`).

#### P1 v2 opt-in guard for large uncertain Flow/Reader disagreements

The 124-frame Dino report found maximum 2D adjustments of **169.18 px**
(Front Tail07, frame 12) and **76.15 px** (Side L_LeftToe03, frame 82),
and a maximum 3D inter-solution joint deviation of **47.95 cm**
(R_RightToe03). Root differed most at frame 82 (**4.715 cm**).
These are *between-solution distances*, not 3D ground-truth errors.
The same-raw-Reader reprojection means worsened slightly in both views.
The root maximum and side toe maximum sharing frame 82 is a correlation,
**not established causation**.

P1 v1's low-peak recovery may replace the raw Reader point with LK flow
irrespective of how far the candidates disagree, even beyond 100px.
This is the only path that can produce very large UV changes: the normal
Reader/LK agreement branch moves <= 0.35 * 9 = 3.15px, and the
high-peak-disagreement branch retains the raw Reader UV. A reliable
forward/backward LK check is not a semantic identity guarantee.

P1 v2 is an **opt-in** `--guard-large-disagreements` mode. When a low-peak
reader and flow-only prediction differ by >30 layout pixels, it leaves
the original Reader coordinate but reduces that observation's Solver
weight multiplier to **0.10**, records `quarantined_large_disagreement`,
and resets the flow-only streak. Both 30px and 0.1 are conservative
engineering hypotheses, not optimized Dino thresholds. Importantly, the
raw Reader *may also be wrong*; the guard is an abstention, not a
claimed tracking improvement.

It does not change P1 v1's existing results. For a **new** complete
capture (including Solver/Blender import preview), the same optional gate
is reachable through `MOCAP_TEMPORAL_TRACKING=1
MOCAP_TEMPORAL_TRACKING_GUARDED=1 bash tools/capture.sh dino ...`.
Use a new capture name and preserve the old baseline files.

On the exact same video,
create distinct artifacts and compare to the original baseline + P1:

```bash
export MOCAP_CREATURE=assets/dino
VIDEO=runs/dino/videos/test_walk/s1/h3max.mp4
CLIP=runs/dino/clips/REPLACE_WITH_CAPTURE_NAME
python.exe -m mocap.track "$VIDEO" "$CLIP/det.npz" \
  "$CLIP/det_tracked_p1_guarded.npz" --guard-large-disagreements

python.exe -m mocap.solve "$CLIP/det_tracked_p1_guarded.npz" \
  assets/dino/rigged/rig_info.json "$CLIP/motion_tracked_p1_guarded.json"

python.exe -m tools.diagnose_dino_tracking \
  "$CLIP/det.npz" "$CLIP/det_tracked_p1_guarded.npz" \
  "$CLIP/motion.json" "$CLIP/motion_tracked_p1_guarded.json" \
  assets/dino/rigged/rig_info.json \
  --out "$CLIP/p1_dino_guarded_diagnostics.json"
```

Verify the v2 `.tracking.json` contains a nonzero quarantine count
for the problematic candidates and assess whether max 2D change and Root
difference change. If no candidates are quarantined, the hypothesized
outlier path was not the cause. Neither a lower reprojection error nor a
smaller between-solution distance certifies better 3D motion.

### Dino test_walk P1 Guarded v2 actual follow-up (124 frames)

The user-provided `p1_dino_guarded_diagnostics.json` reports the same
124 frames, 24fps, 92 joints and same RAW Reader Detection hash
`0d51a6b2c3b1e7fa61a561eae91820a53d1f6eb0f15fa042c88867cf9eec56e4`.
The comparison is between **different detections** with a common raw
Reader reprojection reference; neither result is 3D truth.

| Metric | P1 v1 | P1 guarded v2 | Interpretation |
| --- | ---: | ---: | --- |
| Front maximum UV change | 169.17697 px | 29.84275 px | 82.4% smaller extreme modification |
| Side maximum UV change | 76.14707 px | 29.97502 px | 60.6% smaller |
| Inter-solution Joint max | 47.95042 cm | 15.69275 cm | 67.3% smaller vs baseline, **not** truth |
| Inter-solution Joint mean | 0.97278 cm | 0.84462 cm | 13.2% smaller vs baseline |
| Root mean vs baseline | 1.27534 cm | 1.26223 cm | effectively unchanged |
| Root maximum vs baseline | 4.71506 cm | 4.78629 cm | **slightly worse**, still at frame 82 |

**Confirmed safety effect:** the guard quarantined 27 Front + 10 Side
low-peak conflicts (>30 px). No applied 2D correction exceeded 30px.
Only 596/11,408 Front and 378/11,408 Side joint/frames exceeded 1px;
"changed_joint_frames" counts tiny floating-point changes too.

Against the *same raw Reader detections*, baseline Front/Side mean
reprojection was 4.46175 / 4.48184 px and P1 guarded v2 was
4.45385 / 4.50593 px. Front improves only 0.0079px while Side
worsens 0.0241px; there is no consistent accuracy signal. Front X-only
increases from 1.64135 to 1.67240px. Root peak speed
130.98295 -> 132.23137 cm/s and Root P95 acceleration
936.43260 -> 886.34387 cm/s², but smoother does not mean more correct.

**Decision:** keep both P1 variants opt-in; guarded v2 is a safer
*extreme-correction gate*, not a validated temporal joint tracker.
The persistent Root difference at frame 82 is not explained away by
the outlier guard. Do not tune or enable default on this one AI clip.
Reserve P1 UV-only/weight-only factorial runs and independent Blender
continuous-motion ground truth for follow-up. Advance to P2 quality
supervision on the verified 2,200 independent Dino still poses.

#### Separate coordinate and weight contributions

If Root changes look substantial, use the four-way factorial test:

| Run | 2D coordinates | Solver weight multiplier |
|---|---|---|
| Baseline (existing) | Original Reader | 1 |
| P1 UV-only | Tracked | 1 |
| P1 Weight-only | Original Reader | P1 weights |
| Full P1 (existing) | Tracked | P1 weights |

```bash
python.exe -m tools.make_dino_tracking_ablations \
  "$CLIP/det.npz" "$CLIP/det_tracked_p1.npz" \
  "$CLIP/p1_ablation"

python.exe -m mocap.solve \
  "$CLIP/p1_ablation/det_p1_uv_only.npz" \
  assets/dino/rigged/rig_info.json \
  "$CLIP/p1_ablation/motion_uv_only.json"

python.exe -m mocap.solve \
  "$CLIP/p1_ablation/det_p1_weight_only.npz" \
  assets/dino/rigged/rig_info.json \
  "$CLIP/p1_ablation/motion_weight_only.json"
```

These two extra fits isolate the UV effect, the weight effect, and their
interaction without retraining. None of the four solutions is true 3D
ground truth; review known Blender 3D animation sequences and film
visually before promoting P1. In particular, avoid tuning the LK
thresholds based only on changes in reprojection from an AI-generated clip.

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

### P2.1 Recommended: frozen baseline Reader, train new Quality Head only

The first Quality experiment should NOT update an already verified
`reader_hr_v1.pt` backbone/heatmap decoder. This branch adds
`MOCAP_QUALITY_ONLY=1` (`--quality-only`) which requires
`MOCAP_TRAIN_QUALITY=1` and a compatible `MOCAP_INIT_READER`. It freezes
**every** original Reader weight and trains only `quality_out.weight` and
`quality_out.bias`. During training the pretrained Reader stays in eval
mode, and no gradient is computed for the frozen heatmap MSE objective.
Resume metadata records `quality_only` and the original checkpoint SHA.

P2 also now evaluates visibility on a **separate deterministic held-out
cutout validation set**. The previous clean val set contains no negative
visibility labels and could not measure AUC. New checkpoint
`quality_validation.synthetic_cutout_visibility` reports AUC, Brier and
positive/negative sample counts in addition to the clean held-out
Front/Side predicted-vs-actual localization error calibration.
These are artificial patch labels, **not** real mesh occlusion or calibrated
video probabilities.

First create a non-destructive 30-pose subset from the **audited latest
2200-pose Dino source** (28 training + 2 held-out). Do not reuse an older
smoke directory or retrain the original Reader:

```bash
cd /f/anymove
git pull --ff-only
export PYTHON=python.exe
export MOCAP_CREATURE=assets/dino

python.exe -m tools.make_dino_quality_smoke \
  runs/dino/dataset_hr_v1 \
  runs/dino/dataset_hr_v1_p2_smoke --poses 30

python.exe -m tools.audit_dino_dataset \
  --dataset runs/dino/dataset_hr_v1_p2_smoke \
  --rig assets/dino/rigged/rig_info.json \
  --creature assets/dino/creature.json
```

Run one epoch on local RTX 3070 8GB as an **integration smoke test**;
actual P2 VRAM/runtime is unverified. With the base frozen, gradient
checkpointing is unnecessary, but batch 1 is still safest:

```bash
export MOCAP_INIT_READER=runs/dino/reader_hr_v1.pt
export MOCAP_TRAIN_QUALITY=1
export MOCAP_QUALITY_ONLY=1
unset MOCAP_RESUME_DIR
bash tools/train.sh dino 1 1 672 192 \
  runs/dino/dataset_hr_v1_p2_smoke \
  runs/dino/reader_hr_quality_p2_smoke.pt 0
```

#### 2026-11-01: Dino RTX 3070 P2 one-Epoch smoke result

Actual user-reported run: 28 train / 2 validation static poses, 92 joints,
672/192 Reader, BF16, physical batch 1, 56 optimizer steps, original
`reader_hr_v1.pt` warm-started and frozen, only
`quality_out.weight` / `quality_out.bias` trainable. GPU peak allocated
0.44 GiB (peak reserved 0.54 GiB), and the separate
`reader_hr_quality_p2_smoke.pt` was saved.

The smoke validation measured Front 1.73px / Side 1.71px positional
residual, localization-error calibration gap Front 1.22px / Side 1.56px,
and **synthetic held-out cutout AUC 0.52418**, near the chance value 0.5.
These were only **2 validation poses** (4 images) and one epoch, so they
establish *pipeline execution*, NOT a useful learned visibility model.
The validation errors are not comparable with the earlier full dataset's
Front 3.41px / Side 2.20px because the validation subset differs.
The 1-epoch cosine schedule reaches very low learning rate near the end,
so its AUC is not evidence against a normal 20-epoch experiment.

**Next:** Full 2,112-train / 88-validation static pose run with the same
audited dataset. Keep 20 total epochs fixed for compatible full-state
resume. For Windows, workers 0 avoids worker-spawning issues; batch 2
is a conservative starting experiment after the batch-1 smoke. Set
`MOCAP_LOG_FILE` to keep the smoke log.

```bash
cd /f/anymove
git pull --ff-only
export PYTHON=python.exe
export MOCAP_CREATURE=assets/dino
export MOCAP_INIT_READER=runs/dino/reader_hr_v1.pt
export MOCAP_TRAIN_QUALITY=1
export MOCAP_QUALITY_ONLY=1
export MOCAP_RESUME_DIR=runs/dino/resume_quality_p2_headonly
export MOCAP_KEEP_CHECKPOINTS=3
export MOCAP_LOG_FILE=runs/dino/train_quality_p2_headonly.log
bash tools/train.sh dino 20 2 672 192 \
  runs/dino/dataset_hr_v1 \
  runs/dino/reader_hr_quality_p2_headonly.pt 0
```

Record every held-out AUC, Brier, number of synthetic masked joints,
front/side calibration gap and unchanged clean-location error. The
Brier score may look good merely because most joints were not covered by
an artificial rectangle; do not ignore AUC and the visible/masked counts.
**Do not wire quality predictions into Solver weights** based on synthetic
cutout AUC alone. Full GPU training was not started by this PR.

If the smoke test saves and loads, begin the full **separate**
quality-only experiment with a fresh output and dedicated resume directory.
The full run requires a GPU and may incur cost if using Runpod; check
current GPU availability and cost before starting:

```bash
export MOCAP_RESUME_DIR=runs/dino/resume_quality_p2_headonly
bash tools/train.sh dino 20 2 672 192 \
  runs/dino/dataset_hr_v1 \
  runs/dino/reader_hr_quality_p2_headonly.pt 0
```

Acceptance: matched 92-keypoint order, `672/192` spec, unchanged
pretrained Reader heatmap state dict, held-out clean Front/Side 2D error
stays at the same checkpoint baseline (previous Dino report: approximately
3.41px Front / 2.20px Side), and meaningful held-out synthetic
visibility AUC/Brier + residual calibration. AUC is undefined if all
synthetic labels fall in one class. Even successful AUC does not license
turning new quality scores into Solver visibility probabilities.

The former joint fine-tuning configuration (without
`MOCAP_QUALITY_ONLY=1`) remains available as an opt-in research
ablation, but it can change and degrade the original Reader. Train it only
to a NEW checkpoint and test 2D accuracy against the frozen baseline.

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

## P2.2 — ACTUAL Dino Reader + Motion Solver A/B (same 2D observations)

The P2 smoke/full Quality Head was trained with `MOCAP_QUALITY_ONLY=1`.
Its original DINOv2 heatmap/coordinate Reader was **frozen**. Running the
P2 checkpoint by itself can display predicted synthetic-cutout scores and
2D-error estimates, but cannot physically change keypoint coordinates.
This experiment performs the missing integration: the **trained P2 Reader
predicts errors on every real video frame, and its predictions feed the
EXISTING 3D Solver as experimental per-joint confidence multipliers**.

**Important distinction:** The P2 output is not a trained anatomical
occlusion probability. The proposed mapping is a conservative,
configurable heuristic, not a learned noise distribution:

```text
multiplier = clip(5px / max(predicted_error_px, 5px), 0.40, 1.0)
```

This uses **predicted error only** (NOT synthetic-cutout score). The
original `front_uv/side_uv/front_conf/side_conf` are copied exactly
from the user's proven baseline `det.npz`, so any changed 3D motion is
attributable to these experimental weights, not changed 2D coordinates.
The importer and `motion.json` contract are unchanged. The adapter
will not be enabled by default unless true 3D/visual tests pass.

Use a stable model-only checkpoint snapshot copied between atomic
per-epoch saves. Don't `git pull` or interrupt an actively running
`mocap.train` process; the branch must be updated **before** starting
training or after it exits, never while rewriting the working tree.
Inference on the same RTX 3070 while training may compete for GPU time;
prefer running this experiment after training finishes or is paused.

From Git Bash with the new changes checked out:

```bash
cd /f/anymove
export PYTHON=python.exe
export MOCAP_CREATURE=assets/dino
# Keep original training checkpoint and current output untouched.
cp -n runs/dino/reader_hr_quality_p2_headonly.pt \
  runs/dino/reader_hr_quality_p2_eval_snapshot.pt

# Replace CLIP with the EXISTING capture that generated the raw det.npz
# and baseline motion.json, e.g. runs/dino/clips/test_walk.
VIDEO=runs/dino/videos/test_walk/s1/h3max.mp4
CLIP=runs/dino/clips/REPLACE_WITH_EXISTING_CAPTURE
bash tools/eval_dino_p2.sh "$VIDEO" "$CLIP" \
  runs/dino/reader_hr_quality_p2_eval_snapshot.pt p2_eval01
```

The one-command script runs, in sequence:
1. `mocap.detect` with actual P2 Reader to a new
   `p2_eval01/det_p2_quality.npz` and joint overlay video.
2. `tools.review_dino_p2` makes `quality_review.mp4` with colors
   representing predicted error (<=5 px, 5-12 px, >12 px), a purple ring
   for low **synthetic** cutout score, and per-joint/ per-frame error
   rankings in `quality_review.json`. These color thresholds are only
   display heuristics, not real occlusion labels.
3. `tools.make_dino_p2_weighted_detection` strictly checks that P2
   Joint Reader UV/peaks, frame count, names, FPS and video affine
   calibration still match the original Reader. It refuses mismatches
   >0.25px UV or >0.02 raw heatmap peak before copying the original
   UV/peaks and adding bounded `front_track_weight` and
   `side_track_weight`. The existing Solver consumes these optional
   arrays as confidence multipliers.
4. `mocap.solve` produces `motion_p2_weighted.json` in a separate
   directory. This does NOT affect the original `motion.json`.
5. `tools.compare_dino_p2_motion` checks input SHA/rig provenance,
   evaluates BOTH solved motions against the SAME raw Reader 2D
   observations, and reports Root trajectory differences, speed and
   acceleration plus per-joint 3D changes. These are NOT 3D accuracy
   measurements.

Inspect `quality_review.mp4` alongside the **actual source** video,
then render `motion_p2_weighted.json` on the Dino rig separately and
compare against the original `motion.json` in Blender/iClone.
Pay attention to frame 82, toe sliding, Tail07 and Root movement.
If all predicted errors are <=5px the mapping leaves all weights at
one, so the two Motion results should be indistinguishable; do not
claim the P2 quality head helped if no observations were downweighted.

This is an experiment, not a production integration. If P2 uncertainties
do not generalize from synthetic to AI video, revise supervision or
independently fine-tune the `JointReader` Heatmap Head on held-out
realistic Dino detection data in another checkpoint; do not
unfreeze the production Reader in place.

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
