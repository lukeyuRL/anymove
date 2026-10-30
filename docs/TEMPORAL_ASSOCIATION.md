# Temporal Joint Association — experimental Dragon hind-leg solver

Branch: `iClone_High_Res`. This mode is implemented as an opt-in *pre-solve*
track association; it is NOT a trained detector, a 3D ground-truth system or a
general-purpose skeleton retargeter.

## Why and what changes

On the same generated Dragon roar, 672px and 448px readers both show strongly
negative Front-vs-Side **same** hind-paw height correlations but positive
Front-vs-Side **opposite** hind-paw correlations. Those statistics alone do
not establish which view is mislabeled: whole-body movement, occlusion, and
independently generated camera views can also affect correlation.

The new algorithm tests whether the **Side view L/R hind chains** should be
reassigned, with the Front view kept as reference. In contrast to legacy
`--fix-swaps`, it does NOT switch individual joints independently or decide
each frame in isolation. It keeps five related joints together:

- `hip_L/R`, `knee_L/R`, `hock_L/R`, `hpaw_L/R`, `htoe_L/R`

It uses a two-state dynamic program (Viterbi): identity or swap *both entire
Side hind chains*. Observations combine bounded, confidence-weighted
differences in **hips-relative vertical positions** (to remove common root Z
motion), changes over time, and adjacent bone-height differences. State
changes also pay a switching penalty and a penalty for implausible jumps in
the reassigned Side 2D tracks. A cautious prior favors identity. A proposed
swapped segment shorter than eight frames or lacking sufficient average
benefit is rejected. One wrong paw alone cannot dictate a whole-leg swap.

Limitations: no RGB-based visibility recognition; no guarantee that Front is
the correct reference; Reader heatmap peaks are NOT calibrated probabilities;
the observed AI video may truly animate each view differently. The numerical
thresholds are initial experimental values and must be evaluated on more
than one clip.

## Run it using the already generated 124-frame detections

From `F:/anymove` in Git Bash:

```bash
git switch iClone_High_Res
git pull --ff-only
export MOCAP_CREATURE=assets/dragon

python.exe -m mocap.solve \
  runs/dragon/clips/hr_roar/det.npz \
  assets/dragon/rigged/rig_info.json \
  runs/dragon/clips/hr_roar/motion_temporal.json \
  --temporal-association

# Both solvers use the same det.npz. Compare output, not assumed truth.
python.exe tools/compare_visibility.py \
  runs/dragon/clips/hr_roar/motion.json \
  runs/dragon/clips/hr_roar/motion_temporal.json
```

This creates `motion_temporal.json` and the additional diagnostic file
`motion_temporal.association.json`. The original `det.npz`,
`motion.json`, `reader_hr.pt`, and source video are not overwritten.
The sidecar contains the exact profile, every frame's chosen state, accepted
swap intervals, rejected short/weak proposals, relative-height MAE and
hind-paw correlation before/after, original/corrected paw agreement,
Source Detection SHA-256 and Solver reprojection diagnostics.

The corrected association is applied to the in-memory Side UV **and matching
confidence** arrays before triangulation and optimization; Front, torso,
wings, tail and the original input remain untouched. The final motion retains
the original `fps`, `bones`, `trans`, `root_quat`, `euler`, `joints`
and all existing JSON schemas. The old `--fix-swaps` and experimental
`--visibility-aware` cannot be combined with temporal association because
their overlapping assumptions have not been validated.

For a complete video capture in a **separate** output directory (including
new renders) rather than only re-solving the old detections:

```bash
MOCAP_TEMPORAL_ASSOCIATION=1 bash tools/capture.sh dragon \
  F:/anymove_test/dragon/01_clip/capture_dragon_ai_prompt_001/generated/h3max.mp4 \
  hr_roar_temporal
```

The iClone plugin's capture worker does not silently enable this experimental
mode. If this mode is promoted later, it needs separate end-to-end QA.

## What does a successful A/B comparison mean?

1. Inspect `accepted_swapped_frames` and the start/end of each segment. If
   zero, the algorithm has conservatively retained the identity assignment;
   that is **not** an error.
2. Compare root-relative hind-chain height MAE and the corrected paw curves.
   Improved cross-view agreement may validate a *correspondence hypothesis*
   but does not prove anatomical correctness.
3. Compare the original and temporal `motion.json` using the existing
   comparison helper. Joint differences and reprojection changes are not
   ground-truth error improvements.
4. Render the new 3D Motion and evaluate actual feet, wings and root motion
   against the generated clip. Test a known Blender 3D ground-truth motion.
5. Use a clean iClone Dragon scene to check FK, Root Translation (upwards
   positive), root multiplier 1, feet and wings before considering defaulting
   to the temporal mode.

**Safety gate unchanged:** the `motion.json` paw agreement continues to
report the **unmodified original Reader detections**. The iClone Capture
quality gate for scores below 0.5 is NOT bypassed even when new associated
curves look better. The association sidecar separately reports corrected
agreement for diagnosis. No new AI video request, model training, or changes
to Root Flight are required for the above A/B run.

## CI coverage

The NumPy CI job runs `tests/test_iclone_temporal_association.py` on
synthetic clean and deliberately swapped sequences: full-clip and partial
swaps, short flicker, confidence handling, whole-chain grouping, unrelated
joint immutability, root-relative comparison and JSON metadata. This does
not replace a real 124-frame clip test or a 3D ground-truth accuracy test.


## Frame-by-frame 3D diagnostics for 41–52 association result

For the specific 124-frame `hr_roar` example, the first accepted candidate
reassigns the Side hind chains in frames `[41, 53)`. This means two
potentially sensitive **transitions**: frame 40→41 and frame 52→53.
The average A/B joint difference alone can hide a large difference in only a
few hind joints or frames.

The new tool measures *actual solved joint position differences*, split
between hind joints and the other 40 joints, as well as Root Translation,
Root Quaternion angular changes, and first/second finite differences at
association boundaries. It is not a 3D ground-truth or visual quality metric.

From the checked-out `iClone_High_Res` branch in Git Bash:

```bash
git pull --ff-only

python.exe tools/diagnose_temporal_motion.py \
  runs/dragon/clips/hr_roar/motion.json \
  runs/dragon/clips/hr_roar/motion_temporal.json
```

The tool reads `motion_temporal.association.json` automatically. It fails
closed for mismatched FPS, joint order, source detection SHA or camera layout.
It does **not** modify either input motion or the association report.

Outputs in `runs/dragon/clips/hr_roar/motion_temporal_diagnostics/`:

- `temporal_3d_report.json`: summary, top changed joints and frames,
  Root peak frame, and seam-transition velocity/acceleration comparisons
- `temporal_3d_by_frame.csv`: 124 frame rows, Root/whole-skeleton/hind
  position deltas, and step-vector differences
- `temporal_3d_by_joint.csv`: for each named joint, mean/P95/P99/max
  delta, frame of maximum, inside-vs-outside swapped interval
- `temporal_3d_by_joint_frame.csv`: all 124×50 joint/frame records,
  position, step and acceleration differences
- `temporal_3d_plot.png`: overlay of hind vs other joints, Root
  differences and per-step hind-joint changes; requires matplotlib

The frames and time markers are **zero-based**, 24 FPS. The swap begins
at 41/24 = 1.7083s and ends at 53/24 = 2.2083s. The frame samples selected
in the accepted region are 41 through 52 inclusive. Inspect their neighbors
because the switch-back changes the step from 52 to 53.

To run with an explicit association file or output directory:

```bash
python.exe tools/diagnose_temporal_motion.py \
  runs/dragon/clips/hr_roar/motion.json \
  runs/dragon/clips/hr_roar/motion_temporal.json \
  --association runs/dragon/clips/hr_roar/motion_temporal.association.json \
  --out-dir runs/dragon/clips/hr_roar/temporal_ab_report
```

**Interpretation:** A spike at frame 41 or 53 is evidence of a *candidate*
solver seam discontinuity, not proof it is visually wrong; compare the
original and temporal animations and, where possible, a known true motion.
The report does not count lower reprojection error as accuracy improvement.
Avoid promoting temporal association into the default iClone workflow until
its root and hind-limb trajectories are visually and quantitatively checked.


## v2: Visibility-aware evidence + candidate 3D continuity quarantine

The earlier `motion_temporal.json` candidate from the roar clip changed left
hind toe position by up to **62.02 cm** near frame 43. At the 40→41 seam,
mean hind step length rose from **3.06** to **7.78 cm/frame** (2.54×), while
the proposed 2D association improved its own height-MAE objective by only
about 8%. A large 3D pose correction is not by itself proof of an error, but
a new velocity discontinuity at the switching boundary is strong evidence
for an unsafe **automatic** reassociation.

The revised opt-in `--temporal-association` now has **two separate gates**:

1. **Image-space reliability (not an occlusion model).** For each Front hind
   landmark, measure proximity to Front foreleg landmarks and whether it is
   clearly distinct from foreleg landmarks in the Side view. When the Front
   landmark looks potentially occluded, reduce its evidential weight in
   *both* Viterbi states. Require multiple independently contributing
   hind-chain joints and enough mean visibility. These are only *proxies*;
   raw DINO heatmap peaks are not calibrated visibility probabilities.
2. **Actual solved 3D continuity.** Always solve the original aligned
   detections first. For every accepted 2D swap interval, form a trial
   detection set with **both entire Side hind chains and their matching
   confidences** exchanged over that interval. Solve again with the same
   rig and optimization. At the two interval boundaries, check whether
   average hind-leg velocity increases too sharply relative to the original
   solve, whether any hind joint changes velocity by more than the guarded
   amount, and whether Root Translation or hind acceleration increases
   abnormally. Root Rotation is also checked for sudden angular step
   increases at each association boundary. If the segment fails, **quarantine it and keep the original
   baseline solved pose**; no blending that could create new invalid
   skeleton states. Inspect the audit to understand *why*.

The first gate is a heuristic, not a ground-truth visibility classifier. The
second gate checks continuity relative to the original solved motion, not
anatomical correctness. A true large repair may be quarantined and require
manual review; passing the gate does not establish the swap is correct.
The experiment stays off by default and the iClone Paw Agreement quality
gate based on original detections remains enforced.

### Reproduce the new guarded solve without overwriting old files

```bash
git switch iClone_High_Res
git pull --ff-only
export MOCAP_CREATURE=assets/dragon

python.exe -m mocap.solve \
  runs/dragon/clips/hr_roar/det.npz \
  assets/dragon/rigged/rig_info.json \
  runs/dragon/clips/hr_roar/motion_temporal_guarded.json \
  --temporal-association
```

The diagnostics are written to
`runs/dragon/clips/hr_roar/motion_temporal_guarded.association.json`.
Key fields:

- `pre_3d_candidate_segments`: intervals that passed the visibility-aware
  2D Viterbi stage
- `continuity_validation.status`: `all_candidates_passed`,
  `partially_applied`, or `all_candidates_quarantined`
- `continuity_validation.checks[]`: each interval's baseline/candidate
  3D step lengths, Root changes, acceleration and failure reasons
- `accepted_swapped_frames` and `state_by_frame`: only intervals
  **actually applied** to the final motion
- `pre_3d_candidate_paw_agreement` and
  `aligned_paw_agreement_after`: distinguish hypothetical vs applied
  metrics; the original `sync.paw_agreement` is not modified
- `pre_3d_candidate_relative_hind_height_mae_px` and
  `after_relative_hind_height_mae_px`: distinguish proposed vs applied
  height consistency

The original `motion.json` and the prior unsafe A/B
`motion_temporal.json` remain intact. If the candidate is quarantined,
the new `motion_temporal_guarded.json` is the baseline Solver's actual
motion (not a blend or the rejected candidate), with unchanged keys and
source detection SHA. Baseline fit and each trial are run with the same
Solver configuration; the experimental mode may require multiple fitting
passes.

To summarize the result:

```bash
python.exe - <<'PY'
import json
p = 'runs/dragon/clips/hr_roar/motion_temporal_guarded.association.json'
x = json.load(open(p))
print('2D candidates:', x['pre_3d_candidate_segments'])
print('3D guard:', x['continuity_validation']['status'])
for s in x['continuity_validation']['checks']:
    print(s['start_frame'], s['end_frame_exclusive'],
          s['status'], s['failure_reasons'])
print('Applied frames:', x['accepted_swapped_frames'])
PY
```

For the *known numerical boundary* from the previous report, the
new 3D seam test is specifically designed to quarantine a change of
3.06→7.78 cm/frame. This is checked as a controlled synthetic test,
not yet as a live rerun of the user's private 124-frame `det.npz`;
that original file is only on the user's Windows computer.


## v3: avoid treating persistent projection geometry as occlusion

**Field finding (124-frame high-resolution Dragon roar):** The first v2
front-overlap proxy assigned very low weight to most knees, hocks and paws
for almost the entire clip, not just during the questionable frames:

- `knee_L`: 118/124 frames below 0.6
- `hock_L`: 124/124 frames below 0.6
- `hpaw_L`: 121/124 frames below 0.6
- `hock_R` and `hpaw_R`: 124/124 frames below 0.6

This is systematic downweighting, likely confounded by the fixed two-camera
anatomy/projection. v2 then produced **zero** 2D candidates, which meant
there was no 3D seam trial. Its status field `all_candidates_passed` was
misleading: an empty list trivially passed. This result was **not** an
empirical validation of the continuity guard.

v3 only treats Front foreleg proximity as an *occlusion hypothesis* if the
hind landmark **temporarily becomes substantially closer** than its own
typical Front foreleg separation within the clip. A per-joint 85th-percentile
reference and a minimum separation spread are used; persistent static
overlap no longer triggers near-universal suppression. Side-view separation
must still strengthen the hypothesis. This is deliberately not a true
occlusion model and does not establish which camera is right.

The Association JSON now records each joint's reference separation, temporal
spread, number of transient overlap frames and the resulting weights.
`unweighted_2d_reference` records what the historical unweighted 2D
Viterbi would have proposed *on the same detections* (diagnostic only).
The main algorithm **does not apply the unweighted proposal** unless it
independently passes its own visibility-aware 2D stage and the 3D seam gate.

For **no 2D candidates**, the continuity result now explicitly says
`not_run_no_candidates`, with `continuity_checks_run: 0` and
`validation_skipped_reason: no_2d_candidates`. Never interpret it as
a successful 3D safety validation.

Use a fresh filename; preserve the previous v1 and v2 experiments:

```bash
git pull --ff-only
export MOCAP_CREATURE=assets/dragon

python.exe -m mocap.solve \
  runs/dragon/clips/hr_roar/det.npz \
  assets/dragon/rigged/rig_info.json \
  runs/dragon/clips/hr_roar/motion_temporal_v3.json \
  --temporal-association
```

Inspect `motion_temporal_v3.association.json`:

```bash
python.exe - <<'PY'
import json
p = 'runs/dragon/clips/hr_roar/motion_temporal_v3.association.json'
x = json.load(open(p))
print('2D candidate segments:', x['pre_3d_candidate_segments'])
print('Unweighted 2D reference:', x.get('unweighted_2d_reference'))
print('3D guard result:', x['continuity_validation']['status'])
print('3D checks:', x['continuity_validation']['checks'])
print('Applied frames:', x['accepted_swapped_frames'])
for joint in ('knee_L','hock_L','hpaw_L','knee_R','hock_R','hpaw_R'):
    print(joint, x['visibility_diagnostics']['joint_summary'][joint])
PY
```

The original 12-frame candidate from v1 had an identified
40→41 seam increase from 3.06 to 7.78 cm/frame, a change targeted by
the existing 3D continuity gate. v3 may propose **different** intervals or
none at all; only a real solver run can establish which candidates survive
both stages. This test still does not prove 3D anatomical correctness.
