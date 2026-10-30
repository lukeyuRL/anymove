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
