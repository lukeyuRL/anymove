# Dragon Parent/Child FK Convention: A/B controlled experiment

## Why this test

The old `dragon_preview` wrote Blender Euler XYZ values directly to
`RIClip.GetControl("Layer", bone).GetDataBlock().GetControl("Rotation/RotationX/Y/Z")`.
The actual iClone Dragon successfully stored 41x3x124 FK keys and read them
back accurately. However same-frame iClone vs Blender reference screenshots
show wings and neck/tail disagree (including matched front-frame 26).

An iClone **FK Layer key** sets an absolute numerical channel value at its
frame; it is NOT itself a world-space absolute bone orientation. The
parent-child hierarchy, rest-bone basis and optional other IK/motion layers
affect evaluated final pose. Official iClone 8 docs:
https://wiki.reallusion.com/IC_8_Python_API:End_Effector_Animation

Blender AnyMove FK computes
`P_child = P_parent @ O_child @ R_child` where
`O_child = inverse(B_parent) @ B_child` and Blender local
`R_child = Rz @ Ry @ Rx`.

**Hypothesis A:** iClone expects `R_child` directly in its Layer XYZ.
**Hypothesis B:** iClone might apply Layer rotation before the rest
offset (`R_iClone @ O_child`) and thus require
`R_iClone = O_child @ R_child @ inverse(O_child)`.

**These are hypotheses**. This experiment cannot automatically prove
the iClone internal multiplication order. Several axes and Euler conventions
could also be involved; inspect actual rendered poses before adopting B.

## 1. Generate controlled test JSONs with Python 3.14.3

Open PowerShell:

```powershell
cd F:\anymove
git pull --ff-only origin iClone

py -3.14 -m iclone_adapter.parent_child_probe `
  F:\anymove_test\skeleton.json `
  F:\anymove_test\wing_probe `
  --parent wing_arm_L --child wing_fore_L
```

Four files appear in `wing_probe`:
- `probe_source_motion.json`: original, controlled Blender motion with
  only the two selected bones animated. Pelvis root rotation and translation
  are identity/zero throughout; all other bones stay at rest.
- `probe_fk_raw.json`: iClone candidate A (direct Euler).
- `probe_fk_rest_conjugate.json`: iClone candidate B (relative rest basis
  conjugation).
- `probe_manifest.json`: all frame labels, IDs and basis quaternions.

**No PyTorch** needed for generation; standard Python 3.8+ suffices.
Do not change the validated production `dragon_mapping.json`.

## 2. Produce the matching Blender reference (no Solver)

```powershell
$blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"

& $blender -b --python-exit-code 1 -P blender/apply_motion.py -- `
  assets/dragon/rigged/rigged.blend `
  assets/dragon/rigged/rig_info.json `
  F:\anymove_test\wing_probe\probe_source_motion.json `
  F:\anymove_test\wing_probe\blender `
  --no-root-motion --save-blend
```

Open `F:\anymove_test\wing_probe\blender\animation.blend` in
Blender. This is an actual keyframed 169-frame reference.

## 3. Compare two **clean iClone** scenes

- Scene A: load the original, **unanimated** Dragon FBX into a new
  **disposable** iClone project at **24 FPS**, select Dragon, timeline
  Frame 0. In `launch_iclone.py` click
  **Write FK motion to selected avatar (modifies scene)**,
  select `probe_fk_raw.json`. Confirm TEST ONLY and inspect frames.
- Record screenshots, then **re-open the unmodified original iClone scene
  or create a NEW clean scene**. DO NOT write B on top of A's clip.
- Scene B: repeat with `probe_fk_rest_conjugate.json`. Both files retain
  an explicit uncalibrated warning.
- Frame-for-frame compare the two iClone versions against the same Blender
  `animation.blend`. Use same camera orientation and disable/inspect IK
  overriding FK if any.

Frame index | Meaning
:--|:--
0,24,48,72,96,120,144,168 | both bones at rest
12 | Parent X +25°, Child zero
36 | Parent Y +25°, Child zero
60 | Parent Z +25°, Child zero
84 | Parent zero, Child X +25°
108 | Parent zero, Child Y +25°
132 | Parent zero, Child Z +25°
156 | Parent X +25°, Child Y +25°

Zero-reference frames isolate unintended accumulated transforms. Comparison
MUST use exactly matched frame indexes; the original iClone Dragon Roar
124-frame clip is not this 169-frame experiment.

## 4. Interpret results

- If A agrees with Blender for every probe but B does not, direct
  Euler is likely correct for this pair; previous discrepancies may stem
  from an unrelated clip/layer, import orientation or other bones.
- If B agrees with Blender across all probes, source-to-iClone rest basis
  conversion is implicated. Repeat for **neck1/neck2** and **tail2/tail3**,
  not just the wing.
- If Parent-only matches but Child-only fails, focus on the Child's relative
  rest basis, Euler order, sign and IK constraints.
- If both A and B fail, investigate FK Layer evaluation, default bind
  rotation offsets, parent-transform composition, inherited transforms,
  scene orientation and IK/FK state; never mark mapping calibrated.
- If both pass visually, identify additional frames and quantitative
  joint-position comparison before claiming transform equivalence.

Examples for other chains:

```powershell
py -3.14 -m iclone_adapter.parent_child_probe F:\anymove_test\skeleton.json F:\anymove_test\neck_probe --parent neck1 --child neck2

py -3.14 -m iclone_adapter.parent_child_probe F:\anymove_test\skeleton.json F:\anymove_test\tail_probe --parent tail2 --child tail3
```

## Current limitations

- Writer still requires the exact same iClone Skeleton Hash; if a
  reimport changes object names or hierarchy, export a NEW *unposed*
  skeleton snapshot and rerun the probe generator for that scene.
- This is a nondestructive JSON generator, but importing either test
  clip modifies iClone. Use separate clean test projects.
- The iClone P1 Writer may reuse a default empty clip; otherwise append
  new motion, so the clean-scene instruction is critical.
- Numerical FK key readback does not certify visual pose.
- Motion root is ignored (zero in this test by construction).
