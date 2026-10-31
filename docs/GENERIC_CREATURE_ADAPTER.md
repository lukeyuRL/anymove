# Generic Creature Adapter (experimental same-rig iClone FK import)

## Goal

A **single** AnyMove pipeline for Dragon, Dino, and newly imported creatures:

`creature.json + rig_info.json + 2-view Reader + MP4 -> det.npz ->
motion.json -> source/target rest audit -> generic iClone FK preview ->
in-scene readback -> iProject`.

The Reader is trained per creature; the Detector, Solver, FK format, and
safety checks should be shared. **Joint count is not Bone count**. For
example, the actual Dino dataset has 92 keypoints, but its animation Bone
count must be read from `rig_info.json`, not inferred.

The existing 42-Bone Dragon, 100x, 24-FPS direct-import reference remains
untouched. Its previously validated result is not evidence that another
skeleton's iClone FK axes or root translation are calibrated.

## Current implementation (P0 / read-only package)

`iclone_adapter.generic_creature` prepares a **generic same-rig**
FK-preview package. It does not need to know the creature type, number of
bones, root name, FPS or imported scale in advance.

Required inputs:

- The source `motion.json` from `mocap.solve`.
- The exact corresponding source `rig_info.json` used to produce that
  motion. A new/mismatched Rig must not be substituted.
- A `skeleton.json` snapshot exported in iClone from a newly imported,
  UNPOSED copy of the **same original** skinned rig. The parent hierarchy
  and world Rest Bone matrices must match under a single positive uniform
  scale. Additional iClone helper nodes may exist.
- A fresh iClone project. The snapshot is evaluated current pose, **not**
  certified Skin Bind Pose.

### Windows external Python commands

Dino example (replace the skeleton path with the one you exported):

```powershell
cd F:\anymove
$env:MOCAP_CREATURE = "assets/dino"
.\.venv\Scripts\python.exe -m iclone_adapter.generic_creature `
    --motion runs/dino/clips/test_move/motion.json `
    --rig-info assets/dino/rigged/rig_info.json `
    --skeleton F:/anymove_test/dino_skeleton.json `
    --output-dir F:/anymove_test/dino_fk_preview
```

Dragon example with the same generic converter:

```powershell
.\.venv\Scripts\python.exe -m iclone_adapter.generic_creature `
    --motion runs/dragon/clips/roar/motion.json `
    --rig-info assets/dragon/rigged/rig_info.json `
    --skeleton F:/anymove_test/dragon_skeleton.json `
    --output-dir F:/anymove_test/dragon_fk_preview
```

Output:

- `creature_fk.json`: explicit iClone FK Layer XYZ radians, matching target
  Skin Bone IDs, all non-root bones (default), expected unposed world matrices.
- `creature_import_manifest.json`: independent hash manifest binding FK JSON
  to the source Motion, Rig, target Skeleton hash and unposed snapshot.
- `creature_import_report.json`: hierarchy correspondence, uniform scale,
  maximum rest-matrix and rest-position residuals, extra helper nodes.

The iClone UI's existing **Write FK motion to selected avatar** operation
accepts `creature_fk.json`. The adjacent manifest is checked before
controller writes. It uses the same scene safety preflight, including the
unposed-baseline comparison and empty Motion-layer verification. In the
initial stage, this is an explicit experimental operation in a disposable
project; the automatic one-click Generic Creature Import is not yet verified.

### Optional experimental root

The default is `--root-mode none`; the full animation will omit root
rotation/translation and is **not a final motion**. If a clip needs root
rotation/translation, explicitly opt in:

```powershell
.\.venv\Scripts\python.exe -m iclone_adapter.generic_creature `
    --motion runs/dino/clips/test_move/motion.json `
    --rig-info assets/dino/rigged/rig_info.json `
    --skeleton F:/anymove_test/dino_skeleton.json `
    --output-dir F:/anymove_test/dino_root_preview `
    --root-mode full --root-position-scale 1.0 `
    --root-position-space root_local
```

**WARNING:** `1.0` is an explicit *test hypothesis*, NOT an automatically
valid factor for Dino or a new creature. The FK/Root position controller
unit and direction MUST be established by live same-frame Blender/iClone
measurements. You may instead use `--root-position-space armature` if
calibration proves that is correct. On the known Dragon reference, the
existing `pelvis_layer` convention remains backward compatible.

No run should claim production-quality import on the strength of static
hierarchy hashes, synthetic pose tests or controller readback alone.

## Mandatory live acceptance gates for every new Rig family

1. **Reader**: training rig hash, joint-name order, input resolution,
   detector overlay, both cameras correctly match the source creature.
2. **Solver**: valid frame_count, consistent Bone order, root quaternion,
   root translation, reprojection and 3D temporal continuity; inspect in
   Blender on the identical skin.
3. **Same-rig rest match**: verify unique name/path/parent correspondence,
   no missing/ambiguous source bones, positive uniform imported scale,
   small orientation/position residuals, clean unposed iClone snapshot.
4. **FK basis**: compare local Bone transforms on a small set of numbered
   frames for head, limbs, torso and tail; controller values alone are
   insufficient. Confirm active IK does not override FK.
5. **Root**: verify 3D translation amplitude *and direction* with matching
   Blender frames; reject opposite axis signs, scene-unit mismatches and
   actor-transform offsets.
6. **Full clip**: verify per-frame controller key counts and readback,
   visible skin deformation, original 24/30 FPS matching, clean scene,
   import receipt and saved iProject.

All of these gates apply to both Dragon and Dino; new rigs should not
require a new Python converter. **Different-rig retargeting** (e.g. applying
Dino motion to a separately rigged dinosaur) is a separate capability,
requiring an explicit anatomical Bone Map, FK/IK and bone-length policy.

## Known limitations / next stage

- The generic converter provides a guarded, uncalibrated same-rig preview,
  **not yet proven one-click Dino iClone import**. A real Dino skeleton.json,
  animation and iClone Controller/WorldMatrix readback are still required.
- The original Dragon direct-import profile remains as a compatibility path.
  Productionizing the generic one-click UI should follow a successful
  same-rig Dino test, then reuse the generic package/scene guards and emit
  receipts without any Dragon-specific folder names or FPS assumptions.
- Current `mocap.solve` still has named-paw/ground prior assumptions;
  arbitrary creature motions need validated semantic group defaults and
  appropriate solver gates. Its Dragon-specific visibility/temporal
  experiments remain Dragon-only until generalized with independent tests.

## Offline tests

```bash
python -m unittest discover -s tests -p 'test_iclone_generic_creature.py' -v
python -m unittest discover -s tests -p 'test_iclone_*.py' -v
```

These tests include a synthetic non-pelvis, non-42-bone Dinosaur;
the existing 42-Bone Dragon; mismatched pose/hierarchy;
rotation-only/full/root-only; manifest tamper detection; and no silent
change to the Dragon direct-import profile. No test claims live iClone
runtime success.
