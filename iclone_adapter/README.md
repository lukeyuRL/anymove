# AnyMove for iClone — P0/P1 experimental slice

Target branch: `iClone`. The original Blender/AnyMove pipeline stays untouched.

## Status (do not treat this as end-to-end capture)

- **P0 built:** read one selected rigged Avatar/Prop's Skin Bone hierarchy, native current-pose local/world transforms, supported SDK flags and export JSON. Skin Bone identities are hierarchical paths; non-unique `RINode.GetID()` values are reported as diagnostics rather than blocking export.
- **P0 NOT verified:** true Skin Bind/Rest Pose, native-to-metre scale, per-asset animation control semantics and iClone 8/9 real runtime.
- **P1 SDK audit:** enforce project FPS via `RFps.ToDouble`, extend new clips with `RIClip.SetLength`, and handle all `RStatus.IsError()` outcomes. Matrix layout is runtime-probed rather than assumed.
- **P1 built:** strictly typed in-scene FK **channel** motion, optional synthetic 48-frame sine test, safety preflight, new motion clip and attempted rollback.
- **P1 NOT verified:** end-to-end real SDK write/readback, clip range, FK/IK interaction, arbitrary Skeleton retargeting, AnyMove Blender Euler conversion, root motion and Generic Solver.
- DINOv2 and torch **must run in a separate Python 3.11 or 3.14 process**, not the iClone Python 3.8 host.

## Run in iClone 8 Python console

Clone the repository locally and select **exactly one** Avatar or rigged Prop in iClone.
In the iClone Python Console (use your actual clone path):

```python
import sys
sys.path.insert(0, r"F:\anymove")
from iclone_adapter.run_in_iclone import show
show()
```

1. **Inspect selected skeleton**: check `GetSkinBones`, `GetClipCount`, `AddClip`, `DeleteClip`, and other API capabilities, without modifying the scene.
2. **Export current-pose skeleton snapshot**: write `skeleton.json`. The reference is **current_snapshot**, not verified bind pose. RLPy matrix translation layout is first probed using a synthetic nonzero RMatrix4 translation. The reader uses `GetRow()[index]` rather than `M(row,col)`, since the latter can return an opaque `SwigPyObject`; the resulting canonical matrices retain **unverified native units**. No scene changes.
3. **Create 48-frame diagnostic FK motion**: choose a skin bone; save `test_fk.json` with an X-channel sine curve, 24 fps. This is an *iClone channel-space* diagnostic, NOT AnyMove capture.
4. Open an expendable iClone scene with the **same selected avatar**, **zero existing motion clips**, **project 24 FPS**, **timeline at frame 0**. Use **Write FK motion**. The dialog explicitly requires acknowledgment of the scene mutation.
5. Inspect the result manually. A successful tool return is marked `keys_written_unverified`, not proven correct. Verify that the bone rotates around the intended X channel, moves over 48 frames, and the last frame returns to zero.

**Never** pass the upstream `mocap.solve` `motion.json` directly into the P1 writer. It contains Blender-specific local Euler XYZ + root data. FK Bone Rest-axis mapping and retargeting must be built and calibrated first. The included `examples/dragon_roar/det.npz` is useful for subsequent P2 testing, but is not a stand-in for the bone-channel conversion.

## Invariants

- P0 never modifies the scene. No claim that `GetBoneTPosePosition` is a complete bind matrix: the API documents a position only.
- The P1 writer requires a hierarchy signature from the **current** live scene and rejects any foreign/synthetic skeleton ID.
- Only `anymove.iclone.fk_channels/1` files can be written; no assumptions about arbitrary quaternion axes.
- P1 refuses any avatar that already has one or more Motion Clips. This is an intentional MVP safety constraint, not a product feature.
- A failed write tries to delete the newly created clip. If rollback fails the error must be treated as a potentially modified scene.
- Existing original AnyMove files, assets and its standard Python environment remain unchanged.

## Offline unit tests

```bash
python -m unittest discover -s tests -p "test_iclone_*.py" -v
```

Python 3.8 and 3.11 are targeted. Tests use mock RLPy to cover hierarchy, row-vector matrix conversion, FK channel writes and contract refusal; they **do not substitute for iClone runtime tests**.

## SDK references

- https://wiki.reallusion.com/IC8_Python_API:RLPy_RISkeletonComponent
- https://wiki.reallusion.com/IC_8_Python_API:End_Effector_Animation
- https://wiki.reallusion.com/IC_8_Python_API:Dealing_With_Custom_FPS
- https://wiki.reallusion.com/IC_Python_API:RLPy_RTransform

The attached **OpenPlugin_IC8** SDK reference bundle was reviewed on 2026-10-28. Its legacy binding documents `RMatrix4.GetRow(row) -> RVector4` and `RVector4.__getitem__(index) -> float`, plus `MakeIdentity`, `SetTranslate`, `RFps.ToDouble`, `RIClip.SetLength(RTime)` and `RStatus.IsError()`. These declarations are static evidence, not an iClone runtime pass. The source bundle is not copied into this repository.

## Next engineering blockers

1. Verify true bind/rest matrices and per-bone axis conversion with real iClone avatars (P0 gate).
2. Verify FK channels and clip duration, readback and FK/IK override with real iClone runtime (P1 gate).
3. Implement a Generic Pose quaternion contract and calibrated Blender/AnyMove-to-iClone mapping, before consuming Dragon det.npz/solver motions.
4. Implement direct viewport dual-camera reference/dataset rendering, DINOv2 training/inference service and retargeting.

## Convert the supplied Dragon detection through AnyMove (external Python 3.11)

The original repo includes `examples/dragon_roar/det.npz`. Without DINOv2
training, run in the **external** AnyMove environment:

```bash
MOCAP_CREATURE=assets/dragon python -m mocap.solve \
  examples/dragon_roar/det.npz assets/dragon/rigged/rig_info.json \
  runs/dragon/clips/roar/motion.json
```

Do not load this Blender motion into the P1 writer directly. Instead, export
a snapshot from the SAME iClone scene and generate a mapping template:

```bash
python -m iclone_adapter.anymove_bridge map \
  runs/dragon/clips/roar/motion.json skeleton.json map.json
```

Inspect `map.json`. For each desired source Bone, set `target_id` to the
matching ID in `skeleton.json`, provide the calibrated
`axis_quaternion_wxyz`, and set `calibrated=true` **only after confirming
the iClone FK channel basis**. The template has no approved mappings by design.
Convert with:

```bash
python -m iclone_adapter.anymove_bridge convert \
  runs/dragon/clips/roar/motion.json skeleton.json iclone_fk.json \
  --map-json map.json
```

The output intentionally excludes AnyMove root translation and rotation and
does not compensate for different bone lengths. This is a per-bone rotation
test, **not a finished retarget**. It requires a rig with known correspondence
and verified per-bone axes. To test arbitrary iClone skeletons, first run the
synthetic FK channel diagnostic and confirm axes in the actual iClone UI.

## Fix for iClone matrix export `SwigPyObject` failure

Previously the export operation threw:

```
ContractError: RLPy matrix layout probe failed:
float() argument must be a string or a number, not 'SwigPyObject'
```

The Legacy RLPy `RMatrix4.M(row,col)` may select the SWIG `float&`
overload. This has been replaced in both probe and bone reading paths with
`RMatrix4.GetRow(row)[col]`, which is documented as returning a Python
`float`. Re-run with **the latest iClone branch and reload the Python
plugin/script**. Export a new `skeleton.json`; do not reuse a partial
earlier file.

## 2026-10-28: `duplicate scene bone IDs` fix

The screenshot from a 103-bone `Casual_F_0058` character showed `GetSkinBones()`
working, but the exporter rejected non-unique `RINode.GetID()` values.
This is not evidence of duplicate bones: an SDK scene node ID is not an
appropriate unique key for every skin joint. The exporter now uses
`skin:/<parent>/<bone>` hierarchy-derived keys and preserves native IDs
only as diagnostics. For names duplicated at different places in the
hierarchy, full paths distinguish them. Identically named siblings
receive a numbered suffix determined by the SDK's skin-bone order.

The writer uses the same registry and no longer looks bones up by
`GetID()`. It rejects unresolved ambiguous parent identities rather than
inventing connections. Export now includes `identity_diagnostics`:
`native_id_duplicate_nodes`, `duplicate_name_groups` and `identity_strategy`.

**Note:** Your screenshot also reports `existing_clip_count: 1`. The
experimental P1 writer still intentionally requires zero existing
motion clips, so use a separate empty-motion test scene for P1.

## If iClone export still fails (v0.1.4)

The Diagnostics window now shows its loaded build version and module path,
allowing you to tell an older cached script from the updated one. Export
failures write a read-only report at `<requested filename>.debug.json` (or
the system temp folder if that location cannot be written), with:

- The complete Python traceback
- The module file paths loaded in the iClone process
- One record per Skin Bone: index, name, raw GetID, parent name/GetID
- Per-bone SDK errors, captured without aborting the entire report

You can also press **Save raw Skin Bone diagnostics (no export)** to gather
the same bone facts before retrying export. A CC character with duplicate
native GetID values is expected to be supported, not treated as invalid.

This revision also fixes a potential false positive in the SWIG
`GetParent()` lineage cycle check by retaining proxy references while
walking the chain. It remains essential to preserve the actual error
message for any new failure; CI mocks cannot perfectly model iClone.

## P0 empirical diagnosis: duplicate GetSkinBones records (2026-10-28)

A real `Casual_F_0058` debug report showed **103 raw rows but 102 unique
native IDs**. Indices `0` and `4` are *both* `RL_BoneRoot`, with identical
native ID `6351195944448` and parent `Casual_F_0058`. This was the sole
duplicate-ID and duplicate-name group. The old hierarchy resolver therefore
raised `ambiguous duplicate parent bone 'RL_BoneRoot'`.

Before reconstructing parent links, the registry now coalesces duplicate
records **only if** native ID, name, immediate parent ID/name and local/world
transform fingerprints match (or the Python wrapper is literally the same).
Native ID collisions involving different names, parents or transforms are
still treated as distinct entries. The export includes both
`raw_skin_bone_count` and `canonical_bone_count`, plus audit metadata
`coalesced_duplicates` with original row indices.

**Expected result for this specific actor:** 102 canonical Bone records from
103 raw `GetSkinBones()` entries. This is a deduplicated record count,
not proof that the character's true Skin Bind Pose has been recovered.

## 2026-10-28 P1: keys written but no animation visible

See [P1_VALIDATION.md](P1_VALIDATION.md). Before modifying a scene,
use the new **Inspect existing motion clips / FK key counts** button:
`GetClipCount` alone does not prove a clip has FK keys. The write path now
performs `GetKeyCount` and `GetValue` readback, sets the playhead to the
peak diagnostic pose and calls `Update`. The synthetic test defaults to
`CC_Base_L_Forearm` where present and uses 30 degrees to make motion
easy to see. IK can still override FK, and automatic FK-to-IK baking remains
intentionally out of scope for P1.

## v0.1.7: default Motion Clip is now supported

The previous gate `GetClipCount()==0` was incorrect for iClone avatars.
The official Reallusion guide writes Motion Layer keys into
`GetClip(0)`. The updated writer chooses:

- **reuse_default**: if one Clip starts at Frame 0 and mapped FK
  channels have no keys, write into that existing Clip. Expect
  `clip_count_before: 1` and `clip_count_after: 1`.
- **append_new**: if a Clip has pre-existing FK keys (or there are
  multiple Clips), append at least one frame beyond the last effective
  Clip end, without overwriting the existing clips. P1 moves the
  playhead to the appended animation's peak frame.
- **create_new**: only when GetClipCount()==0, create at Frame 0.

Use a disposable iClone scene. The write is to FK Layer data, not a
general AnyMove/Blender retarget. The writer attempts to undo its own
newly inserted keys if reusing a clip fails. The returned JSON now shows
`clip_strategy`, `clip_count_before`, `clip_count_after`,
`clip_start_ticks`, and whether the Timeline end was extended.
An unchanged WorldTransform despite FK keys is a separate FK/IK
evaluation issue, not a Clip count error.

Reference: https://wiki.reallusion.com/IC_Python_API:Animating_A_Character

## P2 same-rig Dragon Roar preview (v0.1.8)

See [DRAGON_PREVIEW.md](DRAGON_PREVIEW.md) for a **test-only** command
that converts the Solver's original `motion.json` into an explicit
uncalibrated iClone FK preview after validating the original 42-bone rest
alignment. No PyTorch is needed for this conversion. The generated JSON
retains a warning; root motion and automatic FK-axis calibration remain
unsupported. Never pass raw AnyMove `motion.json` to the P1 Motion Writer.

## v0.3.0: guarded experimental Dragon Writer

Every newly generated Dragon FK preview includes the original **unposed**
57-node world-matrix baseline. The Writer checks that the current pose
matches this saved baseline before writing. For unverified Blender previews
it also disallows appending over old clips; one existing default clip is
permitted only if it has no length and no recognized FK/Position/Scale keys
on ANY bone. Actor Transform keys block writing too. This targets a real
Dragon failure caused by reusing an animated project with static Head/Tail
offsets. Checks are not a comprehensive IK/constraint certification;
continue using **fresh test projects**. Regenerate the experimental FK JSON
with this version; old JSON has no baseline and will be rejected.

## v0.3.1 experimental Pelvis Root Motion writer (NOT calibrated)

Generate either `--root-mode rotation` or `--root-mode full` with
`--basis rest` from the original AnyMove Dragon `motion.json`.

- Rotation: conjugate the original root quaternion by the `pelvis`
  Blender Armature-space Rest quaternion, write its XYZ Euler to the
  `pelvis` Bone's RIClip Layer Rotation XYZ.
- Translation (full mode only): `trans` in original armature axes times
  explicit `--root-position-scale`, write Pelvis Layer Position XYZ.
  The scale factor and Pelvis Position convention are **unknown until
  calibrated in your real iClone 8 Dragon**. We refuse to guess a scale.
- Position channels must exist in the live clip before any new keys are
  written. Missing controls cause the new experimental Clip to be rolled
  back; no partial root write is intentionally left behind.
- Root Rotation/Position samples are read back. This proves controller
  storage only, NOT final pose/translation.
- The clean-scene guard applies to both modes.

Use the read-only in-app button **Inspect Pelvis FK Root Position controls**
to check if your Dragon exposes `Position/PositionXYZ` before experimenting.
Root rotation-only allows Blender `--no-root-translation --save-blend`
for a matched reference; the original `--no-root-motion` reference
continues to be correct when no iClone root motion is written.

## v0.3.2 iClone actor Transform default key false-positive correction

A freshly imported Dragon was blocked in v0.3.1 by
`guard: actor already has Transform keys` because the guard incorrectly
assumed *any* Transform key means animation. This is not reliable in iClone:
the imported actor can have an initial Transform key without temporal motion.

The Writer now uses the **documented iClone RControl APIs**:
`GetKeyCount`, `MaxControlTime`, and (if there are multiple keys)
`RTransformControl.GetTransformKeyAt`. One frame-zero key is allowed,
preserved and reported as `single_static_frame0_transform_key`.
Multiple keys are allowed only if all audited XYZ position, quaternion
orientation (accounting for q/-q), and scale remain effectively identical;
unknown/unreadable multiple keys are rejected. Actor Transform DataBlock
Position/Rotation/Scale channels are checked for additional keys beyond
frame zero when exposed. No preexisting Key is removed or reset.

The 57-Bone unposed reference check and all-bone FK/Position/Scale Clip
preflight remain enabled; never reuse an old animated Dragon project.
This is a narrow compatibility fix, not a relaxation of pose safety.
