# P2 experimental Dragon Roar FK preview (NOT calibrated retargeting)

The user already exported Dragon from Blender as a skinned FBX and imported
that very same rig into iClone. The uploaded iClone Dragon `skeleton.json`
contained 57 nodes: the original 42 named animation bones, 12 `_end`
bones, `rigged`, `rig`, and `body`. All original parent links and
rest-space world matrices aligned under a global 100x scene scale.
This supports a **same-rig identity-axis experiment**, not arbitrary
retargeting or an automatic FK-axis calibration assertion.

The AnyMove solver's original `motion.json` is NOT an iClone FK file.
It contains Blender Euler XYZ for 42 bones and `root_quat` / `trans`
independently. Existing `anymove_bridge.convert` correctly refuses a
`calibrated:false` mapping. **Do not mark mapping bones calibrated
solely to force it through.**

The following separate preview command explicitly records that its
identity-axis assumption is unverified, without modifying the mapping:

```powershell
cd F:\anymove
git pull --ff-only origin iClone

New-Item -ItemType Directory -Force F:\anymove_test | Out-Null

py -3.14 -m iclone_adapter.dragon_preview `
    runs\dragon\clips\roar\motion.json `
    F:\anymove_test\skeleton.json `
    F:\anymove_test\dragon_roar_fk_preview.json `
    --bones neck1,head,tail1,tail2
```

Uses only the Python standard library; the Solver has already finished.
If `skeleton.json` is in another directory, use its real path.

In iClone:
1. Work on a **disposable copy** of the Dragon project.
2. Keep the imported Dragon unposed and select it.
3. Match Project FPS to the printed `motion.json` FPS and move to Frame 0.
4. Load `launch_iclone.py` (v0.1.8) and select **Write FK motion to
   selected avatar**; choose `dragon_roar_fk_preview.json` — NOT the
   source `motion.json`.
5. Read the warning that FK axes are NOT calibrated, confirm only for
   this test project, then inspect head/neck/tail through the full timeline.
   Status keys verified + changed pose do not guarantee plausible movement.
6. If axes and deformation look correct, run again in a fresh test scene
   with `--bones all` to cover 41 non-root bones, but do not treat this
   as proven general retargeting.

**P2 limitations**:
- Does NOT include pelvis root Quaternion or Translation;
  the Dragon may not move its body as in AnyMove's original clip.
- Uses Blender Euler XYZ on iClone FK Layer as a temporary basis hypothesis.
  It is explicitly **not** an approved automatic calibration; some bones
  may rotate differently, especially wing axes.
- The iClone P1 writer can re-use a default Motion Clip with zero FK keys
  or append a fresh Clip beyond existing motion; check the reported
  `clip_strategy` and playback range.
- Blender's rendered animation from `blender/apply_motion.py` remains
  the proper independent ground-truth reference for comparing pose.
- The existing approved conversion path through `anymove_bridge.convert`
  still requires `calibrated:true` for **every** mapped bone.

## v0.1.9: measurable iClone FK pose comparison

After writing a Dragon preview to a disposable iClone scene, use the new
**Validate Dragon FK animation vs preview (read-only)** button. Select the
original **unposed** Dragon `skeleton.json`, then the same FK preview JSON
that was actually written, and choose the Motion Clip index (last by default).
The tool samples several scene frames including high-rotation frames, restores
the original playhead, and reports each Bone's evaluated LocalTransform error
against the Blender Euler rotation in the preview. The comparison removes
rest local rotation and global FBX scale before computing angular error.

A `consistent_in_sampled_fk_axes` result means the FK Layer evaluation is
consistent with Blender for **sampled non-root Bones only** (max 5-degree
error for each meaningfully moving Bone). It is not proof of correct mesh
Skinning, interpolated motion, or missing Root Motion.

A `fk_axis_or_evaluation_mismatch` result identifies the worst Bones and
measured errors. Upload `dragon_fk_validation.json` for diagnosis. Use an
unposed skeleton snapshot and the actual matching Motion Clip; otherwise
errors may reflect a wrong baseline or clip rather than wrong FK axes.

## v0.2.1 correction: iClone 8 does not use separate Motion Bones

The real Dragon Bone audit confirms 41/41 XYZ tracks have 124 keys per
axis and samples at 0/62/123 match the source to better than 5e-8 rad.
`GetSkinBones()` and `GetAllAnimationBone()` share the same target IDs.
Do not apply legacy iClone 7 `GetMotionBones()` retarget assumptions.
See [MOTION_BONE_AUDIT.md](MOTION_BONE_AUDIT.md) for the detailed
source-based findings and Blender visual comparison instructions.


## 2026-10-28: comparing the same frame in Blender and iClone

A side-by-side screenshot showed iClone **Frame 0** but Blender **Frame -25**
(top-left Blender viewport overlay). Moreover opening the original
`rigged.blend` in Blender does NOT load the solver's `motion.json`.
The old `blender/apply_motion.py` rendered motion images or wrote an animated
GLB but did not save the keyed Blender scene. This made the comparison look
like a neck/tail retargeting error even if the FK axes were fine.

**New optional Blender flags**:
- `--save-blend`: writes `out_dir/animation.blend`, with all keyframes
  applied and the scene at Frame 0.
- `--no-root-motion`: disables only the solver's pelvis root quaternion and
  world translation. The 41 non-root Euler XYZ tracks are unchanged. This
  matches the current iClone FK-only preview's missing root motion.

From **PowerShell**, create the reference:

```powershell
cd F:\anymove
git pull --ff-only origin iClone

$blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
& $blender -b --python-exit-code 1 -P blender/apply_motion.py -- `
  assets/dragon/rigged/rigged.blend `
  assets/dragon/rigged/rig_info.json `
  runs/dragon/clips/roar/motion.json `
  runs/dragon/clips/roar/fk_reference `
  --no-root-motion --save-blend
```

Open **`F:\anymove\runs\dragon\clips\roar\fk_reference\animation.blend`**
in Blender (NOT the original `rigged.blend`). Blender Timeline should be
Frame 0 with animation frames 0..123; compare at 0, 31, 62, 93, 123
against the same iClone Clip at the same frames and similar side-view
camera angles. Confirm the project/clip start first (previous iClone Dragon
Clip index 0 was at time zero). If iClone project FPS differs from the
source FPS, use the actual time equivalents.

**Frame 0 is NOT necessarily rest pose**. The original Dragon motion has
non-zero Euler values at time zero for the neck and tail, and several
successive curved tail sections accumulate visible displacement. Do not
compare iClone Frame 0 to Blender Rest Pose or Frame -25.

To compare the **full** Blender motion with root separately, run the same
command with another output directory and omit `--no-root-motion`. Do not
interpret missing iClone pelvis motion as a bone retargeting error.

This visual comparison will help diagnose FK axis conventions but is not
an automated bone-pose numerical proof; the current iClone RINode
LocalTransform/WorldTransform reads did not reflect animated FK channels
during validation.
