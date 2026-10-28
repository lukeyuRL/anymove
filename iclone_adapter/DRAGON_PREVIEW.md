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
