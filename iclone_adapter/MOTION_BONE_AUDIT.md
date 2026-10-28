# Dragon iClone 8 FK Layer audit: 2026-10-28 findings

## Correct the iClone 7 / iClone 8 API distinction

The original notes in this file erroneously treated iClone 7's Motion Bones
as a required second skeleton on iClone 8. That was incorrect. The official
[iClone 8 Python API](https://wiki.reallusion.com/IC8_Python_API) explicitly
states that the iClone 7 Motion Bone skeleton was removed in iClone 8, and
`GetMotionBones()` was deprecated. The official
[iClone 8 FK/IK example](https://wiki.reallusion.com/IC_8_Python_API:End_Effector_Animation)
uses `GetControl("Layer", any_bone)` and a Skin Bone to author FK rotations.

An older iClone 7 document and feedback report about distinct Skin/Motion IDs
do not override the iClone 8 architecture.

## Observed actual Dragon data

The user supplied `dragon_bone_channel_audit.json` from the real iClone 8
Dragon, with 124-frame `dragon_roar_fk_all.json` on Clip Index 0.

- `GetSkinBones()`: 57 nodes; 41 of 41 requested non-root Bone names.
- `GetAllAnimationBone()`: 57 nodes, 57 unique names; same 41 mappings.
- `GetMotionBones()`: absent, as expected for iClone 8.
- All 41 target Bone IDs identical between `GetSkinBones()` and
  `GetAllAnimationBone()`.
- All 41 target Bones have exactly 124 FK rotation keys in **each**
  of X, Y and Z: 41 * 3 * 124 = **15,252** keys.
- Frame 0, 62, 123 controller values compare with the expected FK preview:
  **41/41 sampled matches**, worst absolute numerical difference
  **4.421234134355245e-8 radians**.
- This verifies that the intended Layer controller data was written and
  retained. It is NOT evidence of the correctness of the Euler axis
  conversion or mesh deformation, nor of missing Root Motion.

The prior `dragon_fk_validation.json` from v0.1.9 sampled nine timeline
frames and found identical `RINode.LocalTransform/WorldTransform`
orientation values across frames, despite independently reading varying FK
Layer controller values, and despite visible motion in the video.

**Interpretation:** that validator did not obtain an evaluated dynamic Bone
pose from those RINode calls in this scene. It was incorrect to infer
41 wrong Euler axes or that only five bones were animated. v0.2.0 already
blocks such false FK axis conclusions when its Bone sampling is static.

## Correct next verification

1. Keep the existing Writer; do **not** switch it to `GetMotionBones`.
2. Compare a Blender ground-truth render of the **same original Solver
   `motion.json`** against actual iClone viewport/render images at
   identical frame numbers; first 0, 31, 62, 93, 123.
3. Remember Blender original includes `pelvis` `root_quat` and `trans`,
   while current iClone FK preview explicitly **omits** Root motion.
   Normalize/alignment by pelvis before judging joint motion.
4. If the actual target pose differs, test per-bone FK basis/Euler
   interpretation against the rendered reference, not static
   `RINode.LocalTransform/WorldTransform` data.
5. Implement Root quaternion/translation separately with verified SDK
   controls and scale conversion; do not silently pretend it is present.
6. For iClone runtime errors while asynchronously changing scene time,
   do not assume timeline sampling synchronously evaluates RINode;
   verify the API semantics before re-enabling pose metrics.

## Reading current code

Use **Audit iClone 8 FK Bone IDs / Keys (read-only)** in the plugin. The
report includes `channel_integrity` with a deliberately narrow result:
`sampled_fk_controls_match`. No scene or key modifications are made.

## Offline Blender reference screenshots (same source file)

Run from PowerShell with your actual Blender executable path:

```powershell
cd F:\anymove
$blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
& $blender -b --python-exit-code 1 -P blender/apply_motion.py -- `
    assets/dragon/rigged/rigged.blend `
    assets/dragon/rigged/rig_info.json `
    runs/dragon/clips/roar/motion.json `
    runs/dragon/clips/roar/blender_reference `
    --views persp --engine workbench --frames 0,31,62,93,123
```

Outputs 5 PNG files under
`runs/dragon/clips/roar/blender_reference/persp/` in the source's
camera. Match comparable iClone camera angles if testing visually.

Note: This is screenshot-based validation, not ground-truth numerical
pixel correspondence without calibrated cameras and root alignment.
