# 2026-10-28: real Dragon video vs static Skin Bone API measurements

Uploaded `dragon_fk_validation.json` reported 41 tracks, nine sampled frames
(0,9,29,31,38,62,71,93,123), zero axis-valid Bones. On close examination,
the purported `actual_delta_deg` value was exactly identical on ALL NINE
frames for each of 41 Bones. The five nonzero offsets (head, neck1, tail1,
tail2, wing_arm_R) were static offsets relative to the saved unposed
reference, NOT evidence of animated keys. The supplied 24-FPS, 1.625-second
`dragon.mp4` visibly animates the wings/forelegs/neck. It is therefore
incorrect to label that report as proof of a wrong Euler axis.

This is a known distinction in iClone's architecture: [official
documentation](https://wiki.reallusion.com/IC_Python_API:Animating_A_Character)
states that **Motion Bones** receive FK animation and are retargeted to
**Skin Bones**. A [2020 Reallusion feedback issue](https://www.reallusion.com/FeedBackTracker/Issue/Bones-returned-from-GetSkinBones-have-a-different-ID-than-the-same-ones-you-get-from-GetSelectedBones)
specifically warns that GetSkinBones nodes can be invalid for a Motion
Clip. The supplied Legacy iClone8 RLPy.py includes
`RISkeletonComponent.GetAllAnimationBone()`, but not
`GetMotionBones()`. These APIs must be PROBED rather than conflated.

**v0.2.0+**:
- P2 validation checks temporal changes in the actual parent-relative
  Bone rotation. If all sampled Skin Bone transforms stay identical, it
  returns `bone_transform_static_axis_unverifiable` and does NOT return
  the misleading `fk_axis_or_evaluation_mismatch`.
- A new **Audit Skin vs Motion Bones / FK keys (read-only)** button
  examines `GetSkinBones`, `GetMotionBones` if exposed, and
  `GetAllAnimationBone` if exposed, compares name/ID for mapped Bones,
  checks per-clip Layer FK key counts and samples control values.
- It does NOT change the existing Writer to guess which Bone set is
  correct. On a real nonstandard Creature this must be verified from
  the actual adapter SDK before altering playback.

To diagnose:
1. Update repo, reload `launch_iclone.py` in iClone.
2. Open the scene with the ALL-41-bone Dragon Roar, select Dragon.
3. Click **Audit Skin vs Motion Bones / FK keys (read-only)**.
4. Choose the actual `dragon_roar_fk_all.json`, clip index 0 if one.
5. Save and upload `dragon_bone_channel_audit.json`.
6. Examine which SDK Bone collection has actual keyed FK channels
   and which IDs are distinct. Animation source/axis corrections should
   follow evidence, not precede it.
