# P1: test when your iClone avatar already has a Motion Clip

**Do not remove the default clip just to satisfy a count restriction.**
Reallusion's [Animating A Character](https://wiki.reallusion.com/IC_Python_API:Animating_A_Character)
example uses `GetClip(0)` and writes FK Motion Layer rotation keys.

In this version:

1. Download/pull the `iClone` branch; reload the test UI (v0.1.7).
2. In a disposable iClone project, load the CC character and select it.
3. Set project FPS to 24, move Timeline to Frame 0.
4. **Inspect existing motion clips / FK key counts**. One clip is normal.
5. **Export current-pose skeleton snapshot** as `skeleton.json`.
6. **Create visible 48-frame FK motion (30 deg)** from the snapshot, choose
   `CC_Base_L_Forearm` if available, save `test_fk.json`.
7. **Write FK motion to selected avatar** and confirm the test-project
   warning. P1 should now reuse the default clip if its target FK channels
   are empty, otherwise add a new nonoverlapping clip after existing motion.
8. The tool moves the playhead to the largest rotation sample. Inspect
   `clip_strategy` and `clip_start_ticks`, then play the new motion range.

`fk_keys_verified_pose_changed` confirms a changing Bone WorldTransform;
`fk_keys_verified_pose_unchanged` means FK keys were retained but the
bone did not respond, possibly due to IK override. Neither state proves
the skin mesh deformed correctly; check the iClone viewport.

No existing clip is deleted by P1. In the **reuse_default** case the
tool writes to the existing clip's FK channel and on an exception
attempts to remove newly written keys. Test on an expendable scene.
