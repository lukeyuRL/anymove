# P1 Motion Writer – real iClone verification

The previous return value `keys_written_unverified` only meant the SDK
`SetValue()` returned without reporting an error. It did not prove a Motion
Clip contained keys, that the keys could be read, or that the chosen bone
moved. The updated diagnostic separates these facts:

1. **Inspect existing motion clips / FK key counts** (read-only). Uses
   `RISkeletonComponent.GetClipCount/GetClip`,
   `RIClip.GetClipLength`, and `RControl.GetKeyCount`.
   A pre-existing clip (often reported as `existing_clip_count: 1`) will
   trigger the intentionally conservative P1 safety gate. Do not delete or
   overwrite a user's clip just to bypass this gate.
2. **Generate a new synthetic FK test** on a *visibly deforming bone*,
   preferably `CC_Base_L_Forearm` or `CC_Base_L_Upperarm`, 48 frames at
   24 FPS, using an **X-axis amplitude of 30 degrees**.
   This is NOT a generic AnyMove motion import and does not require DINOv2.
3. **Write a new clip in a disposable empty-motion scene only.** The writer
   now checks the clip count afterwards, calls `GetKeyCount()` on every FK
   control, attempts `GetValue()` at peak, and changes the playhead from
   frame 0 to the largest-magnitude rotation sample. It also calls
   `RINode.Update()` as described in the official Reallusion API.
4. **Check Timeline / Edit Motion Layer FK.** A successful FK controller
   write is not sufficient to prove visible animation. In iClone 8, IK may
   override FK, particularly on feet/hands.
5. **Report fields:** `status`, `fk_readback`,
   `pose_evaluation.bone_world_transform_changed`, `peak_frame`,
   `target_bone_name`, and `warnings`. If the pose is unchanged while
   FK keys exist, investigate IK weight, bone visibility, and conversion.

The writer DOES NOT automatically bake FK into IK. `BakeFkToIk` exists,
but it would modify character animation and should be a separate explicit,
user-approved workflow. It also DOES NOT touch existing clips.

Refs:
- https://wiki.reallusion.com/IC_8_Python_API:End_Effector_Animation
- https://wiki.reallusion.com/IC8_Python_API:RLPy_RINode
- User-supplied OpenPlugin_IC8 bundle: Legacy RLPy.py, RControl, RIClip,
  RISkeletonComponent and RGlobal signatures.
