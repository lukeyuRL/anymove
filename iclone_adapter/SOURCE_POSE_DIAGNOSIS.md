# Dragon source-pose diagnosis when Blender and iClone look wrong

The already verified iClone FK controller values prove correct import,
not a correct AnyMove animation. The Dragon original 124-frame det.npz
front view has nearly vertically aligned Head and Hips at sampled frames.
A strong whole-body lateral lean is therefore suspicious.

This read-only script compares three independent data sets:
- Solver: runs/dragon/clips/roar/motion.json, contains world 3D joints
- Blender: full_root_reference/joints.json, evaluated 3D joints
- Original DINOv2: examples/dragon_roar/det.npz, 2D video joints

Run from Windows PowerShell in F:\anymove:

    git pull --ff-only origin iClone

    .venv\Scripts\python.exe -m iclone_adapter.source_pose_diagnose --motion runs\dragon\clips\roar\motion.json --blender-joints runs\dragon\clips\roar\full_root_reference\joints.json --detections examples\dragon_roar\det.npz --output F:\anymove_test\dragon_source_pose_diagnosis.json

Use the EXACT det.npz that originally generated the current motion.json.
If it came from runs/dragon/clips/roar/det.npz use that location instead.
The diagnosis refuses mismatched FPS, joint names or frame counts.

Interpretation:
- blender_pose_or_rig_differs_from_solver: Blender apply/rig error
- solver_deviates_from_detected_video_joints: source solver/fit error
- solver_blender_match_detected_2D_check_3D_visual_and_skin:
  source detections, depth ambiguity, foot contact or skinned mesh issue

World Pelvis tilt is computed as B_rest * R(root_quat) * B_rest.T.
Do not mistake the Pelvis bone's rotated local Rest axis for global lean.
This tool does not edit animation or iClone FK mapping.


## Additional camera-layout proof and prevention

The source-pose report has an extraordinary discrepancy: Solver-to-Blender
joints are within 0.0091 cm, but Solver-to-Dragon-detections mean reprojection
error is ~68px front and ~72px side, while Root is already laterally
~22.23 degrees off at Frame 0. An important candidate is **incorrect
camera Layout when mocap.solve was originally run without
MOCAP_CREATURE=assets/dragon**. In that situation mocap/layout.py silently
uses Wolf's W_FRONT=420, W_SIDE=860, M_PER_PX=1.9/720 instead of
Dragon's W_FRONT=640, W_SIDE=640, M_PER_PX=.006875.

The diagnostic now calculates Solver-joint projections BOTH ways and
compares the side mean to the original motion.json.reproj_px.side value.
The side channel is important because the Solver may time-warp front
landmarks internally. If the Wolf reprojection matches the Solver's
reported side error while Dragon reprojection is ~72px, this is direct
evidence that the wrong camera was loaded during solving.

The original mocap.solve now has a fail-closed check against the Rig's
creature.json and stops before optimizing if the active layout mismatches.
Set the environment BEFORE launching Python:

    $env:MOCAP_CREATURE = 'assets/dragon'

Then use mocap.solve with the original det.npz and a **new** output file.
Do not delete any previous animation or attempt to correct Root by a
fixed Quaternion. Compare new Solver motion against det.npz first.


## v0.7.2 — exact camera and source detection provenance

Regenerated mocap.solve output now embeds the loaded
solver_camera_layout and the source_detections_sha256 (raw NPZ bytes).

The next diagnosis uses the source hash to REFUSE a mismatched det.npz
even when both clips happen to have 124 frames and the same joint names.
The camera candidate test compares reprojection errors to the values
actually recorded by the Solver when it produced the motion.

Correct Dragon solve example from PowerShell:

    cd F:\anymove
    git pull --ff-only origin iClone
    $env:MOCAP_CREATURE = 'assets/dragon'
    New-Item -ItemType Directory -Force F:\anymove_test\roar_corrected | Out-Null
    .venv\Scripts\python.exe -m mocap.solve examples\dragon_roar\det.npz assets\dragon\rigged\rig_info.json F:\anymove_test\roar_corrected\motion.json

Inspect the console: Solver camera layout should include
W_FRONT=640, W_SIDE=640, M_PER_PX=0.006875, GROUND_V=630. Then
repeat source_pose_diagnose against the NEW motion and SAME det.npz;
do not reuse the old Blender joints.json in that new-motion comparison.

Once source-video fit is good, render a new Blender reference and
import the new motion.json into an unposed Dragon iClone project.
