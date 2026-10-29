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
