"""Independent Blender 5 reopen/animation replay QA for ORIGINAL human GT.

Loads each newly saved GT .blend in a fresh Blender process, scrubs ALL
107 frames through the saved animation action and compares evaluated FK
world joints against the independent previously saved motion_gt JSON.
Never edits or saves the .blend. Raises on wrong root/action/timebase.
"""
import bpy
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools.human_v3_fullbody_motion import finger_tip_names,toe_tip_names

BASE=ROOT/"runs/human/fullbody_finger_v2_hr_315_45/continuous_24fps_315_45"
FOLDER=BASE/"gt_blender"
EXPECTED_SOURCE_SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"

def sha256(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open("rb") as f:
        for part in iter(lambda:f.read(1048576),b""):
            h.update(part)
    return h.hexdigest()

def verify():
    blend=pathlib.Path(bpy.data.filepath)
    if not blend.name.endswith("_GT_107frames.blend") or blend.parent.resolve()!=FOLDER.resolve():
        raise RuntimeError("Expected a newly exported original GT animation .blend")
    clip=blend.name[:-len("_GT_107frames.blend")]
    meta_path=FOLDER/(clip+"_GT_export_report.json")
    report=json.loads(meta_path.read_text(encoding="utf8"))
    rec=json.loads((BASE/"motion_gt"/(clip+".json")).read_text(encoding="utf8"))
    if (report["file"]!=str(blend) or
        report["source_motion_gt_sha256"]!=sha256(BASE/"motion_gt"/(clip+".json")) or
        rec["clip"]["clip"]!=clip or
        rec["source_blend_sha256"]!=EXPECTED_SOURCE_SHA):
        raise RuntimeError("Original GT source or clip provenance mismatch")
    scene=bpy.context.scene
    arm=bpy.data.objects.get("Armature")
    mesh=bpy.data.objects.get("Dummy")
    if (not arm or not mesh or not arm.animation_data or
        not arm.animation_data.action or len(arm.pose.bones)!=55 or
        len(mesh.data.vertices)!=30188 or
        not mesh.data.color_attributes.get("AnyMoveHumanSkinMax53")):
        raise RuntimeError("Saved scene did not retain armature action/model/vertex colors")
    names=[b.name for b in arm.pose.bones]+finger_tip_names()+toe_tip_names()
    if rec["joint_names"]!=names or len(names)!=67:
        raise RuntimeError("Reopened GT joint names do not match original GT")
    if (scene.frame_start!=0 or scene.frame_end!=106 or scene.render.fps!=24
        or abs(scene.render.fps_base-1)>1e-7 or
        scene.get("AnyMove_motion_source")!="ORIGINAL_PROCEDURAL_GROUND_TRUTH"
        or not scene.get("AnyMove_GT_not_solver_output",False)):
        raise RuntimeError("Wrong animation timebase/provenance; not original GT")
    expected=np.asarray(rec["joint_position_m"],dtype=np.float64)
    if expected.shape!=(107,67,3):
        raise RuntimeError("Wrong GT shape")
    errors=[]
    coords=[]
    for frame in range(107):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        pts=[arm.matrix_world @ arm.pose.bones[n].head for n in names[:55]]
        pts.extend([arm.matrix_world @ arm.pose.bones[n.replace("Tip","3")].tail for n in finger_tip_names()])
        pts.extend([arm.matrix_world @ arm.pose.bones[n.replace("ToeTip","ToeBase")].tail for n in toe_tip_names()])
        actual=np.asarray(pts,dtype=np.float64)
        if not np.isfinite(actual).all():
            raise RuntimeError("NaN in reloaded Blender GT animation frame")
        errors.append(np.linalg.norm(actual-expected[frame],axis=-1)*100)
        coords.append(actual)
    errors=np.asarray(errors)
    coords=np.asarray(coords)
    maxerr=float(errors.max())
    # A 0.01cm threshold is generous relative to GT file rounding at
    # 1e-7 metre and Blender internal float32 matrix math.
    if maxerr>=0.01:
        raise RuntimeError("Reopened Blender GT deviates from source GT JSON: "+str(maxerr)+" cm")
    # Motion must actually move legs and independent fingertip chains.
    indices={n:i for i,n in enumerate(names)}
    hand_indices=[indices[f"CC_Base_{s}_Hand"] for s in ("L","R")]
    leg_indices=[indices[f"CC_Base_{s}_{j}"] for s in ("L","R") for j in ("Thigh","Calf","Foot","ToeBase")]
    finger_indices=[indices[n] for n in finger_tip_names()]
    def displacement(subset):
        return float(np.linalg.norm(coords[:,subset]-coords[0:1,subset],axis=-1).max())
    leg_motion=displacement(leg_indices)
    hand_motion=displacement(hand_indices)
    finger_motion=displacement(finger_indices)
    if min(leg_motion,hand_motion,finger_motion)<0.03:
        raise RuntimeError("Saved GT animation has static fullbody/fingers")
    report.update({
        "status":"REOPENED_FULL_107FRAMES_PASS",
        "reopened_blend_sha256":sha256(blend),
        "reopen_fk_mean_world_joint_difference_cm":float(errors.mean()),
        "reopen_fk_max_world_joint_difference_cm":maxerr,
        "reopen_fk_p95_world_joint_difference_cm":float(np.percentile(errors,95)),
        "reopen_leg_max_displacement_m":leg_motion,
        "reopen_hand_max_displacement_m":hand_motion,
        "reopen_fingertip_max_displacement_m":finger_motion,
        "all_source_frames_present":True,
        "saved_action_present":True,
        "original_source_unchanged":sha256(ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend")==EXPECTED_SOURCE_SHA,
    })
    meta_path.write_text(json.dumps(report,indent=2),encoding="utf8")
    print("HUMAN_V3_ORIGINAL_GT_REOPEN_PASS",json.dumps({
        "clip":clip,"blend":str(blend),
        "frames":107,"fps":24,
        "max_gt_error_cm":round(maxerr,8),
        "leg_travel_m":round(leg_motion,4),
        "hand_travel_m":round(hand_motion,4),
        "fingertip_travel_m":round(finger_motion,4),
        "action_present":True,
        "source_blend_unchanged":report["original_source_unchanged"],
    }),flush=True)

if __name__=="__main__":
    verify()
