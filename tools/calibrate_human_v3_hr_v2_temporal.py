"""Train-only calibration of human HR V2 temporal/skeleton gates."""
import json
import pathlib
from mocap.temporal_skeleton_tracker import calibration_from_train

DATA=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45/dataset")
OUT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45/reader_v2")
def run():
    audit=json.loads((DATA/"audit.json").read_text(encoding="utf-8"))
    if audit["status"]!="PASS" or audit["image_count"]!=5376:
        raise RuntimeError("Dataset not audited")
    manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
    parent=json.loads((OUT/"parent_skeleton.json").read_text(encoding="utf-8"))
    if set(parent["marker_names"])!=set(manifest["joint_names"]) or len(parent["marker_names"])!=67:
        raise RuntimeError("Blender hierarchy and training marker order differ")
    cal=calibration_from_train(manifest,parent["parent_name"],DATA/"motion_gt")
    path=OUT/"full"/"temporal_skeleton_train_calibration.json"
    path.write_text(json.dumps(cal,indent=2),encoding="utf-8")
    print("TEMPORAL_CALIBRATION_PASS",json.dumps({
        "file":str(path),
        "clips":cal["calibration_clip_count"],
        "joints":len(cal["names"]),
        "source":cal["limits_source"],
        "finger_tip_spatial_bound_px":cal["cameras"]["front"][cal["names"].index("CC_Base_L_ThumbTip")]["length_gate_px"],
        "finger_tip_speed_gate_px_per_frame":cal["cameras"]["front"][cal["names"].index("CC_Base_L_ThumbTip")]["velocity_gate_px_per_frame"],
        "ankle_spatial_bound_px":cal["cameras"]["front"][cal["names"].index("CC_Base_L_Foot")]["length_gate_px"],
    }),flush=True)
if __name__=="__main__":
    run()
