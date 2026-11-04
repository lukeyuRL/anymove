"""Extract human V3 Nose 67-marker parent skeleton, read-only Blender."""
import bpy, json, pathlib
import numpy as np
ROOT=pathlib.Path("F:/anymove")
obj=bpy.data.objects["Armature"]
names=[p.name for p in obj.pose.bones]
if len(names)!=55:raise RuntimeError("Unexpected V3 humanoid rig")
index={name:i for i,name in enumerate(names)}
parents={}
for pb in obj.pose.bones:
    parents[pb.name]=pb.parent.name if pb.parent is not None else None
for side in ("L","R"):
    for digit in ("Pinky","Ring","Mid","Index","Thumb"):
        parents[f"CC_Base_{side}_{digit}Tip"]=f"CC_Base_{side}_{digit}3"
    parents[f"CC_Base_{side}_ToeTip"]=f"CC_Base_{side}_ToeBase"
result={"marker_names":list(parents),"parent_name":parents}
assert len(parents)==67
out=ROOT/"runs/human/fullbody_finger_v2_hr_315_45/reader_v2/parent_skeleton.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print("HUMAN_PARENT_SKELETON_FILE",str(out),len(parents),flush=True)
