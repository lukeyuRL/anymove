"""Generate FRESH Human V3-nose fullbody/finger heldout poses after guard freeze.

No GT from the previously examined test is reused. Different procedural
families/seeds, same audited Blender model and camera geometry. No H3.
"""
import json,math,pathlib,sys
import bpy
import numpy as np

ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools import render_human_v3_fullbody_finger_ab as src
from tools.human_v3_fullbody_motion import finger_tip_names,toe_tip_names

OUT=src.OUT/"fresh_test_20261103"
FRESH_CLIPS=[
    ("march_open_close", 9413),
    ("crouch_fist", 9541),
    ("kick_and_wave", 9671),
    ("balance_point", 9781),
]
FRAMES=(8,24,40,56,72,88,96)

def main():
    arm,rest,scene,camera=src.make_scene()
    names=[b.name for b in arm.pose.bones]+finger_tip_names()+toe_tip_names()
    original=json.loads((src.OUT/"dataset"/"baseline_0_90"/"manifest.json").read_text(encoding="utf-8"))
    if names!=original["joint_names"] or len(names)!=67:
        raise RuntimeError("Fresh evaluation schema does not match trained Readers")
    forbidden=set(original["train_clips"]+original["validation_clips"]+original["test_clips"])
    if any("fresh_"+fam+"_"+str(seed) in forbidden for fam,seed in FRESH_CLIPS):
        raise RuntimeError("Fresh clip overlaps previous training/validation/test")
    rows={k:[] for k in src.CAMERAS}
    clipnames=[]
    audit=[]
    for family,seed in FRESH_CLIPS:
        clip="fresh_"+family+"_"+str(seed)
        clipnames.append(clip)
        xyz=[]
        for frame in range(src.FRAME_COUNT):
            src.apply_pose(arm,rest,names,family,seed,frame)
            xyz.append(src.joints_world(arm,names))
        xyz=np.asarray(xyz)
        side_idx=[names.index("CC_Base_"+s+"_ToeTip") for s in ("L","R")]
        finger_idx=[names.index(x) for x in finger_tip_names()]
        leg_idx=[names.index("CC_Base_"+s+"_"+part) for s in ("L","R") for part in ("Thigh","Calf","Foot")]
        def max_travel(i):
            return float(np.linalg.norm(xyz[:,i]-xyz[0:1,i],axis=-1).max())
        leg=max_travel(leg_idx)
        toe=max_travel(side_idx)
        fingers=max_travel(finger_idx)
        if leg<.04 or toe<.04 or fingers<.02:raise RuntimeError("Fresh clip lacks fullbody motion")
        audit.append({"clip":clip,"family":family,"seed":seed,
                      "leg_max_travel_m":leg,"toe_max_travel_m":toe,
                      "finger_max_travel_m":fingers})
        dest=OUT/"motion_gt"/(clip+".json")
        dest.parent.mkdir(parents=True,exist_ok=True)
        record={"schema":"anymove.human_fullbody_fresh_test_gt/1",
                "source_blend_sha256":src.SHA,
                "clip":{"split":"test","clip":clip,"family":family,"seed":seed},
                "joint_names":names,
                "joint_position_m":np.round(xyz,7).tolist(),
                "frames":src.FRAME_COUNT}
        if dest.is_file():
            previous=json.loads(dest.read_text(encoding="utf-8"))
            if previous["joint_names"]!=names or previous["clip"]!=record["clip"]:
                raise RuntimeError("Existing fresh test GT contract conflicts")
        dest.write_text(json.dumps(record),encoding="utf-8")
        for frame in FRAMES:
            src.apply_pose(arm,rest,names,family,seed,frame)
            if np.max(np.linalg.norm(src.joints_world(arm,names)-xyz[frame],axis=-1))>1e-7:
                raise RuntimeError("Fresh GT replay failed")
            for pair,angles in src.CAMERAS.items():
                for slot,angle in zip(("front","side"),angles):
                    xy=src.projection(xyz[frame],angle)
                    inside=(xy[...,0]>=1)&(xy[...,0]<src.W-1)&(xy[...,1]>=1)&(xy[...,1]<src.H-1)
                    if not np.all(inside):
                        raise RuntimeError("Fresh test camera clipping: "+clip+"/"+str(angle))
                    a=math.radians(angle)
                    camera.location=(-20*math.sin(a),-20*math.cos(a),src.Z_CENTER)
                    camera.rotation_euler=(math.pi/2,0,-a)
                    image=OUT/pair/"test"/(clip+"_f"+str(frame).zfill(3)+"_"+slot+".png")
                    image.parent.mkdir(parents=True,exist_ok=True)
                    if not image.is_file() or image.stat().st_size<1000:
                        scene.render.filepath=str(image)
                        bpy.ops.render.render(write_still=True)
                    rows[pair].append({
                        "split":"test","clip":clip,"frame":frame,
                        "slot":slot,"camera_azimuth_deg":angle,
                        "path":str(image.relative_to(OUT)).replace("\\","/"),
                        "uv":np.round(xy,5).tolist(),
                    })
        print("FRESH_CLIP_OK",clip,flush=True)
    for pair,angles in src.CAMERAS.items():
        manifest={"schema":"anymove.fullbody_finger_fresh_test/1",
            "status":"complete","source_blend_sha256":src.SHA,
            "source_blend":str(src.SCENE),"joint_names":names,
            "angles_deg":list(angles),"width":src.W,"height":src.H,
            "metres_per_pixel":src.M_PER_PX,"z_center":src.Z_CENTER,
            "test_clips":clipnames,"test_frames":list(FRAMES),
            "samples":rows[pair]}
        target=OUT/pair/"manifest.json"
        target.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (OUT/"motion_qa.json").write_text(json.dumps({"status":"PASS","clips":audit},indent=2),encoding="utf-8")
    print("FRESH_TEST_RENDER_PASS",json.dumps({"clips":clipnames,
        "views_each_pair":len(rows["baseline_0_90"]),"images":sum(map(len,rows.values())),
        "out":str(OUT)}),flush=True)

if __name__=="__main__":
    main()
