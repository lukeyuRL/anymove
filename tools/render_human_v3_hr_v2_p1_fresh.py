"""Generate an AFTER-P1-FREEZE independent HR V2 315/45 motion test.

One new procedural seed per each of the 12 motion families, all distinct from
Train/Validation/old Test. Source V3 Nose Blender model is read-only.
This is a new 12-clip / 168-stereo-pair test, not natural-human validation.
"""
import hashlib,json,pathlib,math,sys,time

import bpy
import numpy as np

ROOT=pathlib.Path("F:/anymove")
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import render_human_v3_hr_v2_315_45 as source
from tools import render_human_v3_fullbody_finger_ab as prior
from tools import human_v3_fullbody_motion_v2 as movement

OUT=source.OUT/"p1_fresh_test"
SELECT=source.SELECTED_FRAMES

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,separators=(",",":")),encoding="utf-8")
    tmp.replace(path)

def run():
    if not (source.OUT/"reader_v2/full/best.pt").is_file():
        raise RuntimeError("Reader checkpoint not frozen")
    checkpoint=source.OUT/"reader_v2/full/best.pt"
    calibration=source.OUT/"reader_v2/full/temporal_skeleton_train_calibration.json"
    if not calibration.is_file():
        raise RuntimeError("P1 calibration not frozen")
    checkpoint_sha=sha(checkpoint)
    calibration_sha=sha(calibration)
    prior_manifest=json.loads((source.OUT/"dataset/manifest.json").read_text(encoding="utf8"))
    forbidden=set(prior_manifest["train_clips"]+prior_manifest["validation_clips"]+prior_manifest["test_clips"])
    arm,mesh,rest,scene,cameras=source.studio()
    names=source.marker_names(arm)
    if names!=prior_manifest["joint_names"]:
        raise RuntimeError("Different marker schema")
    clips=[]
    rows=[]
    gt_root=OUT/"motion_gt"
    gt_root.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    for family_idx,family in enumerate(movement.FAMILIES):
        seed=200000+family_idx*1049
        clip=f"hrv2_p1_fresh_{family}_s{seed}"
        if clip in forbidden:raise RuntimeError("Fresh test clip overlaps old splits")
        desc={"clip":clip,"family":family,"seed":seed,"split":"fresh_test"}
        xyz,audit=source.make_trajectory(arm,rest,names,desc)
        gt={
            "schema":"anymove.human_p1_fresh_test_gt/1",
            "source_blend_sha256":source.SHA256,
            "clip":desc,"joint_names":names,
            "joint_position_m":np.round(xyz,7).tolist(),
            "motion_quality":audit,
            "frame_count":107,
        }
        atomic(gt_root/(clip+".json"),gt)
        clips.append(desc)
        for frame in SELECT:
            old=prior.pose_changes
            try:
                prior.pose_changes=movement.pose_changes
                prior.apply_pose(arm,rest,names,family,seed,frame)
            finally:
                prior.pose_changes=old
            if np.max(np.linalg.norm(prior.joints_world(arm,names)-xyz[frame],axis=-1))>5e-6:
                raise RuntimeError("Fresh GT replay mismatch")
            for slot,angle in source.ANGLES.items():
                uv=source.projection(xyz[frame],angle)
                if not np.all((uv>1)&(uv<671)):
                    raise RuntimeError("Fresh camera clipping detected")
                path=OUT/"test"/slot/(clip+"_f"+str(frame).zfill(3)+".png")
                path.parent.mkdir(parents=True,exist_ok=True)
                if source.png_size(path)!=(672,672):
                    scene.camera=cameras[slot]
                    scene.render.filepath=str(path)
                    bpy.ops.render.render(write_still=True)
                if source.png_size(path)!=(672,672):
                    raise RuntimeError("Invalid independent Test image "+str(path))
                rows.append({"clip":clip,"family":family,"frame":frame,
                             "slot":slot,"camera_azimuth_deg":angle,
                             "file":str(path.relative_to(OUT)).replace("\\","/"),
                             "uv":np.round(uv,5).tolist()})
        print("HRV2_P1_FRESH_CLIP",len(clips),family,"images",len(rows),flush=True)
    if len(clips)!=12 or len(rows)!=336:
        raise RuntimeError("Fresh test incomplete")
    manifest={
        "schema":"anymove.human_p1_hrv2_fresh_test_manifest/1",
        "status":"complete",
        "source_blend_sha256":source.SHA256,
        "checkpoint_sha256":checkpoint_sha,
        "calibration_sha256":calibration_sha,
        "camera_angles_deg":[315,45],
        "width":672,"height":672,
        "metres_per_pixel":source.METRES_PER_PIXEL,
        "z_center":source.Z_CENTER,
        "joint_names":names,
        "fps":24,"sampled_frames":list(map(int,SELECT)),
        "clips":clips,"samples":rows,
        "notes":"Fresh unseen procedural motion seed; generated AFTER P1 tracker frozen on Train/Validation",
    }
    atomic(OUT/"manifest.json",manifest)
    print("HRV2_P1_FRESH_COMPLETE",json.dumps({"clips":len(clips),
       "images":len(rows),"pairs":len(rows)//2,
       "checkpoint_sha256":checkpoint_sha,
       "calibration_sha256":calibration_sha,
       "seconds":round(time.monotonic()-start,1)}),flush=True)
if __name__=="__main__":
    run()
