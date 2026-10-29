"""Read-only Dragon source-pose diagnosis: do not compensate in iClone.

For the SAME capture compare:
1) Solver motion.json joints vs Blender evaluated joints.json (both world 3D)
2) Solver joints projected to original two-camera det.npz (2D pixels)
3) Pelvis root Quaternion converted from Bone-rest local to world up-axis.
Requires no Blender, PyTorch or iClone SDK; numpy is used only to load NPZ.
"""
import argparse
import json
import math
import pathlib
import statistics

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RIG = REPO / "assets" / "dragon" / "rigged" / "rig_info.json"
DEFAULT_CREATURE = REPO / "assets" / "dragon" / "creature.json"
SCHEMA = "anymove.iclone.source_pose_diagnosis/1"


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _statistics(seq):
    if not seq:
        return {"count":0,"mean":None,"median":None,"p95":None,"max":None}
    ordered=sorted(seq)
    pos=.95*(len(ordered)-1)
    lo=int(pos)
    hi=min(lo+1,len(ordered)-1)
    return {"count":len(seq),"mean":sum(seq)/len(seq),
            "median":statistics.median(seq),
            "p95":ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo),
            "max":ordered[-1]}


def _world_up_from_root(q, bone_rest):
    """R_world = B_rest @ R_root_local @ B_rest.T (solver convention)."""
    if len(q)!=4:
        raise ValueError("invalid Root Quaternion shape")
    norm=math.sqrt(sum(float(v)**2 for v in q))
    if norm<1e-9:
        raise ValueError("zero Root Quaternion")
    w,x,y,z=[float(a)/norm for a in q]
    rq=[
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    # Blender Rest matrices contain float32 rounding. Orthonormalize
    # columns to avoid false world-up tilt even when q is identity.
    cols=[[float(bone_rest[i][j]) for i in range(3)] for j in range(3)]
    def unit(v):
        n=math.sqrt(sum(x*x for x in v))
        if n<1e-9:
            raise ValueError("degenerate Pelvis Bone Rest rotation")
        return [x/n for x in v]
    x=unit(cols[0])
    projection=sum(a*b for a,b in zip(x,cols[1]))
    y=unit([cols[1][i]-projection*x[i] for i in range(3)])
    z=[x[1]*y[2]-x[2]*y[1],
       x[2]*y[0]-x[0]*y[2],
       x[0]*y[1]-x[1]*y[0]]
    if sum(a*b for a,b in zip(z,cols[2]))<.9:
        raise ValueError("Pelvis Bone Rest rotation is reflected")
    basis=[[x[i],y[i],z[i]] for i in range(3)]
    br=[[sum(basis[i][k]*rq[k][j] for k in range(3))
          for j in range(3)] for i in range(3)]
    world=[[sum(br[i][k]*basis[j][k] for k in range(3))
            for j in range(3)] for i in range(3)]
    return [world[i][2] for i in range(3)]


def _point3(row):
    if not isinstance(row,(list,tuple)) or len(row)!=3:
        raise ValueError("invalid 3D joint")
    x=[float(v) for v in row]
    if not all(math.isfinite(v) for v in x):
        raise ValueError("nonfinite joint")
    return x


def _projection(point, layout):
    x,y,z=point
    m=float(layout["M_PER_PX"])
    if m<=0:
        raise ValueError("invalid creature layout")
    v=float(layout["GROUND_V"])-z/m
    return ([float(layout["W_FRONT"])/2+x/m,v],
            [float(layout["W_SIDE"])/2-y/m,v])



# mocap/layout.py defaults to Wolf when MOCAP_CREATURE is not set.
WOLF_DEFAULT_LAYOUT={
    "W_FRONT":420,"W_SIDE":860,
    "M_PER_PX":1.9/720,"GROUND_V":630
}


def _camera_fit_statistics(motion,detections,layout):
    front=[]
    front_x=[]
    side=[]
    for frame, joints in enumerate(motion["joints"]):
        for bone, xyz in enumerate(joints):
            pf,ps=_projection(_point3(xyz),layout)
            df=detections["front_uv"][frame][bone]
            ds=detections["side_uv"][frame][bone]
            front_x.append(abs(pf[0]-df[0]))
            front.append(math.hypot(pf[0]-df[0],pf[1]-df[1]))
            side.append(math.hypot(ps[0]-ds[0],ps[1]-ds[1]))
    return {
        "front_error_px":_statistics(front),
        "front_x_only_error_px":_statistics(front_x),
        "side_error_px":_statistics(side)
    }


def compare_solver_camera_layouts(motion,detections,dragon_layout):
    """Which camera configuration best reproduces Solver's own losses?

    This is diagnostic, not a silent correction to the animation.
    Side view is strongest provenance signal: Solver time-warps only the
    FRONT view internally, and may have edited its per-frame landmarks.
    """
    dragon=_camera_fit_statistics(motion,detections,dragon_layout)
    wolf=_camera_fit_statistics(motion,detections,WOLF_DEFAULT_LAYOUT)
    reported=motion.get("reproj_px") or {}
    valid={k:float(v) for k,v in reported.items()
           if k in ("front","front_x_only","side") and
           type(v) in (int,float) and math.isfinite(v)}
    source_warp=(motion.get("sync") or {}).get("front_shift_frames")
    side_reported=valid.get("side")
    wolf_side=wolf["side_error_px"]["mean"]
    dragon_side=dragon["side_error_px"]["mean"]
    sw=(abs(wolf_side-side_reported) if side_reported is not None else None)
    sd=(abs(dragon_side-side_reported) if side_reported is not None else None)
    if (sw is not None and
        sw<=max(4.,.3*max(side_reported,1.)) and
        sd>max(15.,sw*3) and wolf_side<dragon_side*.65):
        inference="strong_evidence_solver_used_default_wolf_layout"
    elif (sd is not None and
          sd<=max(4.,.3*max(side_reported,1.)) and
          sw>max(15.,sd*3) and dragon_side<wolf_side*.65):
        inference="solver_used_expected_dragon_layout"
    elif (wolf_side<dragon_side*.55 and
          wolf["front_x_only_error_px"]["mean"] <
          dragon["front_x_only_error_px"]["mean"]*.65):
        inference="possible_wolf_fallback_need_original_solver_reprojection"
    else:
        inference="inconclusive_check_original_detection_provenance"
    return {
        "inference":inference,
        "expected_dragon_layout":dict(dragon_layout),
        "expected_dragon_projection":dragon,
        "wolf_fallback_layout":dict(WOLF_DEFAULT_LAYOUT),
        "wolf_fallback_projection":wolf,
        "solver_reported_reproj_px":valid,
        "side_mean_difference_from_solver_report_px":{
            "dragon":sd,"wolf":sw},
        "solver_front_was_time_aligned":source_warp is not None,
        "note":"iClone Blender errors cannot be corrected by retargeting "
               "if mocap.solve fitted joints under the wrong camera. "
               "Use the EXACT source det.npz, and re-solve after setting "
               "MOCAP_CREATURE=assets/dragon if Wolf fallback is confirmed."
    }

def _front_lean(a,b):
    height=abs(float(a[1])-float(b[1]))
    if height<1e-4:
        return None
    return math.degrees(math.atan2(float(b[0])-float(a[0]),height))


def diagnose(motion,rig,blender=None,detections=None,layout=None):
    """Return a diagnosis of exactly where an unwanted tilt appeared."""
    names=motion.get("joint_names")
    series=motion.get("joints")
    if not isinstance(names,list) or len(set(names))!=len(names):
        raise ValueError("motion.json missing unique joint_names")
    if not all(name in names for name in ("hips","head")):
        raise ValueError("Dragon head/hips joints absent")
    if not isinstance(series,list) or len(series)<2:
        raise ValueError("motion.json lacks per-frame 3D Solver joints")
    count=len(series)
    if (len(motion.get("root_quat",[]))!=count or
        len(motion.get("euler",[]))!=count or
        abs(float(motion.get("fps",0))-24)>1e-5):
        raise ValueError("motion.json must contain complete 24-FPS Dragon data")
    for row in series:
        if not isinstance(row,list) or len(row)!=len(names):
            raise ValueError("motion joint-frame dimensions differ")
        for pos in row:
            _point3(pos)
    bmap=None
    if blender is not None:
        bnames=blender.get("names")
        if (not isinstance(bnames,list) or set(bnames)!=set(names) or
            len(bnames)!=len(names) or len(set(bnames))!=len(names) or
            len(blender.get("joints",[]))!=count or
            abs(float(blender.get("fps",0))-24)>1e-5):
            raise ValueError("Blender joints.json is NOT the same motion/frame/joint set")
        bmap=[bnames.index(name) for name in names]
    if detections is not None:
        if (detections.get("names")!=names or
            abs(float(detections.get("fps",0))-24)>1e-5 or
            any(len(detections.get(key,[]))!=count for key in
                ("front_uv","side_uv","front_conf","side_conf"))):
            raise ValueError("det.npz must be the EXACT original 24-FPS detections")
    layout=layout or {"W_FRONT":640,"W_SIDE":640,
                       "M_PER_PX":.006875,"GROUND_V":630}
    root=[_world_up_from_root(q,rig["bones"][0]["matrix_local"])
          for q in motion["root_quat"]]
    tilt=[math.degrees(math.acos(max(-1.,min(1.,v[2])))) for v in root]
    lateral=[math.degrees(math.atan2(v[0],v[2])) for v in root]

    blender_dist=[]
    by_joint={n:[] for n in names}
    diff_frames=[]
    if bmap is not None:
        for frame in range(count):
            frame_max=0.
            row=blender["joints"][frame]
            for i,name in enumerate(names):
                a=_point3(series[frame][i])
                b=_point3(row[bmap[i]])
                error=math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))*100.
                blender_dist.append(error)
                by_joint[name].append(error)
                frame_max=max(frame_max,error)
            diff_frames.append({"frame":frame,"max_joint_error_cm":frame_max})
    dfront=[]
    dside=[]
    dfront_x=[]
    dlean=[]
    sample=[]
    if detections is not None:
        hips=names.index("hips")
        head=names.index("head")
        for frame in range(count):
            for i in range(len(names)):
                pf,ps=_projection(_point3(series[frame][i]),layout)
                f=detections["front_uv"][frame][i]
                s=detections["side_uv"][frame][i]
                dfront.append(math.hypot(pf[0]-f[0],pf[1]-f[1]))
                dside.append(math.hypot(ps[0]-s[0],ps[1]-s[1]))
                dfront_x.append(abs(pf[0]-f[0]))
            sh=detections["front_uv"][frame][hips]
            sd=detections["front_uv"][frame][head]
            ph,_=_projection(series[frame][hips],layout)
            pd,_=_projection(series[frame][head],layout)
            ref_angle=_front_lean(sh,sd)
            solve_angle=_front_lean(ph,pd)
            if ref_angle is not None and solve_angle is not None:
                dlean.append(abs(ref_angle-solve_angle))
            if frame in (0,count//4,count//2,3*count//4,count-1):
                sample.append({
                    "frame":frame,
                    "detected_front_lean_deg":ref_angle,
                    "solved_front_lean_deg":solve_angle,
                    "pelvis_world_up_tilt_deg":tilt[frame],
                    "pelvis_world_lateral_tilt_deg":lateral[frame]})
    bstat=_statistics(blender_dist)
    fstat=_statistics(dfront_x)
    sstat=_statistics(dside)
    lstat=_statistics(dlean)
    if blender is None:
        diagnosis="needs_matching_blender_joints_json"
    elif bstat["p95"]>2.0:
        diagnosis="blender_pose_or_rig_differs_from_solver"
    elif detections is None:
        diagnosis="solver_matches_blender_missing_source_detections"
    elif fstat["mean"]>10 or sstat["mean"]>15 or lstat["p95"]>5:
        diagnosis="solver_deviates_from_detected_video_joints"
    else:
        diagnosis="solver_blender_match_detected_2D_check_3D_visual_and_skin"
    return {
        "schema_version":SCHEMA,"result":diagnosis,"frames":count,
        "joint_count":len(names),"fps":float(motion["fps"]),
        "solver_to_blender_cm":{
            "status":"measured" if blender is not None else "not_available",
            "error_cm":bstat,
            "worst_joints":sorted(
                ({"name":name,**_statistics(err)}
                 for name,err in by_joint.items() if err),
                key=lambda x:x["mean"],reverse=True)[:12],
            "worst_frames":sorted(diff_frames,key=lambda v:v[
                "max_joint_error_cm"],reverse=True)[:12]},
        "source_detection_fit_px":{
            "status":"measured" if detections is not None else "not_available",
            "front_error":_statistics(dfront),
            "front_horizontal_error":fstat,
            "side_error":sstat,
            "front_lean_disagreement_deg":lstat,
            "sample_frames":sample},
        "camera_layout_comparison":(
            compare_solver_camera_layouts(motion,detections,layout)
            if detections is not None else
            {"inference":"original_det_npz_required"}),
        "pelvis_root_world_orientation":{
            "world_up_tilt_deg":_statistics(tilt),
            "signed_lateral_tilt_deg":_statistics(lateral),
            "first_frame_up_vector":root[0],
            "largest_tilt_frames":sorted(
                ({"frame":i,"world_up_tilt_deg":v,
                  "lateral_tilt_deg":lateral[i]}
                 for i,v in enumerate(tilt)),
                key=lambda x:x["world_up_tilt_deg"],reverse=True)[:12]},
        "note":"All comparisons use full integer frames. Blender-vs-Solver "
               "world joint mismatch implicates apply_motion/rigging. "
               "Solver-vs-detection mismatch implicates fitting. "
               "If both match but mesh looks wrong, inspect DINOv2 "
               "landmark labels, 3D ambiguity and skin weights. "
               "No iClone conversion is edited."
    }


def load_npz(path):
    import numpy as np
    with np.load(str(path),allow_pickle=False) as obj:
        keys=("front_uv","side_uv","front_conf","side_conf","names","fps")
        if any(key not in obj.files for key in keys):
            raise ValueError("det.npz missing required fields")
        return {"front_uv":obj["front_uv"].tolist(),
                "side_uv":obj["side_uv"].tolist(),
                "front_conf":obj["front_conf"].tolist(),
                "side_conf":obj["side_conf"].tolist(),
                "names":[str(n) for n in obj["names"].tolist()],
                "fps":float(obj["fps"])}


def main():
    p=argparse.ArgumentParser(description="Find the origin of a tilted Blender Dragon")
    p.add_argument("--motion",required=True)
    p.add_argument("--blender-joints")
    p.add_argument("--detections")
    p.add_argument("--rig-info",default=str(DEFAULT_RIG))
    p.add_argument("--creature",default=str(DEFAULT_CREATURE))
    p.add_argument("--output",required=True)
    a=p.parse_args()
    motion=_read(a.motion)
    blender=_read(a.blender_joints) if a.blender_joints else None
    # New Solver versions embed a source detection hash. Compare it when
    # present; old clips without provenance remain diagnostic-only.
    detections=load_npz(a.detections) if a.detections else None
    if a.detections and motion.get("source_detections_sha256"):
        import hashlib
        digest=hashlib.sha256()
        with open(a.detections,"rb") as f:
            for chunk in iter(lambda:f.read(1048576),b""):
                digest.update(chunk)
        if digest.hexdigest()!=motion["source_detections_sha256"]:
            raise ValueError(
                "The supplied det.npz is NOT the exact source of this "
                "motion.json (SHA-256 mismatch). Use the original detection "
                "or re-solve from a known source.")
    rig=_read(a.rig_info)
    creature=_read(a.creature)
    result=diagnose(motion,rig,blender=blender,detections=detections,
                    layout=creature.get("layout"))
    result["motion_solver_camera_layout"]=motion.get(
        "solver_camera_layout","not_recorded_in_legacy_motion")
    result["source_detections_hash_verified"]=bool(
        a.detections and motion.get("source_detections_sha256"))
    output=pathlib.Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",
                      encoding="utf-8")
    print("Result:",result["result"])
    print("Solver vs Blender (cm):",result["solver_to_blender_cm"]["error_cm"])
    print("Pelvis root world tilt (deg):",result[
        "pelvis_root_world_orientation"]["world_up_tilt_deg"])
    print("Camera Layout inference:",
          result["camera_layout_comparison"]["inference"])
    if detections is not None:
        compare=result["camera_layout_comparison"]
        print("Dragon side reprojection mean:",
              compare["expected_dragon_projection"]["side_error_px"]["mean"])
        print("Wolf side reprojection mean:",
              compare["wolf_fallback_projection"]["side_error_px"]["mean"])
        print("Original Solver reprojection:",
              compare["solver_reported_reproj_px"])
    print("Saved:",output)


if __name__=="__main__":
    main()
