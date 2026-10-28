"""Conservative, read-only preflight for an experimental Dragon import.

The Scene hierarchy hash alone cannot detect changed initial bone poses.
Compare ALL original unposed world matrices. A single default Clip must have
zero exposed FK/Position/Scale keys on ALL bones and zero length.
Cannot certify unexposed IK or non-FK state: fresh scene remains mandatory.
"""
import math

from .bone_registry import build_bone_registry
from .rlpy_reader import snapshot
from .schema import ContractError


def _ticks(value):
    fn = getattr(value, "ToInt", None)
    return int(fn()) if callable(fn) else int(value)


def _unit(v):
    norm = math.sqrt(sum(x*x for x in v))
    if norm < 1e-9:
        raise ContractError("degenerate rotation in Bone Matrix")
    return [x/norm for x in v]


def _rotation_columns(m):
    x = _unit([float(m[4*i]) for i in range(3)])
    y0 = [float(m[4*i+1]) for i in range(3)]
    dot = sum(a*b for a,b in zip(x,y0))
    y = _unit([a-dot*b for a,b in zip(y0,x)])
    z = [x[1]*y[2]-x[2]*y[1],
         x[2]*y[0]-x[0]*y[2],
         x[0]*y[1]-x[1]*y[0]]
    return [x,y,z]


def _angle(a,b):
    r1,r2 = _rotation_columns(a),_rotation_columns(b)
    trace = sum(sum(x*y for x,y in zip(u,v)) for u,v in zip(r1,r2))
    return math.degrees(math.acos(max(-1.,min(1.,(trace-1.)*.5))))


def check_unposed_pose(baseline,live,max_angle_deg=.5,max_distance=.25):
    if not isinstance(baseline,dict) or not baseline:
        raise ContractError("guard: no unposed Bone baseline. Regenerate FK JSON with current dragon_preview.")
    observed = {b["id"]:b for b in live["bones"]}
    if set(baseline)!=set(observed):
        raise ContractError("guard: baseline and live bone IDs differ")
    ranked=[]
    for bid,ref in baseline.items():
        cur=observed[bid]["snapshot_world_matrix"]
        if not isinstance(ref,list) or len(ref)!=16 or len(cur)!=16:
            raise ContractError("guard: invalid bone reference matrix for "+bid)
        if any(not math.isfinite(float(x)) for x in ref+cur):
            raise ContractError("guard: nonfinite bone world matrix")
        rot=_angle(ref,cur)
        pos=math.sqrt(sum((float(ref[i])-float(cur[i]))**2 for i in (3,7,11)))
        ranked.append({"name":observed[bid]["name"],"rotation_deg":rot,"position_native":pos})
    bad=[x for x in ranked if x["rotation_deg"]>max_angle_deg or x["position_native"]>max_distance]
    if bad:
        details="; ".join("%s %.3fdeg / %.3f units"%(x["name"],x["rotation_deg"],x["position_native"]) for x in bad[:5])
        raise ContractError("guard: Dragon is NOT in original unposed pose; open a fresh Dragon scene, do NOT Reset to Default. "+details)
    return {"checked_bone_count":len(ranked),
            "max_rotation_deg":max(x["rotation_deg"] for x in ranked),
            "max_position_native":max(x["position_native"] for x in ranked),
            "pose_match":True,
            "note":"Reference is current unposed snapshot, NOT a verified bind pose."}


_ALL_CHANNELS=tuple(kind+"/"+kind+axis for kind in ("Rotation","Position","Scale") for axis in "XYZ")


def check_clean_clip(skel,obj,rlpy):
    count=int(skel.GetClipCount())
    if count>1 or count<0:
        raise ContractError("guard: existing Clip count %d; open new clean Dragon project"%count)
    inspected=0
    if count==1:
        clip=skel.GetClip(0)
        if clip is None:
            raise ContractError("guard: default Clip is null")
        zero=rlpy.IndexedFrameTime(0,rlpy.RGlobal.GetFps())
        convert=getattr(clip,"ClipTimeToSceneTime",None)
        if not callable(convert) or _ticks(convert(zero))!=_ticks(zero):
            raise ContractError("guard: default Clip does not begin at Frame 0")
        for name in ("GetLength","GetClipLength"):
            fn=getattr(clip,name,None)
            if not callable(fn) or _ticks(fn())>0:
                raise ContractError("guard: default Clip has nonzero or unknown duration; possible hidden base motion")
        registry,_=build_bone_registry(skel.GetSkinBones() or [])
        for entry in registry:
            ctl=clip.GetControl("Layer",entry["node"])
            if ctl is None:
                continue
            block=ctl.GetDataBlock()
            if block is None:
                raise ContractError("guard: cannot read Layer on "+entry["name"])
            inspected+=1
            for path in _ALL_CHANNELS:
                axis=block.GetControl(path)
                if axis is None:
                    continue
                keys=int(axis.GetKeyCount())
                if keys!=0:
                    raise ContractError("guard: existing FK/Position/Scale keys on %s %s: %d. Use a NEW Dragon project."%(entry["name"],path,keys))
    get_control=getattr(obj,"GetControl",None)
    if callable(get_control):
        control=get_control("Transform")
        if control is not None and int(control.GetKeyCount())>0:
            raise ContractError("guard: actor already has Transform keys")
    return {"existing_clip_count":count,"bones_inspected":inspected,
            "exposed_motion_empty":True,
            "warning":"Unexposed IK constraints or imported base motion cannot be certified."}


def preflight_clean_dragon(obj,skel,motion,rlpy):
    pose=check_unposed_pose(motion.get("expected_unposed_world_matrices"),
                           snapshot(obj,rlpy=rlpy))
    clip=check_clean_clip(skel,obj,rlpy)
    return {"result":"guarded_clean_pose_and_exposed_layer",
            "pose":pose,"clip":clip}
