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


def _key_time(control):
    """Return RControl.MaxControlTime ticks, or None if unavailable."""
    get_last = getattr(control, "MaxControlTime", None)
    return _ticks(get_last()) if callable(get_last) else None


def _xyz(v):
    values = (float(v.x), float(v.y), float(v.z))
    if not all(math.isfinite(x) for x in values):
        raise ContractError("guard: nonfinite actor Transform xyz")
    return values


def _xyzw(v):
    values = (float(v.x), float(v.y), float(v.z), float(v.w))
    if not all(math.isfinite(x) for x in values):
        raise ContractError("guard: nonfinite actor Transform quaternion")
    size = math.sqrt(sum(x*x for x in values))
    if size < 1e-8:
        raise ContractError("guard: invalid actor Transform quaternion")
    return tuple(x/size for x in values)


def _actor_transform_values(transform):
    """SDK RTransform components; absolute quaternion dot handles q / -q."""
    return {
        "position":_xyz(transform.T()),
        "rotation_xyzw":_xyzw(transform.R()),
        "scale":_xyz(transform.S()),
    }


def _actor_transform_distance(a,b):
    distance = math.sqrt(sum((x-y)**2 for x,y in zip(a["position"],b["position"])))
    scale_error = max(abs(x-y) for x,y in zip(a["scale"],b["scale"]))
    rotation_dot=abs(sum(x*y for x,y in zip(a["rotation_xyzw"],b["rotation_xyzw"])))
    angle=math.degrees(2*math.acos(max(-1.,min(1.,rotation_dot))))
    return distance,scale_error,angle


def _read_actor_transform_keys(control,count,rlpy):
    """Use documented RTransformControl.GetTransformKeyAt(i, out_key).

    Only needed when there is >1 actor Transform key. For a single key
    at tick 0, MaxControlTime already proves no time-varying actor motion.
    """
    read = getattr(control,"GetTransformKeyAt",None)
    make_key=getattr(rlpy,"RTransformKey",None)
    if not callable(read) or not callable(make_key):
        raise ContractError(
            "guard: multiple actor Transform keys; SDK key readback is "
            "unavailable. Use a fresh project.")
    result=[]
    for i in range(count):
        key=make_key()
        status=read(i,key)
        if isinstance(status,(tuple,list)):
            if len(status)>1 and callable(getattr(status[1],"GetTransform",None)):
                key=status[1]
            status=status[0] if status else None
        failed=getattr(status,"IsError",None)
        if (callable(failed) and failed()) or status is False:
            raise ContractError("guard: could not read Actor Transform key %d"%i)
        result.append((_ticks(key.GetTime()),_actor_transform_values(key.GetTransform())))
    result.sort(key=lambda pair:pair[0])
    return result


def inspect_actor_transform(obj,rlpy):
    """Allow static actor Transform keys; reject actual motion / ambiguous API.

    Imported FBX normally may have a single initialization Transform key.
    One key at tick 0 is a static pose, not an animated actor. Any extra
    actor keys MUST have readable, identical T/R/S values. Independently
    keyed actor position/rotation/scale channels at nonzero time are
    conservatively refused, since they may override the group controller.
    """
    fn=getattr(obj,"GetControl",None)
    if not callable(fn):
        return {"status":"actor_transform_api_absent","key_count":None}
    control=fn("Transform")
    if control is None:
        return {"status":"actor_transform_control_absent","key_count":0}
    count=int(control.GetKeyCount())
    if count<0:
        raise ContractError("guard: invalid actor Transform key count")
    last=_key_time(control) if count else None
    result={"key_count":count,"last_key_ticks":last,"status":"no_transform_keys",
            "baseline_key_preserved":bool(count)}
    if count==1:
        if last is None:
            # A single key cannot be verified without its time.
            entries=_read_actor_transform_keys(control,count,rlpy)
            last=entries[0][0]
            result["last_key_ticks"]=last
        if last!=0:
            raise ContractError(
                "guard: lone actor Transform key is not at Frame 0 (%d ticks); "
                "cannot prove unchanged initial actor Pose."%last)
        result["status"]="single_static_frame0_transform_key"
    elif count>1:
        entries=_read_actor_transform_keys(control,count,rlpy)
        if entries[0][0]!=0:
            raise ContractError(
                "guard: actor Transform keys do not start at Frame 0")
        reference=entries[0][1]
        for ticks,actual in entries[1:]:
            pos,scale,angle=_actor_transform_distance(reference,actual)
            if pos>1e-4 or scale>1e-5 or angle>0.02:
                raise ContractError(
                    "guard: actor Transform is ANIMATED: key at %d ticks "
                    "differs (position %.6g, scale %.6g, angle %.5f deg)."%
                    (ticks,pos,scale,angle))
        result["status"]="multiple_constant_transform_keys"
        result["checked_key_count"]=len(entries)

    # Object Transform DataBlock sometimes holds independently keyed
    # per-axis controls. Never discard their motion merely because the
    # parent Transform key count says 1.
    get_block=getattr(control,"GetDataBlock",None)
    if callable(get_block):
        block=get_block()
        get_axis=getattr(block,"GetControl",None)
        if callable(get_axis):
            nonzero=[]
            for path in _ALL_CHANNELS:
                axis=get_axis(path)
                if axis is None:
                    continue
                keys=int(axis.GetKeyCount())
                if keys<0:
                    raise ContractError("guard: invalid actor channel count on "+path)
                if keys:
                    last_axis=_key_time(axis)
                    if last_axis is None:
                        raise ContractError(
                            "guard: actor Transform %s keys cannot be time-audited"%path)
                    if last_axis!=0:
                        nonzero.append((path,keys,last_axis))
            if nonzero:
                name,keys,ticks=nonzero[0]
                raise ContractError(
                    "guard: actor Transform channel may animate: %s has %d "
                    "key(s), last at %d ticks. Use a fresh project."%
                    (name,keys,ticks))
            result["separate_axis_channels"]="none_after_frame0"
        else:
            result["separate_axis_channels"]="not_exposed"
    else:
        result["separate_axis_channels"]="not_exposed"
    return result



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
    actor_transform=inspect_actor_transform(obj,rlpy)
    return {"existing_clip_count":count,"bones_inspected":inspected,
            "exposed_motion_empty":True,
            "actor_transform":actor_transform,
            "warning":"Unexposed IK constraints or imported base motion cannot be certified."}


def preflight_clean_dragon(obj,skel,motion,rlpy):
    pose=check_unposed_pose(motion.get("expected_unposed_world_matrices"),
                           snapshot(obj,rlpy=rlpy))
    clip=check_clean_clip(skel,obj,rlpy)
    return {"result":"guarded_clean_pose_and_exposed_layer",
            "pose":pose,"clip":clip}
