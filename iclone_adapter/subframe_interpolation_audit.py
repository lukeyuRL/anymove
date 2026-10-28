"""Read-only native FK and Pelvis Position interpolation midpoint audit.

This probes RFloatControl.GetValue at every midpoint between integer frame
keys. Deviations from linear channel interpolation are DIAGNOSTIC only:
matching Blender F-Curve interpolation and rendered bone poses is a separate
acceptance gate. Never sets time, modifies clips, or adds animation keys.
"""
import math

from .bone_registry import build_bone_registry
from .motion_contract import validate_motion
from .rlpy_reader import selected_object, _get_skeleton, snapshot
from .runtime_controller_audit import _axis_controls
from .schema import ContractError

VERSION = "anymove.iclone.subframe_controller_survey/1"


def _ticks(value):
    # iClone 8 RTime uses 6000 ticks per second and exposes ToInt().
    # Prefer ToInt to the legacy GetValue time-unit convention.
    get = getattr(value, "ToInt", None)
    if callable(get):
        raw = get()
    else:
        get = getattr(value, "GetValue", None)
        raw = get() if callable(get) else value
    if type(raw) not in (int,float) or not math.isfinite(raw):
        raise ContractError("invalid SDK RTime tick")
    tick = int(round(raw))
    if abs(raw-tick)>1e-4:
        raise ContractError("fractional SDK ticks cannot construct midpoint")
    return tick


def _midpoints(rlpy,fps,count):
    time_class = getattr(rlpy, "RTime", None)
    from_value = getattr(time_class, "FromValue", None)
    if not callable(from_value):
        raise ContractError(
            "iClone 8 RLPy.RTime.FromValue unavailable for midpoint sampling; "
            "RTime(tick) is NOT supported")
    anchors = [_ticks(rlpy.IndexedFrameTime(i,fps)) for i in range(count)]
    samples=[]
    for i in range(count-1):
        a,b=anchors[i],anchors[i+1]
        if b-a<2:
            raise ContractError("no representable subframe RTime tick at frame %d"%i)
        tick=(a+b)//2
        if not a<tick<b:
            raise ContractError("invalid subframe time at frame %d"%i)
        # iClone 8 requires the static factory FromValue(ticks).
        # RTime(ticks) raises TypeError in real iClone Python 3.8.
        try:
            instant=from_value(tick)
        except Exception as exc:
            raise ContractError(
                "RTime.FromValue(%d) failed: %s: %s" %
                (tick, type(exc).__name__, exc))
        if instant is None or _ticks(instant)!=tick:
            raise ContractError("RTime.FromValue returned a different tick")
        samples.append((i,instant,float(tick-a)/(b-a),tick))
    return samples


def _value(control,instant):
    raw=control.GetValue(instant,0.0)
    if not isinstance(raw,(tuple,list)) or len(raw)<2:
        raise ContractError("RFloatControl.GetValue must return status and value")
    status,number=raw[0],raw[1]
    if status is False:
        raise ContractError("RFloatControl.GetValue returned failure status")
    is_error=getattr(status,"IsError",None)
    if callable(is_error) and is_error():
        raise ContractError("RFloatControl.GetValue returned SDK error")
    value=float(number)
    if not math.isfinite(value):
        raise ContractError("RFloatControl.GetValue returned nonfinite scalar")
    return value


def survey_subframe_controllers(motion,clip_index=None,rlpy=None,
                                linear_tolerance=1e-4,max_scalar_samples=250000):
    if rlpy is None:
        import RLPy as rlpy
    if (type(linear_tolerance) not in (float,int)
        or not math.isfinite(linear_tolerance) or linear_tolerance<=0):
        raise ContractError("linear_tolerance must be finite and positive")
    obj=selected_object(rlpy)
    live=snapshot(obj,rlpy=rlpy)
    validate_motion(motion,live)
    skel=_get_skeleton(obj)
    count=int(skel.GetClipCount())
    index=count-1 if clip_index is None else int(clip_index)
    if count<=0 or not 0<=index<count:
        raise ContractError("no selected Motion Clip for subframe inspection")
    clip=skel.GetClip(index)
    if clip is None:
        raise ContractError("GetClip returned None")
    fps=rlpy.RGlobal.GetFps()
    if abs(float(fps.ToDouble())-float(motion["fps"]))>1e-5:
        raise ContractError("project FPS mismatch")
    tracks=([(bid,"Rotation",rows) for bid,rows in motion["tracks"].items()]
            +[(bid,"Position",rows)
              for bid,rows in (motion.get("position_tracks") or {}).items()])
    expected=(motion["frame_count"]-1)*len(tracks)*3
    if expected>max_scalar_samples:
        raise ContractError("subframe scalar sample limit exceeded")
    times=_midpoints(rlpy,fps,motion["frame_count"])
    registry,_=build_bone_registry(skel.GetSkinBones() or [])
    nodes={entry["id"]:entry for entry in registry}
    if any(bid not in nodes for bid,_,_ in tracks):
        raise ContractError("motion includes missing Bone IDs")

    initial_time=_ticks(rlpy.RGlobal.GetTime())
    observed=0
    nonlinear=0
    overshoot=0
    maximum={"Rotation":0.0,"Position":0.0}
    per_track=[]
    offenders=[]
    failures=[]
    for bid,kind,rows in tracks:
        entry={"name":nodes[bid]["name"],"kind":kind,
               "samples_expected":(motion["frame_count"]-1)*3,
               "samples_read":0,"deviation_count":0,"overshoot_count":0,
               "max_abs_linear_deviation":0.0,"errors":[]}
        try:
            axes=_axis_controls(clip,nodes[bid]["node"],kind)
            for frame,mid,alpha,tick in times:
                for axis,control in enumerate(axes):
                    try:
                        current=_value(control,mid)
                    except Exception as exc:
                        if len(entry["errors"])<6:
                            entry["errors"].append(
                                "Frames %d-%d axis %s: %s: %s"%
                                (frame,frame+1,"XYZ"[axis],
                                 type(exc).__name__,exc))
                        continue
                    start,end=float(rows[frame][axis]),float(rows[frame+1][axis])
                    linear=start+(end-start)*alpha
                    difference=abs(current-linear)
                    outside=(current<min(start,end)-linear_tolerance or
                             current>max(start,end)+linear_tolerance)
                    observed+=1
                    entry["samples_read"]+=1
                    entry["max_abs_linear_deviation"]=max(
                        entry["max_abs_linear_deviation"],difference)
                    maximum[kind]=max(maximum[kind],difference)
                    if difference>linear_tolerance:
                        nonlinear+=1
                        entry["deviation_count"]+=1
                    if outside:
                        overshoot+=1
                        entry["overshoot_count"]+=1
                    if difference>linear_tolerance or outside:
                        offenders.append({
                            "bone":entry["name"],"kind":kind,
                            "axis":"XYZ"[axis],"frame_pair":[frame,frame+1],
                            "mid_RTime_ticks":tick,"fraction":alpha,
                            "endpoints":[start,end],
                            "native_value":current,"linear_value":linear,
                            "absolute_deviation":difference,
                            "outside_endpoint_range":outside})
        except Exception as exc:
            entry["errors"].append("%s: %s"%(type(exc).__name__,exc))
        if entry["samples_read"]!=entry["samples_expected"]:
            entry["errors"].append(
                "read %d / %d scalar samples"%
                (entry["samples_read"],entry["samples_expected"]))
        if entry["errors"]:
            failures.append({"bone":entry["name"],"kind":kind,
                             "errors":entry["errors"]})
        per_track.append(entry)
    stable=_ticks(rlpy.RGlobal.GetTime())==initial_time
    if not stable:
        failures.append({"bone":"scene","errors":["Timeline time changed unexpectedly"]})
    if failures or observed!=expected:
        status="subframe_readback_incomplete"
    elif nonlinear:
        status="native_subframes_deviate_from_linear_hypothesis"
    else:
        status="native_subframes_match_linear_hypothesis"
    offenders.sort(key=lambda item:item["absolute_deviation"],reverse=True)
    return {
        "schema_version":VERSION,"status":status,
        "actor":str(obj.GetName()),"clip_index":index,
        "frame_count":motion["frame_count"],
        "frame_intervals":motion["frame_count"]-1,
        "rotation_tracks":len(motion["tracks"]),
        "position_tracks":len(motion.get("position_tracks") or {}),
        "expected_scalar_samples":expected,
        "read_scalar_samples":observed,
        "nonlinear_vs_linear_scalar_samples":nonlinear,
        "overshooting_scalar_samples":overshoot,
        "max_abs_deviation_from_linear_by_kind":maximum,
        "linear_tolerance":linear_tolerance,
        "timeline_unchanged":stable,
        "native_subframe_values_readable":not failures and observed==expected,
        "curve_policy_verified_against_blender":False,
        "evaluated_pose_verified":False,
        "source_profile":(motion.get("pipeline") or {}).get("profile_id"),
        "track_reports":per_track,"largest_deviations":offenders[:32],
        "first_failures":failures[:15],
        "warning":"GetValue returns native iClone interpolation. Nonlinear "
                  "midpoints are not automatically wrong: Blender F-Curves "
                  "can also be nonlinear. Rendered pose, IK and true "
                  "Blender subframe equivalence were NOT checked."
    }
