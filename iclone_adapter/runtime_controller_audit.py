"""Read EVERY expected FK/Root Position sample in an iClone Motion Clip.

Read-only. Never calls SetTime/SetValue/AddClip/RemoveKey/Update.
The main Dragon import stores 42 Bone rotation tracks and one pelvis position
track: at 124 frames, this checks 15,996 scalar controller values.

This audits RControl.GetValue and GetKeyCount, NOT FK/IK evaluated pose,
mesh deformation or subframe interpolation. A scene might store all correct
keys yet be visually wrong due to IK/another animation layer.
"""
import math

from .bone_registry import build_bone_registry
from .motion_contract import validate_motion
from .p1_inspector import read_float_value
from .rlpy_reader import _get_skeleton, selected_object, snapshot
from .schema import ContractError

AUDIT_VERSION = "anymove.iclone.runtime_controller_fullframe/1"


def _axis_controls(clip,node,kind):
    layer=clip.GetControl("Layer",node)
    if layer is None:
        raise ContractError("RLPy Layer controller unavailable")
    block=layer.GetDataBlock()
    if block is None:
        raise ContractError("RLPy Layer DataBlock unavailable")
    axes=[block.GetControl(kind+"/"+kind+axis) for axis in "XYZ"]
    if any(c is None for c in axes):
        raise ContractError("one or more "+kind+" XYZ controls unavailable")
    return axes


def audit_full_controller_frames(motion,clip_index=None,rlpy=None,
                                 max_scalar_samples=200000,
                                 tolerance=1e-4):
    """Compare every integer-frame, every-axis controller value with FK JSON.

    Missing/unreadable channels generate a FAIL report. Skeleton identity,
    target Bone ambiguity and Clip index invalidity fail before sampling.
    The source Motion must already be written to the selected same-rig
    iClone actor.
    """
    if rlpy is None:
        import RLPy as rlpy
    if (type(tolerance) not in (float,int) or not math.isfinite(tolerance)
            or tolerance<=0):
        raise ContractError("positive finite readback tolerance required")
    obj=selected_object(rlpy)
    live=snapshot(obj,rlpy=rlpy)
    validate_motion(motion,live)
    skel=_get_skeleton(obj)
    count=int(skel.GetClipCount())
    if not count:
        raise ContractError("no iClone Motion Clip to audit")
    index=count-1 if clip_index is None else int(clip_index)
    if index<0 or index>=count:
        raise ContractError("invalid Motion Clip index")
    clip=skel.GetClip(index)
    if clip is None:
        raise ContractError("GetClip returned null")
    fps=rlpy.RGlobal.GetFps()
    host_fps=float(fps.ToDouble())
    if abs(host_fps-float(motion["fps"]))>1e-5:
        raise ContractError("project FPS does not match source FK FPS")
    frames=motion["frame_count"]
    rotations=motion["tracks"]
    positions=motion.get("position_tracks") or {}
    expected=(len(rotations)+len(positions))*frames*3
    if expected>max_scalar_samples:
        raise ContractError("too many scalar keys for one UI audit (%d); "
                            "split motion before sampling"%expected)
    registry,_=build_bone_registry(skel.GetSkinBones() or [])
    by_id={r["id"]:r for r in registry}
    requested=set(rotations)|set(positions)
    if requested-set(by_id):
        raise ContractError("requested tracks missing from live Skin Bones")
    timestamps=[rlpy.IndexedFrameTime(f,fps) for f in range(frames)]
    rows=[]
    problems=[]
    verified=0
    max_abs_error=0.0
    mismatch_scalar_samples=0
    wrong_key_counts=0
    for bone_id,kind,track in (
            [(bid,"Rotation",values) for bid,values in rotations.items()] +
            [(bid,"Position",values) for bid,values in positions.items()]):
        node=by_id[bone_id]["node"]
        name=by_id[bone_id]["name"]
        entry={
            "name":name,"bone_id":bone_id,"channel_kind":kind,
            "expected_keys_per_axis":frames,
            "key_counts_xyz":None,
            "samples_verified":0,
            "samples_expected":frames*3,
            "max_abs_error":None,
            "errors":[],
        }
        try:
            controls=_axis_controls(clip,node,kind)
            counts=[int(ctrl.GetKeyCount()) for ctrl in controls]
            entry["key_counts_xyz"]=counts
            for axis,keys in enumerate(counts):
                if keys!=frames:
                    wrong_key_counts+=1
                    entry["errors"].append(
                        "%s channel has %d keys; expected %d"%
                        ("XYZ"[axis],keys,frames))
            peak=0.
            for frame,stamp in enumerate(timestamps):
                for axis,ctrl in enumerate(controls):
                    try:
                        value=read_float_value(ctrl,stamp)
                        if not math.isfinite(value):
                            raise ContractError("non-finite GetValue response")
                    except Exception as exc:
                        if len(entry["errors"])<8:
                            entry["errors"].append(
                                "Frame %d axis %s: %s: %s"%
                                (frame,"XYZ"[axis],
                                 type(exc).__name__,exc))
                        continue
                    error=abs(value-float(track[frame][axis]))
                    peak=max(peak,error)
                    verified+=1
                    entry["samples_verified"]+=1
                    if error>tolerance:
                        mismatch_scalar_samples+=1
                        if len(entry["errors"])<8:
                            entry["errors"].append(
                                "Frame %d axis %s: read %.9g expected %.9g "
                                "(error %.5g)"%
                                (frame,"XYZ"[axis],value,
                                 track[frame][axis],error))
            entry["max_abs_error"]=peak
            max_abs_error=max(max_abs_error,peak)
        except Exception as exc:
            entry["errors"].append(
                "%s: %s"%(type(exc).__name__,exc))
        if entry["samples_verified"]!=entry["samples_expected"]:
            entry["errors"].append(
                "GetValue returned only %d of %d expected scalars"%
                (entry["samples_verified"],entry["samples_expected"]))
        if entry["errors"]:
            problems.append({"name":name,"type":kind,
                             "errors":entry["errors"][:8]})
        rows.append(entry)
    passed=(not problems and verified==expected and
            wrong_key_counts==0 and mismatch_scalar_samples==0)
    return {
        "schema_version":AUDIT_VERSION,
        "status":("all_frame_controller_values_match" if passed else
                  "controller_readback_failed"),
        "actor":str(obj.GetName()),
        "clip_count":count,
        "clip_index":index,
        "frame_count":frames,
        "bones_with_rotation_tracks":len(rotations),
        "bones_with_position_tracks":len(positions),
        "expected_scalar_samples":expected,
        "verified_scalar_samples":verified,
        "mismatched_scalar_samples":mismatch_scalar_samples,
        "channels_with_wrong_key_count":wrong_key_counts,
        "maximum_abs_error_in_channel_units":max_abs_error,
        "tolerance":float(tolerance),
        "track_reports":rows,
        "first_failures":problems[:24],
        "source_profile":(motion.get("pipeline") or {}).get("profile_id"),
        "source_motion_sha256":(motion.get("pipeline") or {}).get(
            "source_motion_sha256"),
        "all_integer_frame_controller_data_verified":passed,
        "evaluated_pose_verified":False,
        "subframe_interpolation_verified":False,
        "warning":"Only RControl key storage and exact-frame values verified. "
                  "This does NOT prove iClone evaluated pose, IK/physics "
                  "or between-frame interpolation. Timeline not changed.",
    }
