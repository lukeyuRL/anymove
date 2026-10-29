"""Backed-up Root POSITION BASIS correction with a fixed 1.0 multiplier.

The user's confirmed iClone Pelvis Position scale is one. The old inverse
Pelvis Rest axes turn source +Z lift into negative local Z; the new candidate
writes full source XYZ instead. Only source-verified Position keys are
changed; FK rotations and original motion.json are untouched. Actual iClone
world motion must still be inspected visually after the new iProject save.
"""
import hashlib
import json
import math
import pathlib

from .bone_registry import build_bone_registry
from .motion_writer import _require_sdk_ok, _ticks
from .p1_inspector import read_float_value
from .rlpy_reader import _get_skeleton, selected_object
from .root_motion import convert_pelvis_root
from .schema import ContractError
from .verify_import import DEFAULT_SOURCE


def _load_source(motion_path):
    path=pathlib.Path(motion_path)
    if not path.is_file() or path.suffix.lower()!=".json":
        raise ContractError("Select the original AnyMove Dragon motion.json")
    with open(str(path),encoding="utf-8") as stream:
        motion=json.load(stream)
    with open(str(DEFAULT_SOURCE),encoding="utf-8") as stream:
        info=json.load(stream)
    names=[b["name"] for b in info["bones"]]
    if (motion.get("bones")!=names or
        type(motion.get("fps")) not in (int,float) or
        motion.get("fps")!=24.0):
        raise ContractError("Root calibration requires the original 42-Bone 24-FPS Dragon motion")
    frames=motion.get("trans")
    if not isinstance(frames,list) or len(frames)<2:
        raise ContractError("source root translation missing")
    _,expected,_=convert_pelvis_root(
        motion,info,"source-pelvis",mode="full",position_scale=1.0,
        position_space="pelvis_local")
    return path,motion,expected


def _controls(rlpy):
    obj=selected_object(rlpy)
    skel=_get_skeleton(obj)
    registry,_=build_bone_registry(skel.GetSkinBones() or [])
    names=[r["name"] for r in registry]
    if names.count("pelvis")!=1:
        raise ContractError("selected Dragon requires exactly one pelvis Skin Bone")
    if int(skel.GetClipCount())!=1:
        raise ContractError(
            "Root scale repair requires exactly one imported Motion Clip; "
            "do not edit unrelated clips")
    clip=skel.GetClip(0)
    if clip is None:
        raise ContractError("Motion Clip 0 is not readable")
    fps=rlpy.RGlobal.GetFps()
    if not callable(getattr(fps,"ToDouble",None)) or abs(
            float(fps.ToDouble())-24.0)>1e-5:
        raise ContractError("iClone project must be 24 FPS")
    zero=rlpy.IndexedFrameTime(0,fps)
    converted=getattr(clip,"ClipTimeToSceneTime",None)
    if not callable(converted):
        raise ContractError(
            "Cannot verify Motion Clip origin at Frame 0: "
            "RIClip.ClipTimeToSceneTime unavailable")
    try:
        # iClone 8 RTime exposes ToInt() (6000 ticks/second), NOT the
        # legacy GetValue() used in iClone 7 examples. Share the checked
        # conversion with the normal FK writer and scene guard.
        clip_zero_ticks=_ticks(zero)
        scene_zero_ticks=_ticks(converted(zero))
    except (AttributeError,TypeError,ValueError,OverflowError) as exc:
        raise ContractError(
            "Cannot read Motion Clip time as iClone 8 RTime.ToInt ticks: "
            +str(exc)) from exc
    if scene_zero_ticks!=clip_zero_ticks:
        raise ContractError(
            "Motion Clip is not at Frame 0: clip origin maps to "
            "scene tick %d (expected %d); root repair refused"%
            (scene_zero_ticks,clip_zero_ticks))
    node=next(r["node"] for r in registry if r["name"]=="pelvis")
    layer=clip.GetControl("Layer",node)
    block=layer.GetDataBlock() if layer is not None else None
    if block is None:
        raise ContractError("Pelvis Layer control unavailable")
    axes=[block.GetControl("Position/Position"+axis) for axis in "XYZ"]
    if any(axis is None for axis in axes):
        raise ContractError("Pelvis Position XYZ channels unavailable")
    return obj,skel,clip,fps,axes


def inspect_root_scale(motion_path,rlpy=None,tolerance=0.002):
    """Read every existing source-matching root key, without scene changes."""
    if rlpy is None:
        import RLPy as rlpy
    path,motion,expected=_load_source(motion_path)
    _,_,_,fps,axes=_controls(rlpy)
    n=len(expected)
    counts=[int(axis.GetKeyCount()) for axis in axes]
    if counts!=[n,n,n]:
        raise ContractError(
            "Pelvis Position key counts %s do not match %d motion frames"%
            (counts,n))
    actual=[]
    for frame in range(n):
        t=rlpy.IndexedFrameTime(frame,fps)
        row=[float(read_float_value(axis,t)) for axis in axes]
        if not all(math.isfinite(v) for v in row):
            raise ContractError("nonfinite Pelvis Position controller sample")
        actual.append(row)
    # Readback may already be full source XYZ at 1x (new import). In that
    # case refuse further writes rather than reapplying the correction.
    world_error=max(
        abs(a-float(b)) for observed,raw in zip(actual,motion["trans"])
        for a,b in zip(observed,raw))
    already_world=world_error<=tolerance
    if already_world:
        current_scale=1.0
        max_error=world_error
    else:
        # Previous clips were written in inverse-Rest pelvis-local axes,
        # possibly later amplified 100x. Fit that *OLD* basis to identify
        # source-matching keys only. Do not use the fit as a new multiplier.
        denominator=sum(sum(v*v for v in row) for row in expected)
        if denominator<1e-10:
            raise ContractError("source Pelvis translation is too small to identify")
        current_scale=sum(
            sum(a*b for a,b in zip(observed,reference))
            for observed,reference in zip(actual,expected))/denominator
        if not math.isfinite(current_scale) or not 0.01<=current_scale<=200.:
            raise ContractError(
                "Current Pelvis Position does NOT match a safe positive "
                "scale of the source pelvis-local Dragon motion "
                "(fitted scale %.6g). Already remapped/unknown animation "
                "must not be modified."%current_scale)
        max_error=max(
            abs(a-current_scale*b)
            for observed,reference in zip(actual,expected)
            for a,b in zip(observed,reference))
        if max_error>tolerance:
            raise ContractError(
                "Current Pelvis Position is NOT the original pelvis-local "
                "Dragon Motion at any uniform scale "
                "(fitted %.6g; max per-axis mismatch %.6g; tolerance %.6g). "
                "Never remap unrelated or already axis-corrected animation."%
                (current_scale,max_error,tolerance))
    peak=max(range(n),key=lambda i:motion["trans"][i][2]-
             motion["trans"][0][2])
    z=float(motion["trans"][peak][2]-motion["trans"][0][2])
    delta=[float(actual[peak][i]-actual[0][i]) for i in range(3)]
    return {
        "status":("already_source_xyz_scale_1" if already_world else
                  "source_motion_position_matches_scale_1"
                  if abs(current_scale-1.0)<0.001 else
                  "source_motion_position_matches_scaled_pelvis_local"),
        "current_position_space":("armature" if already_world
                                  else "pelvis_local"),
        "motion_file":str(path),
        "frame_count":n,
        "peak_frame":peak,
        "peak_time_s":float(peak/24.0),
        "source_vertical_rise_m":z,
        "source_world_axis_peak_delta_xyz":[
            float(motion["trans"][peak][i]-motion["trans"][0][i])
            for i in range(3)],
        "original_pelvis_local_peak_delta_xyz":[
            float(expected[peak][i]-expected[0][i])
            for i in range(3)],
        "observed_stored_pelvis_local_delta_xyz":delta,
        "observed_stored_position_delta_xyz":delta,
        "detected_current_scale":float(current_scale),
        "key_counts_xyz":counts,
        "max_source_position_key_error":max_error,
        "note":("Already source XYZ at 1x; no Root Flight correction is "
                "needed." if already_world else
                "Legacy keys match inverse-Rest pelvis-local data; "
                "source +Z lift became negative local Z. Proposed correction "
                "writes full source XYZ at exactly 1x, NOT a Z-only sign flip. "
                "World-space trajectory still requires real iClone validation.")
    }


def _unused_path(folder,stem,suffix):
    for i in range(1,1000):
        candidate=folder/("%s_%03d%s"%(stem,i,suffix))
        if not candidate.exists():
            return candidate
    raise ContractError("No free output filename; do not overwrite projects")


def rescale_root_position(motion_path,output_parent,multiplier=1.0,rlpy=None,
                          position_space="armature"):
    """Backup, remap 3D Position BASIS and scalar, save a new project.

    Requires caller's explicit confirmation. If any post-backup step fails,
    attempt LoadProject of the backup and retain the backup regardless.
    """
    if rlpy is None:
        import RLPy as rlpy
    if position_space!="armature":
        raise ContractError(
            "Root correction requires source armature XYZ (+Z up), "
            "not the old inverse-Rest pelvis-local axes")
    if (type(multiplier) not in (int,float) or
        not math.isfinite(multiplier) or
        abs(float(multiplier)-1.0)>1e-9):
        raise ContractError(
            "Confirmed Root Position multiplier must be exactly 1.0; "
            "100x and all other gains are blocked")
    parent=pathlib.Path(output_parent)
    if not parent.is_dir():
        raise ContractError("Choose an existing output folder for project backup")
    inspected=inspect_root_scale(motion_path,rlpy=rlpy)
    if inspected["status"]=="already_source_xyz_scale_1":
        raise ContractError(
            "Root Position already uses source XYZ at 1x; nothing to repair")
    _,_,_,fps,axes=_controls(rlpy)
    path,motion,expected=_load_source(motion_path)
    # Replace every position sample with full source XYZ at absolute 1x.
    # Legacy 100x is recognized solely for read-only source matching, not
    # multiplied again; no other FK, root rotation or actor keys are touched.
    target=motion["trans"]
    signature=hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    backup=_unused_path(parent,"AnyMove_BeforeRootScale_"+signature,".iProject")
    output=_unused_path(parent,"AnyMove_RootScale_"+signature,".iProject")
    save=getattr(getattr(rlpy,"RFileIO",None),"SaveProject",None)
    load=getattr(getattr(rlpy,"RFileIO",None),"LoadProject",None)
    if not callable(save) or not callable(load):
        raise ContractError(
            "SDK must support SaveProject AND LoadProject for safe root repair")
    _require_sdk_ok(save(str(backup)),"backup original iProject")
    if not backup.is_file() or backup.stat().st_size==0:
        raise ContractError("Backup iProject not verified; no Position keys changed")
    modified=False
    try:
        # Re-check the source-matching keys after the backup.
        inspect_root_scale(path,rlpy=rlpy)
        for frame,row in enumerate(target):
            t=rlpy.IndexedFrameTime(frame,fps)
            for axis,value in zip(axes,row):
                modified=True
                _require_sdk_ok(
                    axis.SetValue(t,float(value)*float(multiplier)),
                    "write backed-up Pelvis Position")
        max_error=0.
        for frame,row in enumerate(target):
            t=rlpy.IndexedFrameTime(frame,fps)
            observed=[float(read_float_value(axis,t)) for axis in axes]
            if not all(math.isfinite(x) for x in observed):
                raise ContractError("rescaled Pelvis Position readback nonfinite")
            max_error=max(max_error,max(abs(a-b*multiplier)
                                        for a,b in zip(observed,row)))
        if max_error>0.002:
            raise ContractError(
                "rescaled Position readback error %.6g > 0.002"%max_error)
        _require_sdk_ok(save(str(output)),"save rescaled iProject")
        if not output.is_file() or output.stat().st_size==0:
            raise ContractError("rescaled project was not saved successfully")
        result={
            "status":"experimental_root_xyz_1x_keys_verified_project_saved",
            "backup_iProject":str(backup),
            "adjusted_iProject":str(output),
            "multiplier":float(multiplier),
            "previous_local_position_scale":inspected["detected_current_scale"],
            "position_space":position_space,
            "source_vertical_rise_m":inspected["source_vertical_rise_m"],
            "source_peak_frame":inspected["peak_frame"],
            "previous_stored_pelvis_delta_xyz":inspected[
                "observed_stored_pelvis_local_delta_xyz"],
            "expected_stored_position_delta_xyz":[
                float(target[inspected["peak_frame"]][axis]-
                      target[0][axis])*float(multiplier)
                for axis in range(3)],
            "readback_max_error":max_error,
            "quality":"NOT_CALIBRATED_OR_VISUALLY_VERIFIED",
            "note":"Only Pelvis Position XYZ was changed. FK rotations "
                   "untouched. Applied source XYZ at confirmed 1x. "
                   "World-axis mapping remains an experimental hypothesis "
                   "until measured in iClone. Reopen the backup if wrong."
        }
        receipt=_unused_path(parent,"AnyMove_RootScale_receipt_"+signature,".json")
        receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),
                           encoding="utf-8")
        result["receipt_path"]=str(receipt)
        return result
    except Exception as exc:
        if not modified:
            raise
        try:
            _require_sdk_ok(load(str(backup)),"restore original backup project")
            raise ContractError(
                "Root scale write failed (%s). Original project reloaded from %s; "
                "source and backup preserved."%(exc,backup)) from exc
        except ContractError as recovered:
            if "Original project reloaded" in str(recovered):
                raise
            raise ContractError(
                "Root scale failed (%s) and project RESTORE also failed (%s). "
                "Open the untouched backup manually: %s"%
                (exc,recovered,backup))
        except Exception as rollback:
            raise ContractError(
                "Root scale failed (%s) and restore raised %s. "
                "Open backup manually: %s"%(exc,rollback,backup))
