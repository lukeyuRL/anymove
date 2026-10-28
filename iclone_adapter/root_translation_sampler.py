"""Read-only manual-frame iClone pelvis Root Translation recorder.

Why manual sampling? Prior iClone RINode WorldTransform results appeared static
under immediate scripted SetTime calls, even though manually switching frames
and calling ConvertToOriginalBoneAxis returned changing rotations. Here the
user selects a frame in iClone and captures a snapshot without calling SetTime.

This snapshot deliberately separates:
- the evaluated pelvis original-axis-local translation
- static/dynamic RINode LocalTransform and WorldTransform translations
- the stored Pelvis Layer Position XYZ controller samples (if available)

No scene writes, clip modifications, or timeline changes.
"""
import math

from .rlpy_reader import selected_object, _get_skeleton
from .p1_inspector import read_float_value
from .schema import ContractError

VERSION="anymove.iclone.root_translation_samples/1"


def _xyz(v):
    try:
        row=[float(v.x),float(v.y),float(v.z)]
    except Exception as exc:
        raise ContractError("expected RVector3 for Pelvis position: "+str(exc))
    if not all(math.isfinite(v) for v in row):
        raise ContractError("nonfinite Pelvis position")
    return row


def capture_root_sample(rlpy=None):
    """Capture ONE manually selected iClone frame. Never call SetTime."""
    if rlpy is None:
        import RLPy as rlpy
    obj=selected_object(rlpy)
    skel=_get_skeleton(obj)
    frames=skel.ConvertToOriginalBoneAxis()
    if frames is None:
        raise ContractError("ConvertToOriginalBoneAxis returned no Bone transforms")
    pelvis=[]
    for pair in frames:
        if not isinstance(pair,(tuple,list)) or len(pair)!=2:
            raise ContractError("unexpected ConvertToOriginalBoneAxis pair format")
        node,tr=pair
        if node.GetName()=="pelvis":
            pelvis.append((node,tr))
    if len(pelvis)!=1:
        raise ContractError("expected unique pelvis, got %d"%len(pelvis))
    node,tr=pelvis[0]
    fps=rlpy.RGlobal.GetFps()
    source_time=rlpy.RGlobal.GetTime()
    frame=int(rlpy.GetFrameIndex(source_time,fps))
    if frame<0:
        raise ContractError("negative iClone frame; move to a nonnegative Clip frame")
    p={
        "frame":frame,
        "original_axis_local_xyz":_xyz(tr.T()),
        "node_local_xyz":_xyz(node.LocalTransform().T()),
        "node_world_xyz":_xyz(node.WorldTransform().T()),
        "pelvis_name":node.GetName(),
        "clip_count":int(skel.GetClipCount()),
    }
    if hasattr(obj,"WorldTransform") and callable(obj.WorldTransform):
        p["actor_world_xyz"]=_xyz(obj.WorldTransform().T())
    else:
        p["actor_world_xyz"]=None
    # Stored channel values are not the same as evaluated translation.
    channels=[]
    if p["clip_count"]==1:
        clip=skel.GetClip(0)
        layer=clip.GetControl("Layer",node) if clip is not None else None
        block=layer.GetDataBlock() if layer is not None else None
        if block is not None:
            for axis in "XYZ":
                ctrl=block.GetControl("Position/Position"+axis)
                if ctrl is None:
                    channels.append(None)
                else:
                    try:
                        channels.append(float(read_float_value(
                            ctrl,rlpy.IndexedFrameTime(frame,fps))))
                    except Exception:
                        channels.append(None)
    p["stored_pelvis_position_xyz"]=(
        channels if len(channels)==3 else None)
    return {
        "schema_version":VERSION,
        "object_name":str(obj.GetName()),
        "fps":float(fps.ToDouble()),
        "sample":p,
        "warning":"Current-frame only; no SetTime. Captured Bone Position and controller values are independent measurements."
    }


def save_samples(path,records):
    """Create a validated JSON from a UI session's frame-indexed samples."""
    import json
    if not records:
        raise ContractError("capture at least Frame 0 and two other frames")
    ordered=sorted(records.values(),key=lambda r:r["sample"]["frame"])
    first=ordered[0]
    if len(ordered)<3 or ordered[0]["sample"]["frame"]!=0:
        raise ContractError("need Frame 0 and at least two other captured frames")
    if any(r["object_name"]!=first["object_name"] or
           r["fps"]!=first["fps"] or
           r["schema_version"]!=VERSION
           for r in ordered):
        raise ContractError("mixed Dragon objects/FPS in root capture session")
    result={
        "schema_version":VERSION,
        "object_name":first["object_name"],
        "fps":first["fps"],
        "samples":[r["sample"] for r in ordered],
        "source":"manually_selected_iClone_frames",
        "warning":"Only current-frame original-axis-local Pelvis positions are evaluated; never infer scale from a single static frame."
    }
    with open(path,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    return result
