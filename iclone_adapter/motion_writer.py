"""Fail-closed diagnostic writer for a new iClone FK Motion Clip.

Accepts explicit iClone Layer XYZ rotation channel angles ONLY.
AnyMove's Blender Euler motion.json requires calibration/retargeting first.
"""
from .motion_contract import validate_motion
from .rlpy_reader import _get_skeleton, selected_object, snapshot
from .bone_registry import build_bone_registry
from .schema import ContractError


def _require_sdk_ok(status, operation):
    """RStatus.IsError handles failure codes beyond the single Failure enum."""
    if status is None:
        raise ContractError("%s returned no SDK status" % operation)
    if callable(getattr(status, "IsError", None)):
        if status.IsError():
            raise ContractError("%s returned SDK error" % operation)
    elif status is not True:
        # bool True is used by the offline mock, not expected from real RLPy.
        raise ContractError("%s returned an unrecognized status" % operation)


def preflight(obj, motion, rlpy):
    skel = _get_skeleton(obj)
    current = snapshot(obj, rlpy=rlpy)
    validate_motion(motion, current)
    required = ("AddClip", "DeleteClip", "GetClipCount", "GetSkinBones")
    missing = [x for x in required if not callable(getattr(skel, x, None))]
    if missing:
        raise ContractError("missing SDK animation APIs: " + ", ".join(missing))
    if int(skel.GetClipCount()) != 0:
        raise ContractError("P1 safety gate: object must have zero motion clips")
    entries, _ = build_bone_registry(skel.GetSkinBones() or [])
    nodes = {entry["id"]: entry["node"] for entry in entries}
    for bone_id in motion["tracks"]:
        if bone_id not in nodes:
            raise ContractError("bone missing from live skeleton: " + bone_id)
        fn = getattr(skel, "IsBoneAnimated", None)
        if callable(fn):
            try:
                animatable = fn(nodes[bone_id])
            except (TypeError, RuntimeError):
                animatable = None
            if animatable is False:
                raise ContractError("non-animatable Skin Bone: " + bone_id)
    return skel, nodes


def write_selected(motion, fps_confirmed=False, rlpy=None):
    """Write a fresh clip at frame 0, requiring an empty clip track.

    On failure, DeleteClip is attempted. If rollback fails a high-severity
    exception states that the scene may contain partial keys.
    """
    if not fps_confirmed:
        raise ContractError("confirm that iClone project FPS equals motion FPS")
    if rlpy is None:
        import RLPy as rlpy
    obj = selected_object(rlpy)
    skel, nodes = preflight(obj, motion, rlpy)
    if rlpy.GetFrameIndex(rlpy.RGlobal.GetTime(), rlpy.RGlobal.GetFps()) != 0:
        raise ContractError("move timeline to frame 0 before writing")
    # RFps.ToDouble() is documented by the supplied Legacy RLPy binding.
    # A user confirmation does not replace a machine-checked FPS match.
    project_fps = rlpy.RGlobal.GetFps()
    if not callable(getattr(project_fps, "ToDouble", None)):
        raise ContractError("SDK RFps.ToDouble unavailable; refusing unverified FPS")
    actual_fps = float(project_fps.ToDouble())
    if abs(actual_fps - float(motion["fps"])) > 1e-5:
        raise ContractError("project FPS %s differs from motion FPS %s" %
                            (actual_fps, motion["fps"]))
    clip = None
    try:
        clip = skel.AddClip(rlpy.IndexedFrameTime(0, project_fps))
        if clip is None:
            raise ContractError("iClone failed to create a motion clip")
        # RIClip.SetLength(RTime) is documented. Establish the full clip
        # duration before creating keys, rather than assuming a default range.
        length = rlpy.IndexedFrameTime(motion["frame_count"], project_fps)
        _require_sdk_ok(clip.SetLength(length), "SetLength")
        channels = {}
        for bone_id in motion["tracks"]:
            transform = clip.GetControl("Layer", nodes[bone_id])
            if transform is None:
                raise ContractError("missing Layer transform control: " + bone_id)
            block = transform.GetDataBlock()
            if block is None:
                raise ContractError("missing FK data block: " + bone_id)
            axes = [block.GetControl("Rotation/Rotation" + axis) for axis in "XYZ"]
            if any(axis is None for axis in axes):
                raise ContractError("missing FK XYZ rotation control: " + bone_id)
            channels[bone_id] = axes

        for frame in range(motion["frame_count"]):
            t = rlpy.IndexedFrameTime(frame, project_fps)
            for bone_id, track in motion["tracks"].items():
                for ctrl, radians in zip(channels[bone_id], track[frame]):
                    _require_sdk_ok(ctrl.SetValue(t, float(radians)), "SetValue")
        return {"status": "keys_written_unverified",
                "frame_count": motion["frame_count"],
                "bone_count": len(motion["tracks"]),
                "warning": "verify clip duration, interpolation, IK overrides and readback in iClone"}
    except Exception as exc:
        if clip is not None:
            try:
                _require_sdk_ok(skel.DeleteClip(clip), "DeleteClip")
            except Exception as rollback_error:
                raise ContractError("P1 write failed (%s); ROLLBACK FAILED (%s), scene may be modified" %
                                    (exc, rollback_error))
        raise
