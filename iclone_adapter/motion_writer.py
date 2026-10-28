"""Fail-closed diagnostic writer for a new iClone FK Motion Clip.

Accepts explicit iClone Layer XYZ rotation channel angles ONLY.
AnyMove's Blender Euler motion.json requires calibration/retargeting first.
"""
from .motion_contract import validate_motion
from .rlpy_reader import _get_skeleton, selected_object, snapshot
from .schema import ContractError


def preflight(obj, motion):
    skel = _get_skeleton(obj)
    current = snapshot(obj)
    validate_motion(motion, current)
    required = ("AddClip", "DeleteClip", "GetClipCount", "GetSkinBones")
    missing = [x for x in required if not callable(getattr(skel, x, None))]
    if missing:
        raise ContractError("missing SDK animation APIs: " + ", ".join(missing))
    if int(skel.GetClipCount()) != 0:
        raise ContractError("P1 safety gate: object must have zero motion clips")
    nodes = {str(b.GetID()): b for b in skel.GetSkinBones()}
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
    skel, nodes = preflight(obj, motion)
    if rlpy.GetFrameIndex(rlpy.RGlobal.GetTime(), rlpy.RGlobal.GetFps()) != 0:
        raise ContractError("move timeline to frame 0 before writing")
    # iClone 8 uses 6,000 ticks/second. Validate project FPS if RTime
    # exposes ToInt(); this also detects accidental 24-vs-30 FPS exports.
    project_fps = rlpy.RGlobal.GetFps()
    one = rlpy.IndexedFrameTime(1, project_fps)
    zero = rlpy.IndexedFrameTime(0, project_fps)
    if callable(getattr(one, "ToInt", None)) and callable(getattr(zero, "ToInt", None)):
        ticks = one.ToInt() - zero.ToInt()
        if abs(ticks - 6000.0 / motion["fps"]) > 0.51:
            raise ContractError("project FPS differs from motion FPS")
    clip = None
    try:
        clip = skel.AddClip(rlpy.IndexedFrameTime(0, rlpy.RGlobal.GetFps()))
        if clip is None:
            raise ContractError("iClone failed to create a motion clip")
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

        project_fps = rlpy.RGlobal.GetFps()
        for frame in range(motion["frame_count"]):
            t = rlpy.IndexedFrameTime(frame, project_fps)
            for bone_id, track in motion["tracks"].items():
                for ctrl, radians in zip(channels[bone_id], track[frame]):
                    result = ctrl.SetValue(t, float(radians))
                    if result == rlpy.RStatus.Failure:
                        raise ContractError("SDK failed writing FK key")
        return {"status": "keys_written_unverified",
                "frame_count": motion["frame_count"],
                "bone_count": len(motion["tracks"]),
                "warning": "verify clip duration, interpolation, IK overrides and readback in iClone"}
    except Exception as exc:
        if clip is not None:
            try:
                result = skel.DeleteClip(clip)
                if result == rlpy.RStatus.Failure:
                    raise ContractError("DeleteClip returned failure")
            except Exception as rollback_error:
                raise ContractError("P1 write failed (%s); ROLLBACK FAILED (%s), scene may be modified" %
                                    (exc, rollback_error))
        raise
