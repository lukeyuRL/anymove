"""Fail-closed iClone FK diagnostic writer with post-write evidence.

Inputs MUST be explicit iClone FK Layer XYZ rotation channel angles.
The original Blender/AnyMove motion.json is not compatible without mapping.
A successful SetValue is not proof of a key, and a key is not proof of
visible mesh motion: evaluate both separately. We never modify existing clips.
"""
from .motion_contract import validate_motion
from .rlpy_reader import _get_skeleton, selected_object, snapshot, _read_matrix_rows
from .bone_registry import build_bone_registry
from .p1_inspector import read_float_value
from .schema import ContractError


def _require_sdk_ok(status, operation):
    if status is None:
        raise ContractError("%s returned no SDK status" % operation)
    fn = getattr(status, "IsError", None)
    if callable(fn):
        if fn():
            raise ContractError("%s returned SDK error" % operation)
    elif status is not True:  # boolean success is only used by offline mocks.
        raise ContractError("%s returned unrecognized status" % operation)


def preflight(obj, motion, rlpy):
    skel = _get_skeleton(obj)
    current = snapshot(obj, rlpy=rlpy)
    validate_motion(motion, current)
    required = ("AddClip", "DeleteClip", "GetClipCount", "GetSkinBones")
    missing = [x for x in required if not callable(getattr(skel, x, None))]
    if missing:
        raise ContractError("missing SDK animation APIs: " + ", ".join(missing))
    clip_count = int(skel.GetClipCount())
    if clip_count:
        raise ContractError(
            "P1 safety gate: %d existing motion clip(s). "
            "Use Inspect existing clips to check them, or test a new scene "
            "with no clips. Existing clips are never overwritten." % clip_count)
    entries, _ = build_bone_registry(skel.GetSkinBones() or [])
    nodes = {entry["id"]: entry["node"] for entry in entries}
    names = {entry["id"]: entry["name"] for entry in entries}
    for bone_id in motion["tracks"]:
        if bone_id not in nodes:
            raise ContractError("bone missing from live skeleton: " + bone_id)
        fn = getattr(skel, "IsBoneAnimated", None)
        if callable(fn):
            try:
                animated = fn(nodes[bone_id])
            except (TypeError, RuntimeError):
                animated = None
            if animated is False:
                raise ContractError("non-animatable Skin Bone: " + names[bone_id])
    return skel, nodes, names


def _readback(motion, controls, fps, rlpy):
    """Read back controller key counts and values, without scene evaluation."""
    peak = max(range(motion["frame_count"]), key=lambda frame:
               max(abs(v) for values in motion["tracks"].values()
                   for v in values[frame]))
    frame_time = rlpy.IndexedFrameTime(peak, fps)
    evidence = {}
    any_nonzero = False
    for bone_id, axes in controls.items():
        track = motion["tracks"][bone_id]
        row = {"xyz_key_counts": [], "sample_frame": peak,
               "target_xyz_radians": track[peak], "sample_xyz_radians": [],
               "readback_warnings": []}
        for axis_index, ctrl in enumerate(axes):
            count = int(ctrl.GetKeyCount())  # SDK RControl.GetKeyCount()
            row["xyz_key_counts"].append(count)
            desired = [f[axis_index] for f in track]
            varies = max(desired) - min(desired) > 1e-6
            if varies:
                any_nonzero = True
                if count < 2:
                    raise ContractError(
                        "FK channel did not retain >=2 keys on bone %s axis %d" %
                        (bone_id, axis_index))
            try:
                measured = read_float_value(ctrl, frame_time)
                row["sample_xyz_radians"].append(measured)
                if abs(measured - desired[peak]) > 1e-3:
                    row["readback_warnings"].append(
                        "channel %d returned %s instead of %s" %
                        (axis_index, measured, desired[peak]))
            except Exception as exc:
                row["sample_xyz_radians"].append(None)
                row["readback_warnings"].append(
                    "readback channel %d failed: %s" % (axis_index, exc))
        evidence[bone_id] = row
    if not any_nonzero:
        raise ContractError("motion has no nonconstant rotation; cannot test animation")
    return peak, evidence


def _evaluate_visible_pose(rlpy, obj, node, fps, frame):
    """Measure WorldTransform at frame 0 and peak; leave timeline at peak.

    The result distinguishes visible bone transforms from FK controller data.
    IK or an unanimated helper bone can keep the world transform unchanged.
    """
    try:
        matrices = []
        for f in (0, frame):
            time = rlpy.IndexedFrameTime(f, fps)
            _require_sdk_ok(rlpy.RGlobal.SetTime(time), "SetTime")
            if callable(getattr(obj, "Update", None)):
                obj.Update()
            if callable(getattr(node, "Update", None)):
                node.Update()  # documented RINode refresh after keys
            mat = node.WorldTransform().Matrix()
            matrices.append([v for row in _read_matrix_rows(mat) for v in row])
        max_delta = max(abs(x - y) for x, y in zip(matrices[0], matrices[1]))
        return {"peak_frame": frame, "world_matrix_max_delta": max_delta,
                "bone_world_transform_changed": max_delta > 1e-5,
                "timeline_left_at_peak_frame": True, "error": None}
    except Exception as exc:
        # Failed preview must not make valid controller keys disappear.
        return {"peak_frame": frame, "bone_world_transform_changed": None,
                "timeline_left_at_peak_frame": False,
                "error": "%s: %s" % (type(exc).__name__, exc)}


def write_selected(motion, fps_confirmed=False, rlpy=None):
    if not fps_confirmed:
        raise ContractError("confirm iClone project FPS matches motion FPS")
    if rlpy is None:
        import RLPy as rlpy
    obj = selected_object(rlpy)
    skel, nodes, names = preflight(obj, motion, rlpy)
    project_fps = rlpy.RGlobal.GetFps()
    if rlpy.GetFrameIndex(rlpy.RGlobal.GetTime(), project_fps) != 0:
        raise ContractError("move timeline to frame 0 before writing")
    if not callable(getattr(project_fps, "ToDouble", None)):
        raise ContractError("RFps.ToDouble unavailable; refusing unverified FPS")
    if abs(float(project_fps.ToDouble()) - float(motion["fps"])) > 1e-5:
        raise ContractError("project FPS differs from motion FPS")
    clip = None
    try:
        clip = skel.AddClip(rlpy.IndexedFrameTime(0, project_fps))
        if clip is None:
            raise ContractError("iClone failed to create a motion clip")
        length = rlpy.IndexedFrameTime(motion["frame_count"], project_fps)
        _require_sdk_ok(clip.SetLength(length), "SetLength")
        channels = {}
        for bone_id in motion["tracks"]:
            transform = clip.GetControl("Layer", nodes[bone_id])
            if transform is None:
                raise ContractError("missing Layer transform control: " + names[bone_id])
            block = transform.GetDataBlock()
            if block is None:
                raise ContractError("missing FK data block: " + names[bone_id])
            axes = tuple(block.GetControl("Rotation/Rotation" + ax) for ax in "XYZ")
            if any(ctrl is None for ctrl in axes):
                raise ContractError("missing FK XYZ rotation control: " + names[bone_id])
            channels[bone_id] = axes
        for frame in range(motion["frame_count"]):
            time = rlpy.IndexedFrameTime(frame, project_fps)
            for bone_id, track in motion["tracks"].items():
                for ctrl, value in zip(channels[bone_id], track[frame]):
                    _require_sdk_ok(ctrl.SetValue(time, float(value)), "SetValue")
        if int(skel.GetClipCount()) < 1:
            raise ContractError("AddClip returned a clip but GetClipCount remains zero")
        peak, evidence = _readback(motion, channels, project_fps, rlpy)
        # From this point a keyed clip exists. UI refresh is best-effort,
        # NOT a reason to delete it or report that the motion disappeared.
        first_bone = max(motion["tracks"], key=lambda bone:
                         max(abs(a) for frame in motion["tracks"][bone]
                             for a in frame))
        preview = _evaluate_visible_pose(rlpy, obj, nodes[first_bone], project_fps, peak)
        changed = preview["bone_world_transform_changed"]
        status = ("fk_keys_verified_pose_changed" if changed is True else
                  "fk_keys_verified_pose_unchanged" if changed is False else
                  "fk_keys_verified_pose_unchecked")
        warnings = []
        if changed is False:
            warnings.append("FK keys exist but tested bone WorldTransform did not change; "
                            "check active IK override or try a deforming arm bone.")
        if preview["error"]:
            warnings.append("Could not evaluate visible pose: " + preview["error"])
        for entry in evidence.values():
            warnings.extend(entry["readback_warnings"])
        return {
            "status": status,
            "clip_count_after": int(skel.GetClipCount()),
            "frame_count": motion["frame_count"],
            "bone_count": len(motion["tracks"]),
            "target_bone_name": names[first_bone],
            "peak_frame": peak,
            "fk_readback": evidence,
            "pose_evaluation": preview,
            "warnings": warnings,
            "note": "Keys verified in FK controller, but actual on-screen animation must be confirmed in iClone."
        }
    except Exception as exc:
        if clip is not None:
            try:
                _require_sdk_ok(skel.DeleteClip(clip), "DeleteClip")
            except Exception as rollback_error:
                raise ContractError(
                    "P1 write failed (%s); ROLLBACK FAILED (%s), scene may be modified" %
                    (exc, rollback_error))
        raise
