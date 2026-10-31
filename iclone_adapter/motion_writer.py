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
from .scene_guard import preflight_clean_dragon


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
    required = ("GetClip", "AddClip", "DeleteClip", "GetClipCount", "GetSkinBones")
    missing = [x for x in required if not callable(getattr(skel, x, None))]
    if missing:
        raise ContractError("missing SDK animation APIs: " + ", ".join(missing))
    # Normal iClone characters can have one default Motion Clip.
    # Reuse one unkeyed clip, or append after the current clips.
    entries, _ = build_bone_registry(skel.GetSkinBones() or [])
    nodes = {entry["id"]: entry["node"] for entry in entries
             if entry["id"] in motion["tracks"]}
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
        positions=motion.get("position_tracks") or {}
        any_nonzero=any(
            any(max(row[axis] for row in track)-
                min(row[axis] for row in track)>1e-6
                for axis in range(3))
            for track in positions.values())
        if not any_nonzero:
            raise ContractError("motion has no varying FK or root Position tracks")
        # For a translation-only action, inspect the frame with maximum root
        # travel instead of comparing the unchanged rotation at frame zero.
        peak=max(range(motion["frame_count"]),key=lambda f:
                 max(abs(track[f][axis]-track[0][axis])
                     for track in positions.values() for axis in range(3)))
    return peak, evidence



def _ticks(value):
    return value if isinstance(value, int) else int(value.ToInt())


def _scene_time(clip, local):
    convert = getattr(clip, "ClipTimeToSceneTime", None)
    if not callable(convert):
        raise ContractError("RIClip.ClipTimeToSceneTime is required for safe placement")
    return convert(local)


def _channels_for_clip(clip, nodes, names):
    channels = {}
    for bone_id, node in nodes.items():
        transform = clip.GetControl("Layer", node)
        if transform is None:
            raise ContractError("missing Layer transform control: " + names[bone_id])
        block = transform.GetDataBlock()
        if block is None:
            raise ContractError("missing FK data block: " + names[bone_id])
        axes = tuple(block.GetControl("Rotation/Rotation" + ax) for ax in "XYZ")
        if any(ctrl is None for ctrl in axes):
            raise ContractError("missing FK XYZ rotation control: " + names[bone_id])
        channels[bone_id] = axes
    return channels


def _position_channels_for_clip(clip,nodes,names,position_tracks):
    """Return Pelvis Layer Position/PositionXYZ, refusing partial coverage."""
    found={}
    for bone_id in (position_tracks or {}):
        if bone_id not in nodes:
            raise ContractError("Root Motion target is not a live Skin Bone: "+bone_id)
        root=clip.GetControl("Layer",nodes[bone_id])
        if root is None:
            raise ContractError("Root Motion pelvis Layer control unavailable")
        block=root.GetDataBlock()
        if block is None:
            raise ContractError("Root Motion pelvis Layer DataBlock unavailable")
        channels=tuple(block.GetControl("Position/Position"+axis) for axis in "XYZ")
        if any(c is None for c in channels):
            raise ContractError(
                "iClone SDK pelvis Position XYZ Layer controls unavailable; "
                "experimental Root Translation blocked without writing keys")
        found[bone_id]=channels
    return found


def _readback_positions(position_tracks,controls,fps,rlpy):
    evidence={}
    for bone_id,track in position_tracks.items():
        axes=controls[bone_id]
        sample_frames=sorted(set((0,len(track)//2,len(track)-1)))
        item={"xyz_key_counts":[],"samples":[],"max_abs_error":0.}
        for axis,ctrl in enumerate(axes):
            count=int(ctrl.GetKeyCount())
            item["xyz_key_counts"].append(count)
            if count<len(track):
                raise ContractError("Root Position channel has fewer keys than frames")
        for frame in sample_frames:
            t=rlpy.IndexedFrameTime(frame,fps)
            got=[read_float_value(ctrl,t) for ctrl in axes]
            expected=track[frame]
            err=max(abs(x-y) for x,y in zip(got,expected))
            item["max_abs_error"]=max(err,item["max_abs_error"])
            item["samples"].append({"frame":frame,"written_xyz":expected,
                                    "readback_xyz":got})
        if item["max_abs_error"]>1e-3:
            raise ContractError("Root Position sample readback exceeded 1e-3")
        evidence[bone_id]=item
    return evidence


def _choose_clip(skel, nodes, names, fps, rlpy, strict_clean=False):
    """Prefer a default unkeyed clip; otherwise create non-overlapping motion."""
    count = int(skel.GetClipCount())
    zero = rlpy.IndexedFrameTime(0, fps)
    one_frame = rlpy.IndexedFrameTime(1, fps)
    clips = [skel.GetClip(i) for i in range(count)]
    if any(clip is None for clip in clips):
        raise ContractError("GetClip returned None for an existing index")
    if count == 1:
        existing = clips[0]
        try:
            if _ticks(_scene_time(existing, zero)) == _ticks(zero):
                ctrls = _channels_for_clip(existing, nodes, names)
                if all(int(ctrl.GetKeyCount()) == 0
                       for axes in ctrls.values() for ctrl in axes):
                    return existing, ctrls, "reuse_default", False, count
        except (AttributeError, TypeError, RuntimeError, ContractError):
            pass
    if strict_clean and count > 0:
        raise ContractError("guard: cannot append experimental Dragon animation to old Motion; use NEW project")
    # No overlap: append AFTER the rightmost existing Clip, not at scene frame 0.
    latest = zero
    for existing in clips:
        try:
            start = _scene_time(existing, zero)
            end = start + existing.GetClipLength()
            last_key = _scene_time(existing, existing.GetLength())
        except Exception as exc:
            raise ContractError("cannot measure existing Clip's range: %s" % exc)
        for candidate in (end, last_key):
            if _ticks(candidate) > _ticks(latest):
                latest = candidate
    start = zero if count == 0 else latest + one_frame
    new_clip = skel.AddClip(start)
    if new_clip is None:
        raise ContractError("AddClip failed to create non-overlapping Motion")
    return new_clip, None, ("create_new" if count == 0 else "append_new"), True, count


def _extend_playback_range(rlpy, clip, fps, frames):
    try:
        scene_end = _scene_time(clip, rlpy.IndexedFrameTime(frames, fps))
        if _ticks(scene_end) > _ticks(rlpy.RGlobal.GetEndTime()):
            _require_sdk_ok(rlpy.RGlobal.SetEndTime(scene_end), "SetEndTime")
        return {"end_ticks": _ticks(scene_end), "valid": True}
    except Exception as exc:
        return {"valid": False, "error": "%s: %s" %
                (type(exc).__name__, exc)}


def _evaluate_visible_pose(rlpy, obj, node, fps, frame, clip):
    """Measure WorldTransform at frame 0 and peak; leave timeline at peak.

    The result distinguishes visible bone transforms from FK controller data.
    IK or an unanimated helper bone can keep the world transform unchanged.
    """
    try:
        matrices = []
        for f in (0, frame):
            time = _scene_time(clip, rlpy.IndexedFrameTime(f, fps))
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
    guarded = str(motion.get("calibration_status","")).startswith("unverified_")
    clean_guard = preflight_clean_dragon(obj,skel,motion,rlpy) if guarded else None
    clip = None
    created = False
    strategy = None
    previous_count = None
    old_length = None
    length_extended = False
    inserted_keys = []
    try:
        clip, channels, strategy, created, previous_count = _choose_clip(
            skel, nodes, names, project_fps, rlpy, strict_clean=guarded)
        length = rlpy.IndexedFrameTime(motion["frame_count"], project_fps)
        old_length = clip.GetLength() if not created else None
        if created or max(_ticks(old_length), _ticks(clip.GetClipLength())) < _ticks(length):
            _require_sdk_ok(clip.SetLength(length), "SetLength")
            length_extended = True
        if channels is None:
            channels = _channels_for_clip(clip, nodes, names)
        # Probe EVERY position channel BEFORE modifying even the first FK
        # value. An unsupported Pelvis root fails here and rolls back any
        # just-created empty test Clip, not a half-written animation.
        position_tracks = motion.get("position_tracks") or {}
        position_channels = _position_channels_for_clip(
            clip,nodes,names,position_tracks)
        if created and any(int(ctrl.GetKeyCount()) for axes in
                           list(channels.values())+list(position_channels.values())
                           for ctrl in axes):
            raise ContractError("new clip unexpectedly has pre-existing FK/Position keys")
        for frame in range(motion["frame_count"]):
            time = rlpy.IndexedFrameTime(frame, project_fps)
            for bone_id, track in motion["tracks"].items():
                for ctrl, value in zip(channels[bone_id], track[frame]):
                    _require_sdk_ok(ctrl.SetValue(time, float(value)), "SetValue")
                    inserted_keys.append((ctrl, time))
            for bone_id,track in position_tracks.items():
                for ctrl,value in zip(position_channels[bone_id],track[frame]):
                    _require_sdk_ok(ctrl.SetValue(time,float(value)), "Root Position SetValue")
                    inserted_keys.append((ctrl,time))
        if int(skel.GetClipCount()) != previous_count + int(created):
            raise ContractError("GetClipCount changed unexpectedly")
        peak, evidence = _readback(motion, channels, project_fps, rlpy)
        root_position_readback = _readback_positions(
            position_tracks,position_channels,project_fps,rlpy)
        # From this point a keyed clip exists. UI refresh is best-effort,
        # NOT a reason to delete it or report that the motion disappeared.
        first_bone = max(motion["tracks"], key=lambda bone:
                         max(abs(a) for frame in motion["tracks"][bone]
                             for a in frame))
        timeline = _extend_playback_range(rlpy, clip, project_fps, motion["frame_count"])
        preview = _evaluate_visible_pose(rlpy, obj, nodes[first_bone],
                                         project_fps, peak, clip)
        changed = preview["bone_world_transform_changed"]
        status = ("fk_keys_verified_pose_changed" if changed is True else
                  "fk_keys_verified_pose_unchanged" if changed is False else
                  "fk_keys_verified_pose_unchecked")
        warnings = []
        if motion.get("root_motion"):
            warnings.append("EXPERIMENTAL Root Motion: root Layer convention and "
                            "Position unit multiplier are not calibrated; visually "
                            "verify matching Blender frames in a clean project.")
        if changed is False:
            warnings.append("FK keys exist but tested bone WorldTransform did not change; "
                            "check active IK override or try a deforming arm bone.")
        if preview["error"]:
            warnings.append("Could not evaluate visible pose: " + preview["error"])
        for entry in evidence.values():
            warnings.extend(entry["readback_warnings"])
        if not timeline["valid"]:
            warnings.append("Timeline may not reach the test Clip: " +
                            timeline.get("error", "unknown"))
        return {
            "status": status,
            "scene_guard":clean_guard,
            "clip_strategy": strategy,
            "clip_count_before": previous_count,
            "clip_start_ticks": _ticks(_scene_time(
                clip, rlpy.IndexedFrameTime(0, project_fps))),
            "timeline": timeline,
            "clip_count_after": int(skel.GetClipCount()),
            "frame_count": motion["frame_count"],
            "bone_count": len(motion["tracks"]),
            "target_bone_name": names[first_bone],
            "peak_frame": peak,
            "fk_readback": evidence,
            "root_motion": motion.get("root_motion"),
            "root_position_readback": root_position_readback,
            "pose_evaluation": preview,
            "warnings": warnings,
            "note": "Keys verified in FK controller, but actual on-screen animation must be confirmed in iClone."
        }
    except Exception as exc:
        if clip is not None:
            try:
                if created:
                    _require_sdk_ok(skel.DeleteClip(clip), "DeleteClip")
                else:
                    # Never delete or overwrite the user's default clip.
                    # All target FK channels had zero keys before writing.
                    for ctrl, time in reversed(inserted_keys):
                        _require_sdk_ok(ctrl.RemoveKey(time), "RemoveKey")
                    if length_extended:
                        _require_sdk_ok(clip.SetLength(old_length), "restore clip length")
            except Exception as rollback_error:
                raise ContractError(
                    "P1 failed (%s); ROLLBACK FAILED (%s), scene may contain keys" %
                    (exc, rollback_error))
        raise
