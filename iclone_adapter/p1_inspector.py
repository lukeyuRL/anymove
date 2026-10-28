"""Read-only iClone Motion Clip and FK controller inspection.

GetClipCount() tells us clips exist, NOT that their FK tracks are keyed.
GetKeyCount() is the supported RControl API for checking actual keyframes.
For iClone 8 a valid FK layer can still be overridden by IK.
"""
from .rlpy_reader import _get_skeleton, selected_object
from .bone_registry import build_bone_registry


def _ticks(time):
    if time is None:
        return None
    fn = getattr(time, "ToInt", None)
    return fn() if callable(fn) else str(time)


def _axes(clip, node):
    control = clip.GetControl("Layer", node)
    if control is None:
        return None
    block = control.GetDataBlock()
    if block is None:
        return None
    return tuple(block.GetControl("Rotation/Rotation" + axis) for axis in "XYZ")


def read_float_value(control, time):
    """SWIG float& out arguments usually return (RStatus, float).

    Never treat a bare SDK status as an animated value.
    """
    raw = control.GetValue(time, 0.0)
    if not isinstance(raw, (tuple, list)) or len(raw) < 2:
        raise ValueError("GetValue did not return status and float out-parameter")
    status, value = raw[0], raw[1]
    if callable(getattr(status, "IsError", None)) and status.IsError():
        raise ValueError("GetValue returned SDK error")
    return float(value)


def inspect_selected_animation(rlpy=None, max_clips=8, top_bones=12):
    """Non-mutating report for existing clips, including ones rejected by P1."""
    if rlpy is None:
        import RLPy as rlpy
    obj = selected_object(rlpy)
    skel = _get_skeleton(obj)
    registry, stats = build_bone_registry(skel.GetSkinBones() or [])
    count = int(skel.GetClipCount())
    report = {
        "object": str(obj.GetName()),
        "clip_count": count,
        "canonical_skin_bones": len(registry),
        "bone_identity": stats.get("identity_strategy"),
        "clips": [],
        "note": "FK keys present does not prove visible animation; active IK can override FK.",
    }
    for index in range(min(count, max_clips)):
        item = {"index": index, "keyed_bones": 0, "total_fk_keys": 0,
                "sample_keyed_bones": [], "errors": []}
        clip = skel.GetClip(index)
        if clip is None:
            item["errors"].append("GetClip returned None")
            report["clips"].append(item)
            continue
        for name in ("GetLength", "GetClipLength", "GetLastKeyTime", "GetStartOffset"):
            try:
                fn = getattr(clip, name, None)
                item[name] = _ticks(fn()) if callable(fn) else None
            except Exception as exc:
                item["errors"].append("%s: %s" % (name, exc))
        keyed = []
        for entry in registry:
            try:
                axes = _axes(clip, entry["node"])
                if axes is None:
                    continue
                nums = [int(axis.GetKeyCount()) if axis is not None else 0
                        for axis in axes]
                if any(nums):
                    item["keyed_bones"] += 1
                    item["total_fk_keys"] += sum(nums)
                    keyed.append({"name": entry["name"], "path": entry["path"],
                                  "xyz_key_counts": nums})
            except Exception as exc:
                if len(item["errors"]) < 20:
                    item["errors"].append("%s: %s" % (entry["name"], exc))
        keyed.sort(key=lambda entry: sum(entry["xyz_key_counts"]), reverse=True)
        item["sample_keyed_bones"] = keyed[:top_bones]
        item["fk_keys_present"] = item["total_fk_keys"] > 0
        report["clips"].append(item)
    report["clips_omitted"] = max(0, count - max_clips)
    return report
