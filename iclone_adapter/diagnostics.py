"""Read-only iClone diagnostic capture usable even if skeleton export fails.

This probe deliberately does not call FK, transform math or hierarchy builder.
It records errors *per bone* rather than failing on the first unexpected
C++/SWIG node, and is intended for reporting real SDK behavior.
"""
import json
import os
import tempfile
import traceback

from . import __version__


def _safe(fn):
    try:
        value = fn()
        return None if value is None else str(value)
    except Exception as exc:
        return "<%s: %s>" % (type(exc).__name__, exc)


def raw_skeleton_report(rlpy, failure=None, traceback_text=None):
    from . import bone_registry, rlpy_reader

    info = {
        "build": __version__,
        "loaded_modules": {
            "bone_registry": getattr(bone_registry, "__file__", ""),
            "rlpy_reader": getattr(rlpy_reader, "__file__", ""),
        },
        "failure": failure,
        "traceback": traceback_text,
        "object": None,
        "skin_bone_count": None,
        "bones": [],
        "issues": [],
    }
    try:
        selected = rlpy.RScene.GetSelectedObjects()
        selection = list(selected or [])
        info["selection_count"] = len(selection)
        if len(selection) != 1:
            info["issues"].append("Expected exactly one selected object")
            return info
        obj = selection[0]
        info["object"] = _safe(obj.GetName)
        skel = obj.GetSkeletonComponent()
        if skel is None:
            info["issues"].append("GetSkeletonComponent returned None")
            return info
        bones = list(skel.GetSkinBones() or [])
        info["skin_bone_count"] = len(bones)
        for index, bone in enumerate(bones):
            entry = {"index": index, "name": _safe(bone.GetName),
                     "native_scene_id": _safe(bone.GetID)}
            try:
                parent = bone.GetParent()
                entry["parent_name"] = _safe(parent.GetName) if parent is not None else None
                entry["parent_native_scene_id"] = (
                    _safe(parent.GetID) if parent is not None else None)
            except Exception as exc:
                entry["parent_error"] = "%s: %s" % (type(exc).__name__, exc)
            info["bones"].append(entry)
    except Exception as exc:
        info["issues"].append("Diagnostic collection failed: %s: %s" %
                              (type(exc).__name__, exc))
        info["diagnostic_traceback"] = traceback.format_exc()
    return info


def save_failure_report(target_path, rlpy, error):
    trace = traceback.format_exc()
    report = raw_skeleton_report(rlpy,
                                 failure="%s: %s" % (type(error).__name__, error),
                                 traceback_text=trace)
    target = target_path + ".debug.json"
    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except OSError:
        target = os.path.join(tempfile.gettempdir(), "anymove_iclone_export_debug.json")
        with open(target, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return target
