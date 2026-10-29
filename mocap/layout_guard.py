"""Fail closed on wrong creature camera layout before fitting any joints.

The default in mocap.layout is Wolf (420/860, 1.9/720 m per pixel).
The Dragon camera is 640/640 with M_PER_PX=0.006875. If users forget
MOCAP_CREATURE=assets/dragon before the Solver process starts, a full
124-frame clip can fit to an incorrect 3D projection and produce a
visibly tilted dragon in both Blender and iClone.

Pure stdlib; no torch or numpy is imported into regression tests.
"""
import json
import math
import pathlib


def verify_layout_for_rig(rig_info_path, layout):
    """Read creature.json adjacent to the original rigged/rig_info.json."""
    path=pathlib.Path(rig_info_path).resolve()
    if path.name!="rig_info.json" or path.parent.name!="rigged":
        return {"status":"rig_config_not_inferred","rig_info_path":str(path)}
    cfg=path.parent.parent/"creature.json"
    if not cfg.is_file():
        return {"status":"creature_config_missing","rig_info_path":str(path)}
    try:
        with open(str(cfg),encoding="utf-8") as f:
            creature=json.load(f)
    except (OSError,ValueError) as exc:
        raise RuntimeError("Could not load Creature Layout: "+str(exc))
    expected=creature.get("layout")
    if not isinstance(expected,dict):
        raise RuntimeError("Missing Creature Layout in "+str(cfg))
    fields=("W_FRONT","W_SIDE","M_PER_PX","GROUND_V")
    if any(key not in expected for key in fields):
        raise RuntimeError("Missing camera field(s) in "+str(cfg))
    current={name:getattr(layout,name,None) for name in fields}
    differences=[]
    for name in fields:
        actual=current[name]
        wanted=expected[name]
        if (type(actual) not in (int,float)
                or type(wanted) not in (int,float)
                or not math.isfinite(actual)
                or not math.isfinite(wanted)
                or not math.isclose(actual,wanted,
                                    rel_tol=1e-6,abs_tol=1e-9)):
            differences.append("%s expected %r, actual %r"%
                               (name,wanted,actual))
    if differences:
        creature_name=str(creature.get("name",cfg.parent.name))
        raise RuntimeError(
            "CAMERA_LAYOUT_MISMATCH: Solver's active camera Layout is "
            "incompatible with the %s Rig (%s). "
            "This can produce a tilted/incorrect motion.json. "
            "Set MOCAP_CREATURE=assets/%s before launching the Solver "
            "and regenerate motion.json in a NEW output path. "
            "On Windows PowerShell: $env:MOCAP_CREATURE='assets/%s'. "
            "Do not alter iClone FK/Rest Basis."%
            (creature_name,"; ".join(differences),
             creature_name,creature_name))
    return {
        "status":"camera_layout_verified_for_rig",
        "creature":creature.get("name",cfg.parent.name),
        "active_layout":current,
        "rig_info_path":str(path)
    }
