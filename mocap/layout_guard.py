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
import os
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
    angle_defaults={"FRONT_AZIMUTH_DEG":0.0,"SIDE_AZIMUTH_DEG":90.0}
    override=os.environ.get("MOCAP_CAMERA_AZIMUTHS")
    if override:
        values=override.split(",")
        if len(values)!=2:
            raise RuntimeError("MOCAP_CAMERA_AZIMUTHS must be two degree values")
        try:
            explicit_angles=tuple(float(v.strip())%360 for v in values)
        except ValueError as exc:
            raise RuntimeError("Invalid MOCAP_CAMERA_AZIMUTHS values") from exc
        if not all(math.isfinite(v) for v in explicit_angles):
            raise RuntimeError("MOCAP_CAMERA_AZIMUTHS must be finite")
        expected=dict(expected)
        expected.update(zip(angle_defaults,explicit_angles))
    if any(key not in expected for key in fields):
        raise RuntimeError("Missing camera field(s) in "+str(cfg))
    current={name:getattr(layout,name,None) for name in fields}
    current.update({name:getattr(layout,name,default) for name,default in angle_defaults.items()})
    differences=[]
    for name in fields+tuple(angle_defaults):
        actual=current[name]
        wanted=expected.get(name,angle_defaults.get(name))
        valid=(type(actual) in (int,float) and type(wanted) in (int,float)
               and math.isfinite(actual) and math.isfinite(wanted))
        if valid and name in angle_defaults:
            # 315 and -45 degrees describe the same world camera.
            valid=abs((actual-wanted+180)%360-180)<1e-6
        elif valid:
            valid=math.isclose(actual,wanted,rel_tol=1e-6,abs_tol=1e-9)
        if not valid:
            differences.append("%s expected %r, actual %r"%(name,wanted,actual))
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
