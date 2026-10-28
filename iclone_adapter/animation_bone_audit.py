"""Read-only diagnostic: Skin Bones vs Motion Bones and their FK Layer keys.

Official iClone documentation distinguishes GetSkinBones from GetMotionBones:
animation is authored through Motion Bones. The attached iClone 8 Legacy RLPy
wrapper exposes GetAllAnimationBone() instead of GetMotionBones(); probe BOTH
when present and report identities without silently assuming equivalence.
No SetValue, AddClip, SetTime or scene mutations in this diagnostic.
"""
import math

from .bone_registry import build_bone_registry
from .motion_contract import validate_motion
from .p1_inspector import _axes, read_float_value
from .rlpy_reader import _get_skeleton, selected_object, snapshot


def _safe_node(node):
    d={}
    for label, method in (("name","GetName"),("native_scene_id","GetID")):
        try:
            d[label]=str(getattr(node,method)())
        except Exception as exc:
            d[label+"_error"]=type(exc).__name__+": "+str(exc)
    try:
        parent=node.GetParent()
        d["parent_name"]=str(parent.GetName()) if parent is not None else None
    except Exception as exc:
        d["parent_error"]=type(exc).__name__+": "+str(exc)
    return d


def _read_track(clip, node, frames, fps, expected):
    out={"available":False, "key_counts_xyz":None,
         "sampled_values":[], "max_abs_error_radians":None,
         "readback_supported":False, "errors":[]}
    try:
        axes=_axes(clip,node)
        if axes is None:
            out["errors"].append("no Layer FK controls")
            return out
        out["available"]=True
        out["key_counts_xyz"]=[
            int(ctrl.GetKeyCount()) if ctrl is not None else 0
            for ctrl in axes]
        errors=[]
        readback_count=0
        for frame in frames:
            vals=[]
            for ai,ctrl in enumerate(axes):
                if ctrl is None:
                    vals.append(None)
                    continue
                try:
                    v=read_float_value(ctrl, __import__("RLPy").IndexedFrameTime(frame,fps)) if fps is None else None
                    # The branch above is not used by supported callers.
                except Exception:
                    v=None
                vals.append(v)
            out["sampled_values"].append({"frame":frame,"xyz":vals})
        # Use explicit clock factory in the main call (no RLPy import needed
        # for mocks and no dependency on a specific SWIG module name).
    except Exception as exc:
        out["errors"].append("%s: %s"%(type(exc).__name__,exc))
    return out


def _inspect_channel(clip, node, frames, project_fps, indexed_frame_time, track):
    out={"available":False, "key_counts_xyz":None, "sampled_values":[],
         "max_abs_error_radians":None, "readback_supported":False,
         "errors":[]}
    try:
        ctrls=_axes(clip,node)
        if ctrls is None:
            out["errors"].append("no Layer control")
            return out
        out["available"]=True
        counts=[int(c.GetKeyCount()) if c is not None else 0 for c in ctrls]
        out["key_counts_xyz"]=counts
        differences=[]
        seen=0
        for frame in frames:
            t=indexed_frame_time(frame,project_fps)
            row=[]
            for ai,c in enumerate(ctrls):
                if c is None:
                    row.append(None)
                    continue
                try:
                    v=read_float_value(c,t)
                    row.append(v)
                    differences.append(abs(v-float(track[frame][ai])))
                    seen+=1
                except Exception as exc:
                    row.append(None)
                    if len(out["errors"])<5:
                        out["errors"].append("frame %d axis %d: %s"%
                                             (frame,ai,exc))
            out["sampled_values"].append({"frame":frame,"xyz_radians":row})
        out["readback_supported"]=seen==3*len(frames)
        out["max_abs_error_radians"]=max(differences) if differences else None
    except Exception as exc:
        out["errors"].append("%s: %s"%(type(exc).__name__,exc))
    return out


def _probe_set(skel, method):
    fn=getattr(skel,method,None)
    if not callable(fn):
        return None, {"method":method,"available":False,
                      "error":"not exposed by this RLPy SDK"}
    try:
        nodes=list(fn() or [])
        return nodes, {
            "method":method,"available":True,
            "node_count":len(nodes),
            "unique_name_count":len(set(_safe_node(n).get("name") for n in nodes)),
            "first_nodes":[_safe_node(n) for n in nodes[:8]],
        }
    except Exception as exc:
        return None, {"method":method,"available":True,
                      "error":"%s: %s"%(type(exc).__name__,exc)}


def audit_animation_bones(motion, clip_index=None, rlpy=None):
    """Inspect candidate animation node classes, without modifying a clip."""
    if rlpy is None:
        import RLPy as rlpy
    obj=selected_object(rlpy)
    live=snapshot(obj,rlpy=rlpy)
    validate_motion(motion,live)
    skel=_get_skeleton(obj)
    clip_count=int(skel.GetClipCount())
    if not clip_count:
        raise ValueError("selected Dragon has no motion clip")
    clip_index=clip_count-1 if clip_index is None else int(clip_index)
    if not 0<=clip_index<clip_count:
        raise ValueError("clip index out of range")
    clip=skel.GetClip(clip_index)
    fps=rlpy.RGlobal.GetFps()
    frames=sorted(set((0,motion["frame_count"]//2,motion["frame_count"]-1)))
    target={b["id"]:b["name"] for b in live["bones"]}
    wanted={target[k]: (k,motion["tracks"][k])
            for k in motion["tracks"]}
    skin_registry,_=build_bone_registry(skel.GetSkinBones() or [])
    skin_nodes=[x["node"] for x in skin_registry]
    sets={"skin_bones":(skin_nodes,{"method":"GetSkinBones",
                "available":True,"node_count":len(skin_nodes)})}
    for label,method in (
        ("motion_bones","GetMotionBones"),
        ("all_animation_bones","GetAllAnimationBone"),
    ):
        sets[label]=_probe_set(skel,method)
    out={
        "result":"diagnostic_only",
        "object":str(obj.GetName()),
        "clip_index":clip_index,
        "clip_count":clip_count,
        "source_motion_frame_count":motion["frame_count"],
        "sampled_frames":frames,
        "target_track_count":len(wanted),
        "node_sets":{},
        "same_name_identity_differences":{},
        "diagnostic_warning":(
            "Reallusion documents Motion Bones as the animation target; "
            "Skin Bone nodes may be distinct. This audit only READS FK "
            "controls; key counts and values do not prove rendered motion."
        ),
    }
    for label,(nodes,metadata) in sets.items():
        report=dict(metadata)
        if nodes is None:
            out["node_sets"][label]=report
            continue
        byname={}
        for n in nodes:
            info=_safe_node(n)
            if info.get("name") in wanted:
                byname.setdefault(info["name"],[]).append(n)
        report["matching_target_names"]=len(byname)
        report["missing_target_names"]=sorted(set(wanted)-set(byname))
        report["ambiguous_target_names"]=sorted(
            k for k,v in byname.items() if len(v)!=1)
        channels={}
        for name,(target_id,track) in wanted.items():
            candidates=byname.get(name,[])
            if len(candidates)!=1:
                continue
            node=candidates[0]
            item=_inspect_channel(clip,node,frames,fps,
                                  rlpy.IndexedFrameTime,track)
            item.update(_safe_node(node))
            channels[name]=item
        report["channels_by_name"]=channels
        report["target_bones_with_any_fk_key"]=sum(
            any(c or 0 for c in (x.get("key_counts_xyz") or []))
            for x in channels.values())
        report["target_bones_readback_matches_1e3"]=sum(
            x.get("readback_supported") and
            x.get("max_abs_error_radians") is not None and
            x["max_abs_error_radians"]<=1e-3
            for x in channels.values())
        report["target_bones_readback_unavailable"]=sum(
            not x.get("readback_supported") for x in channels.values())
        out["node_sets"][label]=report
    skin=out["node_sets"]["skin_bones"]["channels_by_name"]
    for label in ("motion_bones","all_animation_bones"):
        data=out["node_sets"].get(label,{})
        changes=[]
        for name,channel in data.get("channels_by_name",{}).items():
            other=skin.get(name)
            if other and channel.get("native_scene_id")!=other.get("native_scene_id"):
                changes.append({"name":name,
                    "skin_id":other.get("native_scene_id"),
                    "candidate_id":channel.get("native_scene_id"),
                    "skin_keys":other.get("key_counts_xyz"),
                    "candidate_keys":channel.get("key_counts_xyz")})
        out["same_name_identity_differences"][label]=changes
    return out
