"""Safe Dragon reimport using an already CLEAN, independently verified iProject.

iClone can retain the CURRENT pose after a Motion Clip is deleted.
Therefore deleting clips is not a pose reset and is never used here.
This workflow replaces the WHOLE CURRENT SCENE with an explicitly selected
clean template, after saving a full backup. A strict rest-geometry and
no-motion check runs on the loaded scene BEFORE any FK keys are written.
"""
import json
import math
import pathlib

from .direct_import import (
    _selected_and_fps, prepare_direct_import, execute_direct_import
)
from .import_pipeline import _json
from .motion_writer import _require_sdk_ok
from .rlpy_reader import _get_skeleton, snapshot
from .scene_guard import check_clean_clip
from .schema import ContractError
from .verify_import import DEFAULT_SOURCE, audit_rig


def _source_dragon(motion_path):
    source=pathlib.Path(motion_path).resolve()
    if not source.is_file() or source.suffix.lower()!=".json":
        raise ContractError("Select the original Dragon AnyMove motion.json")
    motion=_json(source)
    info=_json(DEFAULT_SOURCE)
    names=[bone["name"] for bone in info["bones"]]
    if (motion.get("bones")!=names or
        type(motion.get("fps")) not in (float,int) or
        float(motion["fps"])!=24.0):
        raise ContractError("Replacement requires original 42-Bone 24-FPS Dragon motion")
    return source,info


def inspect_replacement_eligibility(motion_path,rlpy=None):
    """Purely read-only candidate check; existing clip count can be ZERO.

    A posed skeleton can have identical names/parents, so this check
    NEVER authorizes deletion of its clips or skipping Rest verification.
    """
    if rlpy is None:
        import RLPy as rlpy
    source,reference=_source_dragon(motion_path)
    obj,_=_selected_and_fps(rlpy,24)
    skel=_get_skeleton(obj)
    count=int(skel.GetClipCount())
    if count<0 or count>20:
        raise ContractError("Cannot safely inspect existing Motion Clip count")
    rig=audit_rig(reference,snapshot(obj,rlpy=rlpy))
    if (rig["matched_unique_names"]!=42 or rig["missing_names"] or
        rig["ambiguous_names"] or rig["mismatched_parents"]):
        raise ContractError(
            "Selected Dragon does not match the original 42-Bone hierarchy; "
            "missing: %s; ambiguous: %s; wrong parents: %s"%
            (rig["missing_names"],rig["ambiguous_names"],
             rig["mismatched_parents"]))
    return {
        "status":"original_dragon_hierarchy_matched_but_rest_not_assumed",
        "actor_name":str(obj.GetName()),
        "existing_clip_count":count,
        "matched_dragon_bones":42,
        "rest_pose_verified":rig["result"]=="compatible_static_rest_snapshot",
        "observed_scale":rig["source_to_target_uniform_scale"],
        "source_motion":str(source),
        "warning":"Deleting a Motion Clip can preserve the current pose. "
                  "Use an independently clean Dragon iProject template; "
                  "do NOT assume this current scene is restored."
    }


def _free_path(folder,stem,suffix):
    for i in range(1,1000):
        path=folder/("%s_%03d%s"%(stem,i,suffix))
        if not path.exists():
            return path
    raise ContractError("No unused backup project filename available")


def _scene_objects(rlpy):
    """Use documented RScene enumeration; never fabricate object names."""
    scene=getattr(rlpy,"RScene",None)
    if scene is None:
        raise ContractError("iClone RScene unavailable")
    candidates=[]
    avatars=getattr(scene,"GetAvatars",None)
    if callable(avatars):
        enum=getattr(rlpy,"EAvatarType_All",None)
        try:
            got=avatars(enum) if enum is not None else avatars()
        except (TypeError,AttributeError):
            got=avatars()
        candidates.extend(list(got or []))
    props=getattr(scene,"GetProps",None)
    if callable(props):
        candidates.extend(list(props() or []))
    if len(candidates)>200:
        raise ContractError("Loaded template has too many scene objects")
    return candidates


def select_verified_clean_dragon(rlpy):
    """Find EXACTLY one rest-compatible Dragon in the loaded clean project."""
    reference=_json(DEFAULT_SOURCE)
    matches=[]
    errors=[]
    for obj in _scene_objects(rlpy):
        try:
            skel=_get_skeleton(obj)
            audit=audit_rig(reference,snapshot(obj,rlpy=rlpy))
            if (audit["result"]!="compatible_static_rest_snapshot" or
                audit.get("source_to_target_uniform_scale") is None or
                abs(float(audit["source_to_target_uniform_scale"])-100.)>.25):
                continue
            clean=check_clean_clip(skel,obj,rlpy)
            matches.append((obj,audit,clean))
        except Exception as exc:
            errors.append(str(exc))
    if len(matches)!=1:
        raise ContractError(
            "Clean template needs EXACTLY one 42-Bone Dragon in original "
            "100x rest pose with no animated FK/Position/Scale keys; "
            "found %d compatible candidates. Do not use an animated "
            "backup as a clean template."%len(matches))
    obj,audit,clean=matches[0]
    scene=rlpy.RScene
    clear=getattr(scene,"ClearSelectObjects",None)
    select=getattr(scene,"SelectObject",None)
    if not callable(clear) or not callable(select):
        raise ContractError("RScene selection APIs not available")
    _require_sdk_ok(clear(),"clear template selection")
    _require_sdk_ok(select(obj),"select clean Dragon")
    picked,_=_selected_and_fps(rlpy,24)
    if str(picked.GetName())!=str(obj.GetName()):
        raise ContractError("Cannot verify selected clean Dragon after loading template")
    return {
        "actor":str(obj.GetName()),
        "rest_geometry":audit["result"],
        "observed_fbx_scale":audit["source_to_target_uniform_scale"],
        "clean_layer":clean.get("exposed_motion_empty"),
    }


def replace_dragon_motion_with_backup(motion_path,output_parent,rlpy=None,
                                      clean_project_path=None):
    """Back up current scene, load CLEAN template, strict import or restore.

    THIS REPLACES THE ENTIRE SCENE, NOT MERELY THE SELECTED DRAGON.
    The caller must show this warning and get explicit approval first.
    A clean project path is mandatory: deleting clips cannot reset rest.
    """
    if rlpy is None:
        import RLPy as rlpy
    source,_=_source_dragon(motion_path)
    if not clean_project_path:
        raise ContractError(
            "Cannot repair Rest Pose by deleting Motion Clips. "
            "Provide a verified CLEAN Dragon .iProject template. "
            "If none exists, import the original Dragon FBX into a new "
            "24-FPS scene and run Import once to save a reusable clean template.")
    clean_project=pathlib.Path(clean_project_path).resolve()
    if not clean_project.is_file() or clean_project.suffix.lower()!=".iproject":
        raise ContractError("Select an existing CLEAN Dragon .iProject template")
    folder=pathlib.Path(output_parent).resolve()
    if not folder.is_dir():
        raise ContractError("Choose existing output folder for iProject backup")
    saver=getattr(getattr(rlpy,"RFileIO",None),"SaveProject",None)
    loader=getattr(getattr(rlpy,"RFileIO",None),"LoadProject",None)
    if not callable(saver) or not callable(loader):
        raise ContractError(
            "Safe scene replacement requires SaveProject AND LoadProject")
    backup=_free_path(folder,"AnyMove_BeforeSceneReplace",".iProject")
    _require_sdk_ok(saver(str(backup)),"save entire scene before reimport")
    if not backup.is_file() or backup.stat().st_size<=0:
        raise ContractError("Backup iProject not verified; scene was not replaced")
    phase="backup_saved"
    try:
        _require_sdk_ok(loader(str(clean_project)),"load verified CLEAN Dragon template")
        phase="clean_template_loaded"
        fps=rlpy.RGlobal.GetFps()
        _require_sdk_ok(
            rlpy.RGlobal.SetTime(rlpy.IndexedFrameTime(0,fps)),
            "set clean scene Frame 0")
        selected=select_verified_clean_dragon(rlpy)
        phase="clean_rest_verified"
        prepared=prepare_direct_import(str(source),rlpy=rlpy)
        phase="import_prepared"
        result=execute_direct_import(
            prepared,folder,rlpy=rlpy,save_project=True,
            preserve_clean_template=True)
        phase="import_executed"
        if result.get("status")!=(
                "completed_full_frame_verified_project_saved"):
            raise ContractError(
                "Imported keys were NOT fully verified/saved: "+
                str(result.get("status")))
        receipt={
            "status":"scene_recreated_from_clean_dragon_and_motion_imported",
            "original_scene_backup":str(backup),
            "clean_dragon_template":str(clean_project),
            "new_iProject":result["iProject_path"],
            "new_import_receipt":str(pathlib.Path(result["output_directory"])/
                                     "import_receipt.json"),
            "selected_dragon":selected,
            "note":"The WHOLE scene was replaced by the clean template. "
                   "The previous scene is recoverable from the backup. "
                   "Root World Position still requires visual confirmation."
        }
        path=_free_path(folder,"AnyMove_SceneReplace_receipt",".json")
        path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),
                        encoding="utf-8")
        receipt["receipt_path"]=str(path)
        return receipt
    except Exception as exc:
        try:
            _require_sdk_ok(
                loader(str(backup)),"restore original complete scene")
        except Exception as rollback:
            raise ContractError(
                "Clean-scene import failed at %s: %s. Restoring the previous "
                "scene ALSO FAILED (%s). Open the intact backup manually: %s"%
                (phase,exc,rollback,backup)) from exc
        raise ContractError(
            "Clean-scene import aborted at %s: %s. Previous scene reloaded "
            "from backup %s. No replacement kept."%
            (phase,exc,backup)) from exc
