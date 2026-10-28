"""Direct AnyMove Dragon motion.json -> iClone 8 native Motion and iProject.

Single operation in the iClone Python host (Python 3.8):
  1. Validate the exact known Dragon Rest rig, target unposed Skeleton,
     24 FPS, clean motion-free scene, and frame-zero playhead.
  2. Create an immutable source / FK / audit / SHA manifest package.
  3. Write all 42 Bone Rotation and Pelvis Position XYZ keys.
  4. Read back every integer frame of every Layer Controller.
  5. Only if readback passes, optionally save a native .iProject.
  6. Persist a concise import receipt, including any partial failure.

Per-frame Key values have been validated on a real iClone Dragon.
This is the SAME-RIG Dragon reference profile, NOT universal retargeting.
No subframe/interpolation work is required in this import flow.
"""
import json
import os
import pathlib

from . import __version__
from .import_pipeline import (
    DEFAULT_SOURCE, OUTPUT_MOTION, OUTPUT_MANIFEST, PROFILE_ID,
    build_bundle, save_bundle, verify_bundle,
    verify_fk_manifest_for_writer, _digest, _json
)
from .motion_writer import write_selected, _require_sdk_ok
from .rlpy_reader import selected_object, snapshot, _get_skeleton
from .runtime_controller_audit import audit_full_controller_frames
from .scene_guard import preflight_clean_dragon
from .schema import ContractError

RECEIPT_NAME = "import_receipt.json"
READBACK_NAME = "runtime_controller_audit.json"
PROJECT_NAME = "AnyMove_Dragon.iProject"
DIRECT_VERSION = "anymove.iclone.direct_dragon_import/1"


def _put_json(path, data):
    """Atomically update a small JSON report; never replace animation inputs."""
    path=pathlib.Path(path)
    part=path.with_name(path.name+".partial")
    try:
        with open(str(part),"w",encoding="utf-8") as stream:
            json.dump(data,stream,ensure_ascii=False,indent=2,
                      allow_nan=False)
            stream.write("\n")
        os.replace(str(part),str(path))
    finally:
        if part.exists():
            part.unlink()


def _selected_and_fps(rlpy, fps_expected):
    obj=selected_object(rlpy)
    fps=rlpy.RGlobal.GetFps()
    to_double=getattr(fps,"ToDouble",None)
    if not callable(to_double) or abs(float(to_double())-fps_expected)>1e-5:
        raise ContractError(
            "iClone project must be %d FPS; change Project FPS before import"%
            fps_expected)
    current=rlpy.RGlobal.GetTime()
    if int(rlpy.GetFrameIndex(current,fps))!=0:
        raise ContractError(
            "iClone Timeline must be at Frame 0 before direct import")
    return obj, fps


def prepare_direct_import(source_motion_path,rlpy=None):
    """Read-only preparation. Call before the UI asks permission to mutate."""
    if rlpy is None:
        import RLPy as rlpy
    path=pathlib.Path(source_motion_path)
    if not path.is_file() or path.suffix.lower()!=".json":
        raise ContractError("Select a readable original AnyMove motion.json")
    source=_json(path)
    reference=_json(DEFAULT_SOURCE)
    obj,_=_selected_and_fps(rlpy,24)
    target=snapshot(obj,rlpy=rlpy)
    fk,offline,manifest=build_bundle(
        source,target,reference,reference)
    scene_guard=preflight_clean_dragon(
        obj,_get_skeleton(obj),fk,rlpy)
    return {
        "source_motion_path":str(path),
        "source_motion":source,
        "source_rig":reference,
        "target_snapshot":target,
        "fk":fk,
        "offline_audit":offline,
        "manifest":manifest,
        "scene_guard":scene_guard,
        "source_fingerprint":_digest(source),
        "source_profile":PROFILE_ID,
        "frames":int(fk["frame_count"]),
        "rotation_bones":len(fk["tracks"]),
        "position_bones":len(fk.get("position_tracks") or {}),
        "expected_scalar_samples":fk["frame_count"]*3*(
            len(fk["tracks"])+len(fk.get("position_tracks") or {})),
        "preview_note":"READ-ONLY. Only original 42-Bone Dragon with a clean "
                       "100x FBX import, 24 FPS and Frame 0 is supported."
    }


def _fresh_output_directory(parent, fingerprint):
    folder=pathlib.Path(parent)
    if not folder.is_dir():
        raise ContractError("Choose an EXISTING output parent folder")
    stem="AnyMove_Dragon_"+fingerprint[:10]
    for index in range(1,1000):
        candidate=folder/(stem if index==1 else stem+"_%03d"%index)
        if not candidate.exists():
            return candidate
    raise ContractError("too many prior Dragon imports in output parent folder")


def _reset_frame_zero(rlpy, fps):
    """Best-effort restore of initial playhead before native project save."""
    try:
        wanted=rlpy.IndexedFrameTime(0,fps)
        _require_sdk_ok(rlpy.RGlobal.SetTime(wanted),"SetTime Frame 0")
        return int(rlpy.GetFrameIndex(rlpy.RGlobal.GetTime(),fps))==0
    except Exception:
        return False


def execute_direct_import(prepared,output_parent,rlpy=None,
                          save_project=True):
    """Complete import after the user's explicit scene-write confirmation.

    Retains a receipt if the writer succeeds but controller audit or native
    SaveProject subsequently fails. Never claims success solely from SetValue.
    A failed write may leave keys if the underlying SDK rollback failed; the
    receipt distinguishes the phase and tells the user not to import again.
    """
    if rlpy is None:
        import RLPy as rlpy
    if not isinstance(prepared,dict) or prepared.get(
            "source_profile")!=PROFILE_ID:
        raise ContractError("invalid prepared Dragon import plan")
    if type(save_project) is not bool:
        raise ContractError("save_project must be bool")
    obj,fps=_selected_and_fps(rlpy,prepared["fk"]["fps"])
    # Read-only recheck after modal UI interaction: reject scene changes.
    guard=preflight_clean_dragon(
        obj,_get_skeleton(obj),prepared["fk"],rlpy)
    output=_fresh_output_directory(output_parent,
                                   prepared["source_fingerprint"])
    source=prepared["source_motion"]
    target=prepared["target_snapshot"]
    reference=prepared["source_rig"]
    manifest=save_bundle(output,source,target,reference,reference)
    verified=verify_bundle(output,source,target,reference,reference)
    package_fk=_json(output/OUTPUT_MOTION)
    package_integrity=verify_fk_manifest_for_writer(
        output/OUTPUT_MOTION,package_fk)

    receipt={
        "schema_version":DIRECT_VERSION,
        "tool_version":__version__,
        "status":"package_ready_no_scene_write",
        "source_file":prepared["source_motion_path"],
        "source_profile":PROFILE_ID,
        "source_motion_sha256":prepared["source_fingerprint"],
        "output_directory":str(output),
        "package_manifest":OUTPUT_MANIFEST,
        "package_verified":verified["status"],
        "package_fk_integrity":package_integrity["status"],
        "frame_count":prepared["frames"],
        "rotation_bones":prepared["rotation_bones"],
        "position_bones":prepared["position_bones"],
        "expected_scalar_samples":prepared["expected_scalar_samples"],
        "clean_scene_preflight":guard,
        "clip_index":None,
        "scene_may_contain_new_keys":False,
        "controller_audit_status":None,
        "controller_audit_file":None,
        "iProject_path":None,
        "iProject_saved":False,
        "playhead_restored_to_frame0":False,
        "warnings":[]
    }
    _put_json(output/RECEIPT_NAME,receipt)
    try:
        written=write_selected(package_fk,rlpy=rlpy,fps_confirmed=True)
    except Exception as exc:
        receipt["status"]="sdk_write_failed_rollback_status_must_be_checked"
        receipt["write_exception"]=type(exc).__name__+": "+str(exc)
        receipt["scene_may_contain_new_keys"]=True
        receipt["warnings"].append(
            "Check iClone scene before retrying. SDK rollback may have failed; "
            "DO NOT import twice into the same scene.")
        _put_json(output/RECEIPT_NAME,receipt)
        raise ContractError(
            "Dragon SDK write failed. Receipt: %s. %s"%
            (output/RECEIPT_NAME,exc))
    receipt["scene_may_contain_new_keys"]=True
    receipt["clip_index"]=int(written["clip_count_after"])-1
    receipt["writer_status"]=written["status"]
    receipt["writer_clip_strategy"]=written["clip_strategy"]
    receipt["writer_warnings"]=written.get("warnings",[])
    receipt["status"]="sdk_keys_written_controller_audit_pending"
    _put_json(output/RECEIPT_NAME,receipt)

    try:
        audit=audit_full_controller_frames(
            package_fk,rlpy=rlpy,clip_index=receipt["clip_index"])
        audit["bundle_manifest_check"]=package_integrity
        _put_json(output/READBACK_NAME,audit)
        receipt["controller_audit_file"]=READBACK_NAME
        receipt["controller_audit_status"]=audit["status"]
        receipt["verified_scalar_samples"]=audit["verified_scalar_samples"]
        receipt["mismatched_scalar_samples"]=audit["mismatched_scalar_samples"]
        receipt["channels_with_wrong_key_count"]=audit[
            "channels_with_wrong_key_count"]
        receipt["maximum_controller_abs_error"]=audit[
            "maximum_abs_error_in_channel_units"]
        valid=(audit["status"]=="all_frame_controller_values_match"
               and audit["verified_scalar_samples"]==
                   receipt["expected_scalar_samples"]
               and audit["mismatched_scalar_samples"]==0
               and audit["channels_with_wrong_key_count"]==0)
        if not valid:
            receipt["status"]="controller_audit_failed_project_not_saved"
            receipt["warnings"].append(
                "Motion Clip has been written but some controller data FAILED "
                "readback. Inspect runtime_controller_audit.json. "
                "Do not layer another clip or claim import complete.")
            _put_json(output/RECEIPT_NAME,receipt)
            return receipt
    except Exception as exc:
        receipt["controller_audit_status"]="exception"
        receipt["controller_audit_exception"]=type(exc).__name__+": "+str(exc)
        receipt["status"]="controller_audit_exception_project_not_saved"
        receipt["warnings"].append(
            "Clip already contains written keys. Inspect existing animation, "
            "DO NOT re-import into this scene.")
        _put_json(output/RECEIPT_NAME,receipt)
        return receipt

    receipt["status"]="controller_data_verified"
    receipt["playhead_restored_to_frame0"]=_reset_frame_zero(rlpy,fps)
    if not receipt["playhead_restored_to_frame0"]:
        receipt["warnings"].append(
            "Could not restore Timeline to Frame 0; project may save at "
            "the current playback frame.")
    _put_json(output/RECEIPT_NAME,receipt)

    if save_project:
        project=output/PROJECT_NAME
        saver=getattr(getattr(rlpy,"RFileIO",None),"SaveProject",None)
        if not callable(saver):
            receipt["status"]="controller_verified_manual_project_save_required"
            receipt["warnings"].append(
                "RFileIO.SaveProject unsupported in this SDK. Save this "
                "modified scene manually in iClone File > Save As.")
        else:
            try:
                if project.exists():
                    raise ContractError("refusing to overwrite existing iProject")
                _require_sdk_ok(saver(str(project)),"RFileIO.SaveProject")
                if not project.is_file() or project.stat().st_size==0:
                    raise ContractError(
                        "SaveProject returned success but no nonempty "
                        ".iProject file was found")
                receipt["iProject_saved"]=True
                receipt["iProject_path"]=str(project)
                receipt["status"]="completed_full_frame_verified_project_saved"
            except Exception as exc:
                receipt["status"]="controller_verified_native_project_save_failed"
                receipt["project_save_error"]=type(exc).__name__+": "+str(exc)
                receipt["warnings"].append(
                    "Controller import PASS but native iProject save failed. "
                    "Keep iClone project open and save manually; do not "
                    "re-import into an already keyed scene.")
    else:
        receipt["status"]="completed_full_frame_verified_manual_project_save"

    _put_json(output/RECEIPT_NAME,receipt)
    return receipt
