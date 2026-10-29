"""Backed-up, confirmed replacement of pre-existing Dragon Motion Clips.

A posed Dragon is not an unposed target, even when the same FK motion was
previously imported. The replacement path saves the complete original scene
before deleting any clip, then RE-RUNS strict, unchanged same-rig Rest
and clean-scene guards before writing the new source motion.
"""
import json
import pathlib

from .direct_import import (
    _selected_and_fps, prepare_direct_import, execute_direct_import
)
from .import_pipeline import _json
from .motion_writer import _require_sdk_ok
from .rlpy_reader import _get_skeleton, snapshot
from .schema import ContractError
from .verify_import import DEFAULT_SOURCE, audit_rig


def inspect_replacement_eligibility(motion_path,rlpy=None):
    """Read-only. A hierarchy match is NOT permission to modify the scene."""
    if rlpy is None:
        import RLPy as rlpy
    source_path=pathlib.Path(motion_path)
    if not source_path.is_file() or source_path.suffix.lower()!=".json":
        raise ContractError("Select a valid source AnyMove Dragon motion.json")
    motion=_json(source_path)
    reference=_json(DEFAULT_SOURCE)
    names=[b["name"] for b in reference["bones"]]
    if (motion.get("bones")!=names or
        type(motion.get("fps")) not in (int,float) or
        abs(float(motion["fps"])-24.0)>1e-5):
        raise ContractError(
            "Cannot automatically replace Motion for a foreign/non-24-FPS Dragon")
    obj,fps=_selected_and_fps(rlpy,24)
    skel=_get_skeleton(obj)
    count=int(skel.GetClipCount())
    if count<=0 or count>20:
        raise ContractError(
            "Expected 1-20 existing Motion Clips for backed-up replacement")
    rig=audit_rig(reference,snapshot(obj,rlpy=rlpy))
    if (rig["matched_unique_names"]!=42 or rig["missing_names"] or
        rig["ambiguous_names"] or rig["mismatched_parents"]):
        raise ContractError(
            "Existing Motion cannot be cleared safely: selected actor "
            "does not have the exact 42-Bone Dragon hierarchy. "
            "Missing %s; ambiguous %s; wrong parents %s"%
            (rig["missing_names"],rig["ambiguous_names"],
             rig["mismatched_parents"]))
    return {
        "status":"candidate_for_backed_up_clip_replacement",
        "actor_name":str(obj.GetName()),
        "existing_clip_count":count,
        "matched_dragon_bones":42,
        "rest_pose_verified":False,
        "source_motion":str(source_path),
        "warning":"Only hierarchy verified so far. After backup and removal "
                  "the SAME strict Rest Pose and clip safety gates MUST pass; "
                  "otherwise the original scene will be restored."
    }


def _free_path(folder,stem,suffix):
    for i in range(1,1000):
        path=folder/("%s_%03d%s"%(stem,i,suffix))
        if not path.exists():
            return path
    raise ContractError("No unused backup project filename available")


def replace_dragon_motion_with_backup(motion_path,output_parent,rlpy=None):
    """Explicitly approved destructive replacement with verified backup.

    The caller MUST separately obtain confirmation after explaining the
    current clip count and full-scene backup. A successful backup is retained
    even after the new native iProject has been saved.
    """
    if rlpy is None:
        import RLPy as rlpy
    plan=inspect_replacement_eligibility(motion_path,rlpy=rlpy)
    folder=pathlib.Path(output_parent)
    if not folder.is_dir():
        raise ContractError("Choose existing output folder for iProject backup")
    saver=getattr(getattr(rlpy,"RFileIO",None),"SaveProject",None)
    loader=getattr(getattr(rlpy,"RFileIO",None),"LoadProject",None)
    if not callable(saver) or not callable(loader):
        raise ContractError(
            "Safe Motion replacement requires RFileIO.SaveProject and LoadProject")
    backup=_free_path(folder,"AnyMove_BeforeMotionReplace",".iProject")
    # No scene mutation is permitted until the whole scene is backed up.
    _require_sdk_ok(saver(str(backup)),"save pre-replacement project")
    if not backup.is_file() or backup.stat().st_size<=0:
        raise ContractError(
            "Could not verify full-scene backup: no old clips were deleted")
    phase="backup_verified"
    try:
        obj,fps=_selected_and_fps(rlpy,24)
        skel=_get_skeleton(obj)
        # No other scene changes should have occurred while saving the backup.
        if int(skel.GetClipCount())!=plan["existing_clip_count"]:
            raise ContractError("Clip count changed while backing up scene")
        for expected_count in range(plan["existing_clip_count"],0,-1):
            old=skel.GetClip(expected_count-1)
            if old is None:
                raise ContractError("Could not inspect previous Motion Clip")
            result=skel.DeleteClip(old)
            if callable(getattr(result,"IsError",None)) and result.IsError():
                raise ContractError("DeleteClip reported SDK error")
            if int(skel.GetClipCount())!=expected_count-1:
                raise ContractError(
                    "DeleteClip did not reduce clip count from %d"%
                    expected_count)
        phase="old_clips_deleted"
        # Deleting clips can reset the original pose. SetTime ensures the
        # strict rest snapshot is evaluated at frame zero, not a later time.
        _require_sdk_ok(
            rlpy.RGlobal.SetTime(rlpy.IndexedFrameTime(0,fps)),
            "set frame zero after Motion cleanup")
        obj_update=getattr(obj,"Update",None)
        if callable(obj_update):
            obj_update()
        # Do not skip either exact Rest Geometry or clean FK/Position/Scale gate.
        prepared=prepare_direct_import(motion_path,rlpy=rlpy)
        phase="strict_rest_pose_passed"
        result=execute_direct_import(
            prepared,folder,rlpy=rlpy,save_project=True)
        phase="import_executed"
        if result.get("status")!=(
                "completed_full_frame_verified_project_saved"):
            raise ContractError(
                "New Motion did not pass all-frame verification/native save: "+
                str(result.get("status")))
        receipt={
            "status":"backed_up_previous_motion_and_imported_new_motion",
            "backup_iProject":str(backup),
            "new_iProject":result["iProject_path"],
            "removed_clip_count":plan["existing_clip_count"],
            "new_import_receipt":str(pathlib.Path(result["output_directory"])/
                                     "import_receipt.json"),
            "note":"Previous animation remains recoverable in the backup. "
                   "New animation passed controller readback, not evaluated "
                   "Skin/Root translation visual calibration."
        }
        path=_free_path(folder,"AnyMove_MotionReplace_receipt",".json")
        path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),
                        encoding="utf-8")
        receipt["receipt_path"]=str(path)
        return receipt
    except Exception as exc:
        try:
            _require_sdk_ok(
                loader(str(backup)),"restore saved pre-replacement project")
        except Exception as rollback:
            raise ContractError(
                "Motion replacement failed at %s: %s. ALSO FAILED to reload "
                "original project: %s. Open the untouched backup manually: %s"%
                (phase,exc,rollback,backup)) from exc
        raise ContractError(
            "Motion replacement aborted at %s: %s. The original project "
            "was restored from: %s. No replacement was kept."%
            (phase,exc,backup)) from exc
