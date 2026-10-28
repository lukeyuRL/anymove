"""Repeatable same-Dragon-rig import bundle: source -> full FK -> offline audit.

This is a *reference-profile* import, NOT a generic iClone retarget.
For the original AnyMove Dragon with its 100x FBX iClone scene hierarchy,
the user visually verified 41 joint rotations and root rotation and
sampled 5 root translation frames. We preserve those exact formulas.

Command:
 python -m iclone_adapter.import_pipeline prepare
       --motion runs/dragon/clips/roar/motion.json
       --skeleton F:/anymove_test/skeleton.json
       --output F:/anymove_test/dragon_roar_import

The generated FK file is passed to the existing iClone Writer, which still
requires a new, unposed project. The offline 100%-of-frame quaternion
audit checks *data conversion*, NOT SDK interpolation or evaluated skin pose.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import tempfile

from .anymove_bridge import from_euler_xyz, qnormalize, qmul, qinverse_unit
from .dragon_preview import make_preview
from .motion_contract import validate_motion
from .rest_basis import relative_rest_basis, root_rest_basis
from .root_motion import _rotate_vector
from .schema import ContractError, skeleton_hash, validate_skeleton
from .verify_import import DEFAULT_SOURCE, audit_rig

BUNDLE_VERSION = "anymove.iclone.same_rig_import_bundle/1"
PROFILE_ID = "dragon_fbxsame_100x_rest_pelvis_local_v1"
REFERENCE_SOURCE_BONES = 42
REFERENCE_FPS = 24
REFERENCE_TARGET_SCALE = 100.0
REFERENCE_ROOT_POSITION_MULTIPLIER = 1.0
OUTPUT_MOTION = "iclone_fk_full.json"
OUTPUT_AUDIT = "offline_frame_audit.json"
OUTPUT_MANIFEST = "import_manifest.json"


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            part = stream.read(1048576)
            if not part:
                break
            digest.update(part)
    return digest.hexdigest()


def _json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write("\n")


def _qerror_deg(a,b):
    qa,qb = qnormalize(a),qnormalize(b)
    dot = abs(sum(x*y for x,y in zip(qa,qb)))
    return math.degrees(2.*math.acos(max(-1.,min(1.,dot))))


def _norm(values):
    return math.sqrt(sum(v*v for v in values))


def _vsub(a,b):
    return [x-y for x,y in zip(a,b)]


def _validate_profile(motion,target,source_info,reference_source):
    validate_skeleton(target)
    original = reference_source["bones"]
    supplied = source_info.get("bones")
    if not isinstance(supplied,list) or supplied != original:
        raise ContractError(
            "Dragon reference profile requires the exact checked-in "
            "assets/dragon/rigged/rig_info.json Bone rest data. "
            "Other characters require a separately validated profile.")
    if len(original) != REFERENCE_SOURCE_BONES:
        raise ContractError("Dragon reference profile expects exactly 42 source bones")
    names = [b["name"] for b in original]
    if names[0] != "pelvis" or motion.get("bones") != names:
        raise ContractError("source motion Bone names/order do not match Dragon profile")
    fps = motion.get("fps")
    if type(fps) not in (float,int) or float(fps) != REFERENCE_FPS:
        raise ContractError("Dragon profile requires original 24 FPS without resampling")
    report = audit_rig(source_info,target)
    if report.get("result") != "compatible_static_rest_snapshot":
        raise ContractError("Dragon target is not an unposed, same-rig snapshot: "+
                            str(report.get("result")))
    scale = report["source_to_target_uniform_scale"]
    if not math.isfinite(scale) or abs(scale - REFERENCE_TARGET_SCALE) > 0.25:
        raise ContractError(
            "Dragon reference requires observed 100x imported FBX "
            "hierarchy (actual: %.5g); do not assume another rig is calibrated"%scale)
    if len(target["bones"]) < REFERENCE_SOURCE_BONES:
        raise ContractError("too few iClone bones for Dragon")
    return report


def _audit_all_frames(source,target,source_info,converted):
    """Recompute expected quaternion for all frames of all 42 bones.

    Also check source Quaternion continuity, per-bone XYZ wrap warnings,
    and root translation converted at every integer frame. This is
    independent math/readback of the serialized output, but NOT a skin
    evaluation under the iClone SDK.
    """
    samples = source["euler"]
    names = source["bones"]
    count = len(samples)
    if converted["frame_count"] != count:
        raise ContractError("converted full-frame count differs")
    if len(converted["tracks"]) != REFERENCE_SOURCE_BONES:
        raise ContractError("profile requires all 42 rotation tracks")
    by_name = {b["name"]:b["id"] for b in target["bones"]
               if b["name"] in names}
    if len(by_name) != REFERENCE_SOURCE_BONES:
        raise ContractError("ambiguous or missing Dragon target bones")
    if set(converted["tracks"]) != set(by_name.values()):
        raise ContractError("converted FK tracks must contain exactly 42 Dragon bones")
    positions = converted.get("position_tracks")
    root_id=by_name["pelvis"]
    if not isinstance(positions,dict) or set(positions)!={root_id}:
        raise ContractError("profile must emit exactly one pelvis Position XYZ track")

    by_bone=[]
    maximum_rotation_error=0.
    max_quat_step=0.
    discontinuities=[]
    for index,name in enumerate(names):
        bone_id=by_name[name]
        track=converted["tracks"][bone_id]
        rest=(root_rest_basis(source_info) if index==0
              else relative_rest_basis(source_info,name))
        rest_inv=qinverse_unit(rest)
        previous_euler=None
        previous_quat=None
        max_err=0.
        max_step=0.
        wrap_events=[]
        for frame in range(count):
            original=(qnormalize(source["root_quat"][frame])
                      if index==0 else from_euler_xyz(samples[frame][index]))
            expected=qmul(qmul(rest,original),rest_inv)
            row=track[frame]
            actual=from_euler_xyz(row)
            error=_qerror_deg(expected,actual)
            max_err=max(max_err,error)
            if previous_quat is not None:
                max_step=max(max_step,_qerror_deg(previous_quat,actual))
            if previous_euler is not None:
                for axis in range(3):
                    if abs(row[axis]-previous_euler[axis])>math.pi:
                        wrap_events.append({"frame":frame,"axis":"XYZ"[axis],
                                            "delta_radians":row[axis]-previous_euler[axis]})
            previous_euler=row
            previous_quat=actual
        if max_err>.001:
            raise ContractError(
                "full-frame quaternion conversion failed: %s error %.9f degrees"%
                (name,max_err))
        if wrap_events:
            discontinuities.append({"bone":name,"count":len(wrap_events),
                                    "first_events":wrap_events[:5]})
        maximum_rotation_error=max(maximum_rotation_error,max_err)
        max_quat_step=max(max_quat_step,max_step)
        by_bone.append({
            "bone":name,
            "frames_checked":count,
            "maximum_quaternion_error_deg":max_err,
            "maximum_adjacent_frame_rotation_deg":max_step,
            "XYZ_euler_wrap_events":len(wrap_events),
        })

    root=converted["root_motion"]
    if (root["target_bone_id"]!=root_id or
        root.get("root_position_space")!="pelvis_local" or
        root.get("root_position_scale")!=1.0 or
        root.get("calibrated") is not False):
        raise ContractError("reference Root Motion profile configuration drift")
    rest_inv=qinverse_unit(root_rest_basis(source_info))
    pos_track=positions[root_id]
    if len(pos_track)!=count:
        raise ContractError("root position track frame count drift")
    max_position_error=0.
    cumulative_travel=0.
    last=None
    for frame in range(count):
        wanted=_rotate_vector(rest_inv,source["trans"][frame])
        actual=pos_track[frame]
        error=_norm(_vsub(wanted,actual))
        max_position_error=max(max_position_error,error)
        if last is not None:
            cumulative_travel+=_norm(_vsub(actual,last))
        last=actual
    if max_position_error>1e-8:
        raise ContractError("root Position conversion mismatch")
    if not math.isfinite(cumulative_travel):
        raise ContractError("nonfinite root translation arc length")
    audit={
        "schema_version":"anymove.iclone.full_frame_fidelity_audit/1",
        "result":"all_serialized_frames_match_source_math",
        "reference_profile":PROFILE_ID,
        "frames_checked":count,
        "rotation_bones_checked":len(by_bone),
        "rotation_samples_checked":count*len(by_bone),
        "root_position_samples_checked":count,
        "maximum_quaternion_error_deg":maximum_rotation_error,
        "maximum_root_position_error_source_units":max_position_error,
        "maximum_adjacent_frame_rotation_deg":max_quat_step,
        "pelvis_position_arc_length_source_units":cumulative_travel,
        "euler_wrap_bones":discontinuities,
        "per_bone":by_bone,
        "validation_limitations":[
            "Every integer-frame serialized FK sample was checked mathematically.",
            "No iClone RLPy Layer key, evaluated bone pose or mesh deformation was read.",
            "Subframe interpolation, looping, IK/physics, other skeletons unverified.",
            "The 100x target-import scale is from Dragon FBX scene hierarchy.",
        ]
    }
    return audit


def build_bundle(source_motion,target_skeleton,source_rig_info,
                 reference_source_info):
    """Pure data conversion: no RLPy, Blender or files written."""
    profile_report=_validate_profile(
        source_motion,target_skeleton,source_rig_info,reference_source_info)
    prepared=make_preview(
        source_motion,target_skeleton,source_rig_info,
        bones="all",basis="rest",root_mode="full",
        root_position_space="pelvis_local",
        root_position_scale=REFERENCE_ROOT_POSITION_MULTIPLIER)
    # Force unchanged contracts; pipeline metadata is a manifest reference,
    # not an artificial production calibration=true flag.
    prepared["pipeline"]={
        "schema_version":BUNDLE_VERSION,
        "profile_id":PROFILE_ID,
        "source_motion_sha256":_digest(source_motion),
        "source_dragon_bones_sha256":_digest(source_rig_info["bones"]),
        "target_skeleton_hash":skeleton_hash(target_skeleton),
        "target_unposed_world_sha256":_digest({
            b["id"]:b["snapshot_world_matrix"] for b in target_skeleton["bones"]
        }),
        "offline_audit":"see offline_frame_audit.json",
        "runtime_evidence":"sampled Dragon reference; all-frame SDK validation pending",
    }
    validate_motion(prepared,target_skeleton)
    audit=_audit_all_frames(source_motion,target_skeleton,source_rig_info,
                           prepared)
    manifest={
        "schema_version":BUNDLE_VERSION,
        "profile_id":PROFILE_ID,
        "status":"offline_all_frame_conversion_pass_runtime_not_certified",
        "source":{
            "motion_canonical_sha256":_digest(source_motion),
            "rig_bones_canonical_sha256":_digest(source_rig_info["bones"]),
            "fps":source_motion["fps"],
            "frames":len(source_motion["euler"])
        },
        "target":{
            "hierarchy_sha256":skeleton_hash(target_skeleton),
            "unposed_world_matrices_sha256":prepared["pipeline"][
                "target_unposed_world_sha256"],
            "reference_kind":target_skeleton["reference"]["kind"],
            "imported_scale":profile_report["source_to_target_uniform_scale"],
            "bone_count":len(target_skeleton["bones"]),
        },
        "conversion":{
            "basis":"parent_relative_rest_quaternion_conjugation",
            "root_rotation":"pelvis_rest_quaternion_conjugation",
            "root_translation":"pelvis_local",
            "root_position_scale":REFERENCE_ROOT_POSITION_MULTIPLIER,
            "full_bone_count":REFERENCE_SOURCE_BONES,
            "calibrated_for_other_skeletons":False
        },
        "files":{
            OUTPUT_MOTION:_digest(prepared),
            OUTPUT_AUDIT:_digest(audit)
        },
        "observed_evidence":{
            "Dragon_41_nonroot_FK":"user_visual_comparison_pass",
            "Dragon_pelvis_rotation":"user_visual_comparison_pass",
            "Dragon_pelvis_translation":"sampled_frames_0_31_62_93_123_numeric_pass",
            "per_frame_iClone_pose_evaluation":"NOT_PERFORMED",
            "other_avatars_or_rigs":"NOT_VERIFIED",
        },
        "notes":"Reference-only Dragon same-rig profile. FK JSON is experimental "
                "and the iClone Writer clean-scene safety gate remains REQUIRED."
    }
    return prepared,audit,manifest


def verify_fk_manifest_for_writer(motion_path,motion=None):
    """Check selected bundled FK JSON against adjacent manifest *before write*.

    This proves that the selected imported FK file is exactly the package
    payload prepared by the offline converter. The independent verify CLI
    additionally compares against the original motion and skeleton files.
    Original legacy experimental FK previews (no pipeline field) are not
    subject to the bundle sidecar and still run under the scene guard.
    """
    path=pathlib.Path(motion_path)
    current=_json(path) if motion is None else motion
    meta=current.get("pipeline")
    if meta is None:
        return {"status":"legacy_FK_json_no_bundle_manifest",
                "note":"Use import_pipeline prepare for source provenance."}
    if not isinstance(meta,dict) or meta.get("schema_version")!=BUNDLE_VERSION:
        raise ContractError("unknown embedded FK import pipeline metadata")
    if path.name!=OUTPUT_MOTION:
        raise ContractError(
            "bundled FK file must keep its original filename "+OUTPUT_MOTION)
    manifest_path=path.parent/OUTPUT_MANIFEST
    if not manifest_path.is_file():
        raise ContractError(
            "bundled FK JSON requires adjacent import_manifest.json")
    manifest=_json(manifest_path)
    if (manifest.get("schema_version")!=BUNDLE_VERSION or
        manifest.get("profile_id")!=PROFILE_ID or
        meta.get("profile_id")!=PROFILE_ID):
        raise ContractError("FK payload has foreign or mismatched profile")
    if _digest(current)!=manifest.get("files",{}).get(OUTPUT_MOTION):
        raise ContractError("FK payload differs from import_manifest.json SHA-256")
    if (meta.get("source_motion_sha256")!=manifest.get("source",{}).get(
            "motion_canonical_sha256") or
        meta.get("source_dragon_bones_sha256")!=manifest.get("source",{}).get(
            "rig_bones_canonical_sha256") or
        meta.get("target_skeleton_hash")!=manifest.get("target",{}).get(
            "hierarchy_sha256") or
        meta.get("target_unposed_world_sha256")!=manifest.get("target",{}).get(
            "unposed_world_matrices_sha256")):
        raise ContractError("embedded FK metadata differs from bundle manifest")
    if (current.get("calibration_status")!=
            "unverified_rest_basis_conjugation_preview" or
        current.get("root_motion",{}).get("root_position_space")!="pelvis_local" or
        current.get("root_motion",{}).get("root_position_scale")!=1.0):
        raise ContractError("bundled FK payload differs from Dragon reference profile")
    return {
        "status":"fk_payload_matches_adjacent_manifest",
        "profile_id":PROFILE_ID,
        "motion_canonical_sha256":_digest(current),
        "source_motion_sha256":meta["source_motion_sha256"],
        "note":"Package hash checked. Original source still requires external "
               "import_pipeline verify. iClone pose still requires SDK testing."
    }


def save_bundle(output_dir,source,target,source_info,reference_source,
                overwrite=False):
    """Write 3 mutually consistent files; refuse accidental overwrites."""
    motion,audit,manifest=build_bundle(source,target,source_info,reference_source)
    folder=pathlib.Path(output_dir)
    folder.parent.mkdir(parents=True,exist_ok=True)
    if folder.exists() and not overwrite:
        raise ContractError("package directory exists; use a new --output path")
    scratch=pathlib.Path(tempfile.mkdtemp(
        prefix=".anymove_import_",dir=str(folder.parent)))
    try:
        _write_json(scratch/OUTPUT_MOTION,motion)
        _write_json(scratch/OUTPUT_AUDIT,audit)
        _write_json(scratch/OUTPUT_MANIFEST,manifest)
        if folder.exists():
            if not folder.is_dir():
                raise ContractError("output path exists and is not a directory")
            # An explicit --overwrite only replaces known generated files
            # and refuses a folder with unrelated user data.
            allowed={OUTPUT_MOTION,OUTPUT_AUDIT,OUTPUT_MANIFEST}
            if {f.name for f in folder.iterdir()}-allowed:
                raise ContractError("refusing to overwrite folder with unrelated files")
            backup=folder.with_name(folder.name+".old")
            if backup.exists():
                raise ContractError("backup .old already exists; refusing overwrite")
            os.replace(str(folder),str(backup))
            try:
                os.replace(str(scratch),str(folder))
            except Exception:
                os.replace(str(backup),str(folder))
                raise
            shutil.rmtree(str(backup))
        else:
            os.replace(str(scratch),str(folder))
    finally:
        if scratch.exists():
            shutil.rmtree(str(scratch))
    return manifest


def verify_bundle(package_dir,source,target,source_info,reference_source):
    folder=pathlib.Path(package_dir)
    manifest=_json(folder/OUTPUT_MANIFEST)
    if manifest.get("schema_version")!=BUNDLE_VERSION or (
        manifest.get("profile_id")!=PROFILE_ID):
        raise ContractError("unsupported/foreign import manifest profile")
    loaded_motion=_json(folder/OUTPUT_MOTION)
    loaded_audit=_json(folder/OUTPUT_AUDIT)
    if _digest(loaded_motion)!=manifest["files"][OUTPUT_MOTION]:
        raise ContractError("Motion FK payload hash mismatch: bundle modified")
    if _digest(loaded_audit)!=manifest["files"][OUTPUT_AUDIT]:
        raise ContractError("audit hash mismatch: bundle modified")
    expected_motion,expected_audit,expected_manifest=build_bundle(
        source,target,source_info,reference_source)
    if _digest(loaded_motion)!=_digest(expected_motion):
        raise ContractError("FK payload does not match current input source/profile")
    if _digest(loaded_audit)!=_digest(expected_audit):
        raise ContractError("full-frame audit does not match input source/profile")
    if _digest(manifest)!=_digest(expected_manifest):
        raise ContractError("manifest/source/skeleton differs from current inputs")
    return {"status":"import_bundle_verified_for_same_dragon_rig",
            "frames":loaded_audit["frames_checked"],
            "rotation_samples_checked":loaded_audit["rotation_samples_checked"],
            "root_position_samples_checked":loaded_audit["root_position_samples_checked"],
            "maximum_quaternion_error_deg":loaded_audit[
                "maximum_quaternion_error_deg"],
            "runtime_validation":"still_requires_fresh_iClone_scene_test"}


def main():
    parser=argparse.ArgumentParser(description="Repeatable, reference-only Dragon -> iClone full FK import")
    cmd=parser.add_subparsers(dest="command")
    for sub in ("prepare","verify"):
        p=cmd.add_parser(sub)
        p.add_argument("--motion",required=True,help="original AnyMove motion.json")
        p.add_argument("--skeleton",required=True,help="unposed iClone Dragon skeleton.json")
        p.add_argument("--source-rig",default=str(DEFAULT_SOURCE))
        p.add_argument("--output",required=True,help="bundle folder containing 3 JSON files")
        if sub=="prepare":
            p.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    if args.command not in ("prepare","verify"):
        parser.error("specify prepare or verify")
    reference=_json(DEFAULT_SOURCE)
    source_info=_json(args.source_rig)
    source=_json(args.motion)
    target=_json(args.skeleton)
    if args.command=="prepare":
        manifest=save_bundle(args.output,source,target,source_info,
                             reference,overwrite=args.overwrite)
        print("Bundle saved:",args.output)
        print("FK Motion:",str(pathlib.Path(args.output)/OUTPUT_MOTION))
        print("Offline frames:",manifest["source"]["frames"],
              "BONES:",manifest["conversion"]["full_bone_count"])
        print("Status:",manifest["status"])
        print("IMPORTANT: Import FK JSON only into an unposed fresh iClone Dragon scene.")
    else:
        print(json.dumps(verify_bundle(args.output,source,target,source_info,
                                       reference),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
