"""Compare iClone manually sampled pelvis movement to Blender's hips path.

No Blender, RLPy or third-party packages are required. Expected Blender data:
blender/apply_motion.py out_dir/joints.json with names including 'hips'.
Observed iClone data: root_translation_sampler.save_samples() output.

Both paths are converted to offsets from Frame 0 (eliminating actor placement).
A single best-fit uniform scale diagnoses trajectory direction/shape independent
of FBX unit scaling. Set --expected-scale 100 for the known Dragon 100x FBX
case to also test actual displacement magnitude; 100 is a user-supplied
hypothesis, NOT auto-claimed as the correct Layer scale.
"""
import argparse
import json
import math
import pathlib

from .root_translation_sampler import VERSION
from .schema import ContractError


def _vector(row):
    if not isinstance(row,list) or len(row)!=3 or any(
            type(x) not in (float,int) or not math.isfinite(x) for x in row):
        raise ContractError("expected finite 3-vector")
    return [float(x) for x in row]


def _delta(a,b):
    return [x-y for x,y in zip(a,b)]


def _dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def _norm(a):
    return math.sqrt(_dot(a,a))


def _angle(a,b):
    l=_norm(a)*_norm(b)
    if l<1e-12:
        return None
    return math.degrees(math.acos(max(-1.,min(1.,_dot(a,b)/l))))


def compare_root_trajectory(blender,samples,expected_scale=None):
    if samples.get("schema_version")!=VERSION:
        raise ContractError("not an iClone manually sampled root JSON")
    if blender.get("fps")!=samples.get("fps"):
        raise ContractError("Blender/iClone project FPS differs")
    names=blender.get("names")
    if not isinstance(names,list) or names.count("hips")!=1:
        raise ContractError("Blender joints.json requires a single 'hips' keypoint")
    idx=names.index("hips")
    all_joints=blender.get("joints")
    if not isinstance(all_joints,list):
        raise ContractError("invalid Blender joints frames")
    observation=samples.get("samples")
    if not isinstance(observation,list) or len(observation)<3:
        raise ContractError("need at least 3 manually sampled iClone frames")
    keyed={}
    for row in observation:
        frame=row.get("frame")
        if type(frame) is not int or frame<0 or frame>=len(all_joints) or frame in keyed:
            raise ContractError("invalid/duplicate iClone frame index")
        keyed[frame]=_vector(row.get("original_axis_local_xyz"))
    if 0 not in keyed:
        raise ContractError("iClone sample set must contain Frame 0")
    if expected_scale is not None and (
            type(expected_scale) not in (int,float) or
            not math.isfinite(expected_scale) or expected_scale<=0):
        raise ContractError("expected_scale must be finite positive number")
    observed0=keyed[0]
    blender0=_vector(all_joints[0][idx])
    samples_out=[]
    for frame in sorted(keyed):
        observed=_delta(keyed[frame],observed0)
        reference=_delta(_vector(all_joints[frame][idx]),blender0)
        samples_out.append({
            "frame":frame,
            "blender_hips_delta":reference,
            "iclone_pelvis_delta_native":observed,
            "blender_magnitude":_norm(reference),
            "iclone_magnitude_native":_norm(observed),
            "direction_angle_deg":_angle(reference,observed),
        })
    all_reference=[row["blender_hips_delta"] for row in samples_out]
    all_observed=[row["iclone_pelvis_delta_native"] for row in samples_out]
    denom=sum(_dot(v,v) for v in all_reference)
    if denom<1e-10:
        raise ContractError("Blender hips is nearly static; cannot calibrate trajectory")
    fit=sum(_dot(a,b) for a,b in zip(all_reference,all_observed))/denom
    sum_obs=sum(_dot(v,v) for v in all_observed)
    if sum_obs<1e-8:
        classification="iclone_pelvis_translation_static"
    elif fit<=0:
        classification="opposite_or_incompatible_travel_direction"
    else:
        classification="trajectory_measurable"
    rms=math.sqrt(sum(_dot(_delta(a,[fit*x for x in b]),
                            _delta(a,[fit*x for x in b]))
                      for a,b in zip(all_observed,all_reference))/
                  max(1,len(samples_out)-1))
    normalized=rms/(abs(fit)*math.sqrt(denom/max(1,len(samples_out)-1))) if fit>1e-8 else None
    angles=[r["direction_angle_deg"] for r in samples_out
            if r["direction_angle_deg"] is not None and
            r["blender_magnitude"]>1e-5]
    expected_err=None
    if expected_scale is not None:
        expected_err=math.sqrt(sum(
            _dot(_delta(a,[expected_scale*x for x in b]),
                 _delta(a,[expected_scale*x for x in b]))
            for a,b in zip(all_observed,all_reference))/
            max(1,len(samples_out)-1))
    for row in samples_out:
        row["fitted_scale_predicted_xyz"]=[
            fit*v for v in row["blender_hips_delta"]]
        row["fitted_scale_error_native"]= _norm(_delta(
            row["iclone_pelvis_delta_native"],
            row["fitted_scale_predicted_xyz"]))
    if classification=="trajectory_measurable":
        if normalized is not None and normalized<=.05 and (not angles or max(angles)<=5):
            classification="direction_and_path_match_after_uniform_scaling"
        else:
            classification="direction_or_path_mismatch"
    return {
        "result":classification,
        "frame_count":len(samples_out),
        "samples":samples_out,
        "best_fit_native_units_per_blender_unit":fit,
        "path_shape_relative_rms_error":normalized,
        "max_direction_angle_deg":max(angles) if angles else None,
        "mean_direction_angle_deg":sum(angles)/len(angles) if angles else None,
        "expected_native_units_per_blender_unit":(
            float(expected_scale) if expected_scale is not None else None),
        "rms_error_at_expected_scale_native_units":expected_err,
        "suggested_position_scale_multiplier":(
            expected_scale/fit if expected_scale is not None and fit>1e-8
            else None),
        "warning":"A scale fit can hide magnitude error; target expected scale must come from independently verified FBX units. Compare clean, same-frame reference scenes. RINode/SDK pose reads are experimental.",
    }


def main():
    p=argparse.ArgumentParser(description="Compare Blender joints.json hips travel against manual iClone pelvis samples")
    p.add_argument("blender_joints_json")
    p.add_argument("iclone_root_samples_json")
    p.add_argument("--expected-scale",type=float,default=None,
                   help="e.g. 100 for observed Dragon FBX scene-scale (verify before relying on it)")
    p.add_argument("--report-out",default=None)
    args=p.parse_args()
    with open(args.blender_joints_json,encoding="utf-8") as f:
        blender=json.load(f)
    with open(args.iclone_root_samples_json,encoding="utf-8") as f:
        samples=json.load(f)
    result=compare_root_trajectory(blender,samples,
                                   expected_scale=args.expected_scale)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.report_out:
        path=pathlib.Path(args.report_out)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding="utf-8")
        print("Saved:",path)
    print("Result:",result["result"])
    print("Best fit unit ratio:",result["best_fit_native_units_per_blender_unit"])
    print("Path shape RMS error:",result["path_shape_relative_rms_error"])
    print("Maximum direction error:",result["max_direction_angle_deg"])
    print("Scale correction factor:",result["suggested_position_scale_multiplier"])


if __name__=="__main__":
    main()
