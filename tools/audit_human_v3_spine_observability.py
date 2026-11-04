"""Local 315/45 2D observability audit for Human V3 Spine rotations.

For already examined *Validation* Solver poses only, finite-difference
the projected 67-marker 2-view observations with respect to 3 global root
angles and 4x3 local pelvis/waist/spine01/spine02 Euler angles.

This identifies position/rotation nullspace and parameter correlations:
low image loss can coexist with large local bone-rotation jerk.
No training, threshold tuning, Test GT, iClone or image generation.
"""
import json
import os
import pathlib
from collections import defaultdict

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
os.environ["MOCAP_CREATURE"]=str((ROOT/"solver").resolve())
os.environ["MOCAP_CAMERA_AZIMUTHS"]="315,45"

from mocap import solve

BASE=ROOT/"solver/spine_so3_research/multifamily_validation"
INPUT=BASE/"multifamily_1takes_400iter_validation.json"
RIG=ROOT/"solver/rigged/rig_info_trainlimits.json"
TARGET_BONES=("CC_Base_Pelvis","CC_Base_Waist","CC_Base_Spine01","CC_Base_Spine02")
FRAME_INDICES=(17,53,88)
EPS=1e-4


def finite_difference(clip,rig,params):
    pred=np.asarray(clip["joints"],dtype=np.float64)
    euler=np.asarray(clip["euler"],dtype=np.float64)
    trans=np.asarray(clip["trans"],dtype=np.float64)
    root_q=np.asarray(clip["root_quat"],dtype=np.float64)
    out=[]
    for frame in FRAME_INDICES:
        q=Rotation.from_quat(root_q[frame,[1,2,3,0]])
        u=euler[frame].copy()
        position=trans[frame].copy()

        def projected(R,eul):
            with torch.no_grad():
                world,_=rig.forward(torch.as_tensor(position[None],dtype=torch.float64),
                    torch.as_tensor(R.as_matrix()[None],dtype=torch.float64),
                    torch.as_tensor(eul[None],dtype=torch.float64))
                a,b=solve.project(world)
                return np.concatenate((a.cpu().numpy().ravel(),b.cpu().numpy().ravel()))
        base_uv=projected(q,u)
        jac=[]
        labels=[]
        for index in range(3):
            delta=np.zeros(3);delta[index]=EPS
            plus=projected(q*Rotation.from_rotvec(delta),u)
            minus=projected(q*Rotation.from_rotvec(-delta),u)
            jac.append((plus-minus)/(2*EPS))
            labels.append("OBJECT_ROOT."+"XYZ"[index])
        for name in TARGET_BONES:
            b=rig.names.index(name)
            for axis in range(3):
                pos=u.copy();neg=u.copy()
                pos[b,axis]+=EPS;neg[b,axis]-=EPS
                jac.append((projected(q,pos)-projected(q,neg))/(2*EPS))
                labels.append(name+"."+"XYZ"[axis])
        J=np.asarray(jac).T
        if J.shape!=(268,15) or not np.isfinite(J).all():
            raise RuntimeError("Unexpected 2-view projection Jacobian shape")
        sv=np.linalg.svd(J,compute_uv=False)
        col=np.linalg.norm(J,axis=0)*np.deg2rad(1)
        weak=np.argsort(col)[:6].tolist()
        cond=float(sv[0]/max(sv[-1],1e-12))
        # Joint parameter CRLB, assuming all synthetic per-pixel errors are
        # independent Gaussian ~1px (optimistic; no occlusion modeled).
        covariance=np.linalg.pinv(J.T@J,rcond=1e-12)
        uncertainty=np.rad2deg(np.sqrt(np.maximum(np.diag(covariance),0)))
        out.append({
            "frame":frame,"jacobian_shape":[268,15],
            "singular_values_px_per_radian":sv.round(7).tolist(),
            "condition_number":cond,
            "smallest_singular_value_px_per_radian":float(sv[-1]),
            "weakest_1deg_rotational_pixel_changes":[
                {"dof":labels[k],"norm_px":float(col[k])} for k in weak
            ],
            "optimistic_1px_noise_independent_parameter_std_deg":{
                name:float(val) for name,val in zip(labels,uncertainty)
            },
            "max_optimal_uncertainty_deg":float(uncertainty.max()),
            "caution":"Local Jacobian, single pose; bound assumes ideal independent 1px pixel errors at all 67 visible joints."
        })
    return out


def main():
    report=json.loads(INPUT.read_text(encoding="utf8"))
    if (report["status"]!="VALIDATION_COMPLETE_TEST_24_CLIPS_UNTOUCHED"
        or report["test_clips_loaded"]!=0 or report["iterations_per_mode"]!=400):
        raise RuntimeError("Requires frozen four-family Validation-only A/B")
    info=json.loads(RIG.read_text(encoding="utf8"))
    rig=solve.Rig(info,"cpu")
    for target in TARGET_BONES:
        if target not in rig.names:raise RuntimeError("Spine missing from rig")
    all_records={}
    for clip,row in report["scores_per_clip"].items():
        if row["source_split"]!="validation":raise RuntimeError("Potential Test contamination")
        path=BASE/(clip+".baseline.400iter.motion.json")
        solved=json.loads(path.read_text(encoding="utf8"))
        if (solved.get("root_bone_static") is not True
            or np.abs(np.asarray(solved["euler"])[:,0]).max()>1e-8):
            raise RuntimeError("Source Root Bone is animated")
        rows=finite_difference(solved,rig,TARGET_BONES)
        all_records[clip]={"family":row["family"],"per_frame":rows}
        print("SPINE_OBSERVABILITY",row["family"],
              "minimum_singular_values",[round(x["smallest_singular_value_px_per_radian"],5) for x in rows],
              "largest_cond_numbers",round(max(x["condition_number"] for x in rows),1),
              "highest_optimistic_axis_uncertainty_deg",round(max(x["max_optimal_uncertainty_deg"] for x in rows),3),
              flush=True)
    audit={
        "schema":"anymove.human_v3_spine_local_two_view_jacobian/1",
        "status":"VALIDATION_ONLY_NUMERICAL_OBSERVABILITY",
        "family_count":len(all_records),
        "frames_per_family":len(FRAME_INDICES),
        "perturbation_radians":EPS,
        "camera_angles":[315,45],
        "unseen_Test_loaded":0,
        "root_bone_static":True,
        "record":all_records,
        "limitations":"Linearized local observability, all markers assumed directly visible, optimistic 1px independent noise; not motion/retarget quality acceptance.",
    }
    path=BASE/"spine_jacobian_observability.json"
    path.write_text(json.dumps(audit,indent=2),encoding="utf8")
    print("SPINE_OBSERVABILITY_COMPLETE",json.dumps({"families":len(all_records),
        "frames":len(all_records)*len(FRAME_INDICES),"saved":str(path)}),flush=True)


if __name__=="__main__":
    torch.set_num_threads(4)
    main()
