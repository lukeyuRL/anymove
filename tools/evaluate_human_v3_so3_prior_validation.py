"""Controlled validation of frozen V3 Spine SO(3) prior vs baseline FK.

Unlike the old already-inspected Reader TEST, input is entirely new
rotation-diverse VALIDATION GT under 315/45 calibrated orthographic cameras,
with deterministic 1px Gaussian observation noise. Test clips are NEVER read.

Compares fixed 200-iteration Solver with or without the opt-in SO(3)
prior. Metrics: 67-marker 3D MPJPE, Root world orientation, local Spine
orientation, mean angular acceleration and time displacement error.

This is a synthetic controlled Solver ablation, NOT actual Reader inference
on natural videos.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import zlib

import numpy as np
from scipy.spatial.transform import Rotation
from mocap.camera_geometry import project_orthographic

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
DATA=ROOT/"spine_motion_prior_v1"
OUT=ROOT/"solver/spine_so3_research/controlled_validation"
RIG_BASE=ROOT/"solver/rigged/rig_info_trainlimits.json"
RIG_SO3=ROOT/"solver/spine_so3_research/rigged/rig_info.json"
SRC_SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"
M_PER_PX=2.432802200317383/672
Z_CENTER=.8688581585884094
SEED=20261104
FAMILIES=("pivot_turn_left","walk_turn_right",
          "waist_twist_right","reach_counterrotate")
SPINE=("CC_Base_Pelvis","CC_Base_Waist","CC_Base_Spine01",
       "CC_Base_Spine02","CC_Base_NeckTwist01",
       "CC_Base_NeckTwist02","CC_Base_Head")


def magnitude_stats(values,unit=""):
    v=np.asarray(values,dtype=np.float64)
    return {
        "n":int(v.size),"mean"+unit:float(v.mean()),
        "p95"+unit:float(np.percentile(v,95)),
        "max"+unit:float(v.max()),
    }


def root_world_rotation(solver_root_wxyz,rest_basis):
    """Rig Root rotation applies AFTER root bone rest basis, so conjugate."""
    q=np.asarray(solver_root_wxyz,dtype=float)
    local=Rotation.from_quat(q[:,[1,2,3,0]])
    world_R=rest_basis@local.as_matrix()@rest_basis.T
    return Rotation.from_matrix(world_R)


def get_bone_rest_rotation(rig):
    matrix=np.asarray(rig["bones"][0]["matrix_local"],dtype=float)[:3,:3]
    U,S,Vt=np.linalg.svd(matrix)
    R=U@np.diag([1,1,np.sign(np.linalg.det(U@Vt))])@Vt
    return R


def angular_metrics(euler,gt):
    r=Rotation.from_euler("xyz",euler)
    gr=Rotation.from_euler("xyz",gt)
    de=np.rad2deg((gr.inv()*r).magnitude())
    vel=(r[:-1].inv()*r[1:]).as_rotvec()
    acc=np.linalg.norm(np.diff(vel,axis=0),axis=-1)
    gtvel=(gr[:-1].inv()*gr[1:]).as_rotvec()
    gtacc=np.linalg.norm(np.diff(gtvel,axis=0),axis=-1)
    return {
        "local_orientation_error_deg":magnitude_stats(de),
        "local_angular_acceleration_deg_per_frame2":magnitude_stats(np.rad2deg(acc)),
        "gt_angular_acceleration_deg_per_frame2":magnitude_stats(np.rad2deg(gtacc)),
    }


def validate_one(row,rig_names,marker_names,rig_rest,iters):
    clip=row["clip"]
    if row["split"]!="validation":
        raise RuntimeError("Cannot evaluate or tune on TEST GT")
    with np.load(DATA/row["npz"],allow_pickle=False) as gt:
        t=np.asarray(gt["frames"],dtype=np.int32)
        xyz=np.asarray(gt["world_joints_m"],dtype=np.float64)
        eul=np.asarray(gt["bone_euler_xyz_rad"],dtype=np.float64)
        q=np.asarray(gt["root_rotation_wxyz"],dtype=np.float64)
    if t.shape!=(107,) or not np.array_equal(t,np.arange(107)):
        raise RuntimeError("Invalid root-motion validation source frames")
    rng=np.random.default_rng(SEED+int(zlib.crc32(clip.encode("utf8"))))
    front=project_orthographic(xyz,315,672,672,M_PER_PX,Z_CENTER)
    side=project_orthographic(xyz,45,672,672,M_PER_PX,Z_CENTER)
    f_noisy=front+rng.normal(0,1.,front.shape)
    s_noisy=side+rng.normal(0,1.,side.shape)
    OUT.mkdir(parents=True,exist_ok=True)
    det=OUT/(clip+".controlled_1px.npz")
    if not det.exists():
        np.savez_compressed(det,front_uv=f_noisy,side_uv=s_noisy,
            front_conf=np.ones((107,67),dtype=np.float32),
            side_conf=np.ones((107,67),dtype=np.float32),
            fps=np.float64(24),frame_indices=t,camera_azimuth_deg=np.array([315,45],dtype=float),
            names=np.asarray(marker_names))
    scores={}
    for mode,rigfile,creature in ((
        "baseline",RIG_BASE,ROOT/"solver"),(
        "train_so3_prior",RIG_SO3,ROOT/"solver/spine_so3_research"),(
        "root_anchor_so3",ROOT/"solver/spine_so3_root_research/rigged/rig_info.json",
                         ROOT/"solver/spine_so3_root_research"),(
        "root_evidence_so3",ROOT/"solver/spine_so3_root_geom_evidence_research/rigged/rig_info.json",
                         ROOT/"solver/spine_so3_root_geom_evidence_research"),(
        "root_evidence_strong_so3",ROOT/"solver/spine_so3_root_geom_evidence_strong_research/rigged/rig_info.json",
                         ROOT/"solver/spine_so3_root_geom_evidence_strong_research")):
        target=OUT/(clip+"."+mode+".motion.json")
        log=OUT/(clip+"."+mode+".log")
        if not target.exists():
            env=os.environ.copy()
            env["MOCAP_CREATURE"]=str(creature.resolve())
            env["MOCAP_CAMERA_AZIMUTHS"]="315,45"
            cmd=[sys.executable,"-m","mocap.solve",str(det),str(rigfile),str(target),
                 "--no-align","--iters",str(iters),"--allow-provisional-human-limits"]
            start=time.monotonic()
            p=subprocess.run(cmd,env=env,cwd=pathlib.Path.cwd(),
                             capture_output=True,text=True,
                             timeout=180,encoding="utf8",errors="replace")
            log.write_text(p.stdout+"\nSTDERR:\n"+p.stderr,encoding="utf8")
            if p.returncode:
                raise RuntimeError("SO3 prior study failed "+clip+"/"+mode+
                                    " code="+str(p.returncode)+" "+p.stderr[-1200:])
            print("SO3_SOLVER_COMPLETED",clip,mode,"seconds",round(time.monotonic()-start,1),flush=True)
        prediction=json.loads(target.read_text(encoding="utf8"))
        if prediction["joint_names"]!=marker_names or prediction["bones"]!=rig_names:
            raise RuntimeError("Solver changed V3 bone/marker identities")
        if prediction["source_frame_indices"]!=t.tolist() or prediction.get("source_frames_contiguous") is not True:
            raise RuntimeError("Output contains missing/non-source time")
        positions=np.asarray(prediction["joints"],dtype=float)
        rotations=np.asarray(prediction["euler"],dtype=float)
        rq=np.asarray(prediction["root_quat"],dtype=float)
        if positions.shape!=(107,67,3) or rotations.shape!=(107,55,3):
            raise RuntimeError("Incomplete solver animation")
        de=np.linalg.norm(positions-xyz,axis=-1)*100
        temporal=np.linalg.norm(np.diff(positions,axis=0)-np.diff(xyz,axis=0),axis=-1)*100
        local={}
        for bone in SPINE:
            index=rig_names.index(bone)
            local[bone]=angular_metrics(rotations[:,index],eul[:,index])
        rootpred=root_world_rotation(rq,rig_rest)
        rootgt=Rotation.from_quat(q[:,[1,2,3,0]])
        err=np.rad2deg((rootgt.inv()*rootpred).magnitude())
        root_acc=np.linalg.norm(np.diff(
            (rootpred[:-1].inv()*rootpred[1:]).as_rotvec(),axis=0),axis=-1)
        scores[mode]={
            "3d_mpjpe_cm":magnitude_stats(de),
            "3d_temporal_error_cm":magnitude_stats(temporal),
            "spine_bone_stats":local,
            "root_rotation_world_error_deg":magnitude_stats(err),
            "root_rotation_angular_acc_deg_per_frame2":magnitude_stats(np.rad2deg(root_acc)),
            "root_gt_turn_range_deg":row["root_yaw_range_deg"],
            "reprojection":prediction["reproj_px"],
            "motion_file":str(target),
        }
    print("SO3_CLIP_AB",clip,json.dumps({
        k:{"mpjpe":v["3d_mpjpe_cm"]["mean"],
            "waist_acc":v["spine_bone_stats"]["CC_Base_Waist"]["local_angular_acceleration_deg_per_frame2"]["mean"],
            "root_error_deg":v["root_rotation_world_error_deg"]["mean"],
            "root_ang_acc":v["root_rotation_angular_acc_deg_per_frame2"]["mean"]}
        for k,v in scores.items()}),flush=True)
    return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--families",nargs="*",default=list(FAMILIES))
    p.add_argument("--iterations",type=int,default=200)
    a=p.parse_args()
    if not 1<=a.iterations<=1500:
        raise ValueError("Invalid evaluation iteration count")
    manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf8"))
    audit=json.loads((DATA/"audit.json").read_text(encoding="utf8"))
    if audit["status"]!="PASS" or manifest["status"]!="PASS" or manifest["source_blend_sha256"]!=SRC_SHA:
        raise RuntimeError("Motion dataset not audited")
    rig=json.loads(RIG_BASE.read_text(encoding="utf8"))
    rig_names=[b["name"] for b in rig["bones"]]
    names=[k["name"] for k in rig["keypoints"]]
    if manifest["source_rig_bones"]!=rig_names or manifest["tracked_markers"]!=names:
        raise RuntimeError("Rig/GT marker contract drift")
    rest=get_bone_rest_rotation(rig)
    rows=[]
    for family in a.families:
        eligible=[r for r in manifest["rows"] if r["family"]==family and r["split"]=="validation"]
        if len(eligible)!=2:
            raise RuntimeError("Validation must provide exactly two takes per family "+family)
        rows.append(eligible[0])
    report={"schema":"anymove.human_v3_so3_prior_controlled_validation/1",
            "status":"VALIDATION_AB_RUNNING",
            "source_sha":SRC_SHA,
            "test_clips_untouched":24,
            "input":"GT Blender 3D projected to 315/45 with deterministic 1px Gaussian noise",
            "setting":"two variants, same 200 optimization iterations, NO Reader model or H3 video",
            "validation_clips":[],"results":{}}
    for row in rows:
        result=validate_one(row,rig_names,names,rest,a.iterations)
        report["validation_clips"].append(row["clip"])
        report["results"][row["clip"]]=result
        outfile=OUT/"validation_ab.json"
        outfile.write_text(json.dumps(report,indent=2),encoding="utf8")
    report["status"]="VALIDATION_AB_COMPLETE_NOT_UNSEEN_TEST"
    (OUT/"validation_ab.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("SO3_CONTROLLED_VALIDATION_COMPLETE",json.dumps({
        "clips":report["validation_clips"],
        "frames":len(rows)*107,
        "report":str(OUT/"validation_ab.json")}),flush=True)


if __name__=="__main__":
    main()
