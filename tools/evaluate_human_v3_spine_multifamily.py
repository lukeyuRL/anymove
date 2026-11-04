"""Strict multi-family Human V3 Spine/Root Validation: matched FK Solver A/B/C.

Validation-only: two potential takes per family exist, this run evaluates
preselected take 8 for each of four families. The 24 heldout Test clips
are NEVER read. Uses the same audited 107-frame Blender GT, calibrated
315/45 orthographic projection and fixed deterministic 1.0-px observation
noise; no Reader image inference, neural training or iClone import.

Three variants share the exact same input observations, 400 optimization
iterations, optimizer and camera geometry:
  baseline = broad Train-derived joint limits
  root_so3 = 96 Train-clip SO(3) prior + Kabsch Root observation
  neutral = same prior + Train median Spine local orientation initialization
Root world orientation is carried by the Armature Object; RL_BoneRoot local
pose is NEVER keyed and must remain all-zero in Solver JSON.

Metrics include location, per-bone SO(3) angular acceleration, Root
orientation and validation-contact-proxy foot velocity.
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
from tools.evaluate_human_v3_so3_prior_validation import get_bone_rest_rotation

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
DATA=ROOT/"spine_motion_prior_v1"
OUT=ROOT/"solver/spine_so3_research/multifamily_validation"
RIG_BASE=ROOT/"solver/rigged/rig_info_trainlimits.json"
RIG_ROOT=ROOT/"solver/spine_so3_root_geom_evidence_research/rigged/rig_info.json"
RIG_NEUTRAL=ROOT/"solver/spine_so3_train_neutral_init_research/rigged/rig_info.json"
V3_SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"
FAMILIES=("walk_turn_right","waist_twist_right","lateral_bend_left","reach_counterrotate")
MODES=(
    ("baseline",RIG_BASE,ROOT/"solver"),
    ("root_so3",RIG_ROOT,ROOT/"solver/spine_so3_root_geom_evidence_research"),
    ("neutral_so3",RIG_NEUTRAL,ROOT/"solver/spine_so3_train_neutral_init_research"),
)
SPINE=("CC_Base_Pelvis","CC_Base_Waist","CC_Base_Spine01",
       "CC_Base_Spine02","CC_Base_NeckTwist01","CC_Base_NeckTwist02","CC_Base_Head")
METER_PX=2.432802200317383/672
Z_CENTER=.8688581585884094
SEED=20261104


def metric(a):
    values=np.asarray(a,dtype=np.float64).reshape(-1)
    if values.size==0:raise RuntimeError("Metric requires observations")
    return {"n":int(values.size),"mean":round(float(values.mean()),6),
            "p95":round(float(np.percentile(values,95)),6),
            "max":round(float(values.max()),6)}


def angular_motion(euler):
    rotation=Rotation.from_euler("xyz",euler)
    velocity=(rotation[:-1].inv()*rotation[1:]).as_rotvec()
    accel=np.linalg.norm(np.diff(velocity,axis=0),axis=-1)
    jerk=np.linalg.norm(np.diff(velocity,n=2,axis=0),axis=-1)
    return np.rad2deg(accel),np.rad2deg(jerk)


def score_one(motion,xyz,source_euler,root_quat,ground_proxy,rig_names,markers,rest_R,frame):
    if (motion.get("source_frame_indices")!=frame.tolist() or
        motion.get("source_frames_contiguous") is not True or
        motion.get("solver_camera_azimuth_deg")!=[315,45] or
        motion.get("joint_names")!=markers or motion.get("bones")!=rig_names or
        motion.get("root_bone_static") is not True or
        motion.get("root_motion_carrier")!="ArmatureObject"):
        raise RuntimeError("Solver violates 24FPS camera/root ownership contract")
    euler=np.asarray(motion["euler"],dtype=np.float64)
    pred=np.asarray(motion["joints"],dtype=np.float64)
    root_q=np.asarray(motion["root_quat"],dtype=np.float64)
    if (pred.shape!=(107,67,3) or euler.shape!=(107,55,3) or
        root_q.shape!=(107,4) or not np.isfinite(pred).all() or
        not np.isfinite(euler).all() or np.max(np.abs(euler[:,0,:]))>1e-8):
        raise RuntimeError("RL_BoneRoot animated, Solver motion incomplete or non-finite")
    positions=np.linalg.norm(pred-xyz,axis=-1)*100
    temporal=np.linalg.norm(np.diff(pred,axis=0)-np.diff(xyz,axis=0),axis=-1)*100
    root_expected=Rotation.from_quat(root_quat[:,[1,2,3,0]])
    root_local=Rotation.from_quat(root_q[:,[1,2,3,0]])
    root_world=Rotation.from_matrix(rest_R@root_local.as_matrix()@rest_R.T)
    root_orientation_error=np.rad2deg((root_expected.inv()*root_world).magnitude())
    root_speed=(root_world[:-1].inv()*root_world[1:]).as_rotvec()
    root_acceleration=np.rad2deg(np.linalg.norm(np.diff(root_speed,axis=0),axis=-1))
    root_gt_speed=(root_expected[:-1].inv()*root_expected[1:]).as_rotvec()
    root_gt_acceleration=np.rad2deg(np.linalg.norm(np.diff(root_gt_speed,axis=0),axis=-1))

    per_joint={}
    extra=[]
    all_angular=[]
    for bone in SPINE:
        bi=rig_names.index(bone)
        predicted_R=Rotation.from_euler("xyz",euler[:,bi])
        expected_R=Rotation.from_euler("xyz",source_euler[:,bi])
        relative=np.rad2deg((expected_R.inv()*predicted_R).magnitude())
        acc,jerk=angular_motion(euler[:,bi])
        gt_acc,gt_jerk=angular_motion(source_euler[:,bi])
        per_joint[bone]={
            "orientation_error_deg":metric(relative),
            "angular_acc_deg_frame2":metric(acc),
            "gt_angular_acc_deg_frame2":metric(gt_acc),
            "angular_jerk_deg_frame3":metric(jerk),
            "gt_angular_jerk_deg_frame3":metric(gt_jerk),
        }
        all_angular.extend(acc.tolist())
        extra.extend((acc-gt_acc).tolist())
    # This foot metric is evaluated against the Blender procedural proxy,
    # not physical contact. It measures solver drift only where the proxy
    # says the corresponding sole region should have low lateral movement.
    slide=[]
    for side_no,side in enumerate(("L","R")):
        foot=markers.index("CC_Base_"+side+"_Foot")
        toe=markers.index("CC_Base_"+side+"_ToeTip")
        anchor=(pred[:,foot]+pred[:,toe])/2
        speed=np.linalg.norm(np.diff(anchor[:,:2],axis=0),axis=-1)*100
        proxy=np.asarray(ground_proxy[1:,side_no],bool)
        slide.extend(speed[proxy].tolist())
    return {
        "body_mpjpe_cm":metric(positions),
        "body_temporal_error_cm":metric(temporal),
        "root_orientation_error_deg":metric(root_orientation_error),
        "root_angular_acc_deg_frame2":metric(root_acceleration),
        "root_gt_angular_acc_deg_frame2":metric(root_gt_acceleration),
        "spine_acc_mean_deg_frame2":round(float(np.mean(all_angular)),6),
        "spine_excess_acc_mean_deg_frame2":round(float(np.mean(extra)),6),
        "spine_bones":per_joint,
        "contact_proxy_foot_horizontal_speed_cm_per_frame":metric(slide) if slide else None,
        "reprojection_px":motion.get("reproj_px"),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--iterations",type=int,default=400)
    ap.add_argument("--takes",type=int,choices=(1,2),default=1)
    ap.add_argument("--families",nargs="*",default=list(FAMILIES))
    args=ap.parse_args()
    if not 1<=args.iterations<=2000:
        raise ValueError("Invalid iteration count")
    if len(set(args.families))!=len(args.families) or any(f not in FAMILIES for f in args.families):
        raise ValueError("Unapproved Validation action family")
    manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf8"))
    audit=json.loads((DATA/"audit.json").read_text(encoding="utf8"))
    if (manifest["status"]!="PASS" or audit["status"]!="PASS" or
        manifest["train_val_test_clips"]!={"train":96,"validation":24,"test":24} or
        manifest["source_blend_sha256"]!=V3_SHA):
        raise RuntimeError("Motion Prior dataset provenance not validated")
    rig=json.loads(RIG_BASE.read_text(encoding="utf8"))
    rig_names=[r["name"] for r in rig["bones"]]
    markers=[r["name"] for r in rig["keypoints"]]
    if (rig_names[0]!="RL_BoneRoot" or len(rig_names)!=55 or
        len(markers)!=67 or rig_names!=manifest["source_rig_bones"] or
        markers!=manifest["tracked_markers"]):
        raise RuntimeError("V3 rigid root, bone or marker schema drift")
    rest_R=get_bone_rest_rotation(rig)
    rows=[]
    for family in args.families:
        choices=sorted((r for r in manifest["rows"] if r["family"]==family and r["split"]=="validation"),
                       key=lambda r:r["seed"])
        if len(choices)!=2:
            raise RuntimeError("Exactly two isolated Validation takes required")
        rows.extend(choices[:args.takes])
    if len(rows)!=len(args.families)*args.takes or any("clips/validation/" not in row["npz"] for row in rows):
        raise RuntimeError("Test/Train split leakage")
    results={}
    OUT.mkdir(parents=True,exist_ok=True)
    for row in rows:
        clip=row["clip"]
        with np.load(DATA/row["npz"],allow_pickle=False) as gt:
            xyz=np.asarray(gt["world_joints_m"],dtype=np.float64)
            eul=np.asarray(gt["bone_euler_xyz_rad"],dtype=np.float64)
            quat=np.asarray(gt["root_rotation_wxyz"],dtype=np.float64)
            foot_proxy=np.asarray(gt["contact_proxy"],dtype=np.int32)
            times=np.asarray(gt["frames"],dtype=np.int32)
        if (xyz.shape!=(107,67,3) or eul.shape!=(107,55,3) or
            quat.shape!=(107,4) or not np.array_equal(times,np.arange(107)) or
            np.max(np.abs(eul[:,0,:]))>1e-8):
            raise RuntimeError("Input GT violates static RL_BoneRoot and source frames")
        front=project_orthographic(xyz,315,672,672,METER_PX,Z_CENTER)
        side=project_orthographic(xyz,45,672,672,METER_PX,Z_CENTER)
        rng=np.random.default_rng(SEED+zlib.crc32(clip.encode("utf8")))
        det_path=OUT/(clip+".controlled_1px.npz")
        if not det_path.exists():
            np.savez_compressed(det_path,
                 front_uv=front+rng.normal(0,1,front.shape),
                 side_uv=side+rng.normal(0,1,side.shape),
                 front_conf=np.ones((107,67),dtype=np.float32),
                 side_conf=np.ones((107,67),dtype=np.float32),
                 names=np.asarray(markers),
                 fps=np.float64(24),frame_indices=times,
                 camera_azimuth_deg=np.asarray([315.,45.]))
        sample={}
        for mode,rig_file,creature in MODES:
            motion_path=OUT/(clip+"."+mode+f".{args.iterations}iter.motion.json")
            log_path=OUT/(clip+"."+mode+f".{args.iterations}iter.log")
            if not motion_path.exists():
                env=os.environ.copy()
                env["MOCAP_CREATURE"]=str(creature.resolve())
                env["MOCAP_CAMERA_AZIMUTHS"]="315,45"
                cmd=[sys.executable,"-m","mocap.solve",
                     str(det_path),str(rig_file),str(motion_path),
                     "--no-align","--iters",str(args.iterations),
                     "--allow-provisional-human-limits"]
                t0=time.monotonic()
                p=subprocess.run(cmd,capture_output=True,text=True,env=env,
                                 cwd=pathlib.Path.cwd(),timeout=240,
                                 encoding="utf8",errors="replace")
                log_path.write_text(p.stdout+"\nSTDERR:\n"+p.stderr,encoding="utf8")
                if p.returncode:
                    raise RuntimeError("Solver failure on "+clip+"/"+mode+
                                       " exit="+str(p.returncode)+" "+p.stderr[-1200:])
                print("SO3_MULTIFAMILY_SOLVED",clip,mode,"elapsed_seconds",
                      round(time.monotonic()-t0,1),flush=True)
            prediction=json.loads(motion_path.read_text(encoding="utf8"))
            sample[mode]=score_one(prediction,xyz,eul,quat,foot_proxy,
                                    rig_names,markers,rest_R,times)
        results[clip]={"family":row["family"],"seed":row["seed"],
                       "source_split":"validation","metrics":sample}
        (OUT/"progress_validation.json").write_text(json.dumps(
            {"status":"RUNNING","iterations":args.iterations,
             "completed_clips":list(results),
             "results":results},indent=2),encoding="utf8")
        for mode in sample:
            m=sample[mode]
            print("SO3_MULTIFAMILY_METRIC",clip,mode,
                "mpjpe",m["body_mpjpe_cm"]["mean"],
                "root_error_deg",m["root_orientation_error_deg"]["mean"],
                "waist_acc",m["spine_bones"]["CC_Base_Waist"]["angular_acc_deg_frame2"]["mean"],
                "spine_acc",m["spine_acc_mean_deg_frame2"],flush=True)
    per_mode={}
    for mode,_,_ in MODES:
        tmp=[x["metrics"][mode] for x in results.values()]
        per_mode[mode]={
            "mean_3d_mpjpe_cm":float(np.mean([m["body_mpjpe_cm"]["mean"] for m in tmp])),
            "mean_3d_p95_cm":float(np.mean([m["body_mpjpe_cm"]["p95"] for m in tmp])),
            "mean_root_orientation_error_deg":float(np.mean([m["root_orientation_error_deg"]["mean"] for m in tmp])),
            "mean_waist_angular_acc_deg_frame2":float(np.mean([m["spine_bones"]["CC_Base_Waist"]["angular_acc_deg_frame2"]["mean"] for m in tmp])),
            "mean_pelvis_angular_acc_deg_frame2":float(np.mean([m["spine_bones"]["CC_Base_Pelvis"]["angular_acc_deg_frame2"]["mean"] for m in tmp])),
            "mean_spine_all_angular_acc_deg_frame2":float(np.mean([m["spine_acc_mean_deg_frame2"] for m in tmp])),
        }
    report={
        "schema":"anymove.human_spine_so3_multifamily_val/1",
        "status":"VALIDATION_COMPLETE_TEST_24_CLIPS_UNTOUCHED",
        "source_blend_sha256":V3_SHA,
        "families":list(args.families),"takes_per_family":args.takes,
        "clips":len(rows),"source_frames_per_clip":107,
        "observations":"synthetic ground truth projected 315/45 + 1px deterministic Gaussian noise",
        "model_modes":[m[0] for m in MODES],
        "iterations_per_mode":args.iterations,
        "all_samples_bonerooot_static":True,
        "test_clips_loaded":0,
        "scores_per_clip":results,
        "aggregate":per_mode,
        "limitations":[
            "Validation only, not independent sealed Test",
            "Procedural GT from same source rig, not natural MoCap or actual Reader",
            "Foot contact labels are unphysical proxy, not planted-sole GT",
            "Lowest MPJPE and lowest Spine angular jerk may select different modes",
            "All displayed Solver bone rotations are Blender local Euler, NOT iClone FK channels",
        ],
    }
    dst=OUT/("multifamily_"+str(args.takes)+"takes_"+str(args.iterations)+"iter_validation.json")
    dst.write_text(json.dumps(report,indent=2),encoding="utf8")
    print("SO3_MULTIFAMILY_VALIDATION_COMPLETE",
          json.dumps({"clips":len(rows),"frames":len(rows)*107,
                      "summary":per_mode,"report":str(dst)}),flush=True)

if __name__=="__main__":
    main()
