"""Exercise the real mocap.solve CLI with the Human V3 P1 guard opt-in.

Uses synthetic TRAIN GT with an injected single-camera joint failure.
This is a CLI/integration smoke, never production-quality iClone motion.
"""
import json,os,pathlib,subprocess,sys
import numpy as np

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
SOLVER=ROOT/"solver"
RIG=SOLVER/"rigged/rig_info.json"
CAL=ROOT/"reader_v2/full/temporal_skeleton_train_calibration.json"
INPUT=SOLVER/"p1_solver_main_input.npz"
OUTPUT=SOLVER/"p1_solver_main_smoke_motion.json"

def main():
    info=json.loads(RIG.read_text(encoding="utf-8"))
    cf=json.loads(CAL.read_text(encoding="utf-8"))
    manifest=json.loads((ROOT/"dataset/manifest.json").read_text(encoding="utf-8"))
    names=manifest["joint_names"]
    if names!=[r["name"] for r in info["keypoints"]] or names!=cf["names"]:
        raise RuntimeError("Rig and Reader marker identities differ")
    clip=next(name for name in manifest["train_clips"] if "finger_independent_roll" in name)
    gt=json.loads((ROOT/"dataset/motion_gt"/(clip+".json")).read_text(encoding="utf-8"))
    frames=np.asarray([8,15,22,29,36],dtype=np.int32)
    world=np.asarray(gt["joint_position_m"],dtype=np.float64)[frames]
    from mocap.camera_geometry import project_orthographic
    mpp=manifest["metres_per_pixel"]
    zc=manifest["z_center"]
    front=project_orthographic(world,315,672,672,mpp,zc)
    side=project_orthographic(world,45,672,672,mpp,zc)
    target=names.index("CC_Base_L_ThumbTip")
    front[2,target]+=[208,-89]
    np.savez_compressed(INPUT,front_uv=front,side_uv=side,
         front_conf=np.ones(world.shape[:2]),side_conf=np.ones(world.shape[:2]),
         frame_indices=frames,names=np.asarray(names),
         camera_azimuth_deg=np.asarray([315.,45.]),
         fps=np.array(24.0))
    env=os.environ.copy()
    env["MOCAP_CREATURE"]=str(SOLVER.resolve())
    env["MOCAP_CAMERA_AZIMUTHS"]="315,45"
    command=[sys.executable,"-m","mocap.solve",str(INPUT),str(RIG),str(OUTPUT),
             "--no-align","--temporal-calibration",str(CAL),"--iters","40","--allow-provisional-human-limits"]
    p=subprocess.run(command,cwd=str(pathlib.Path.cwd()),env=env,
        capture_output=True,text=True,timeout=120,encoding="utf-8",errors="replace")
    if p.returncode:
        raise RuntimeError("Solver CLI failed: "+p.stdout[-1600:]+" "+p.stderr[-1600:])
    motion=json.loads(OUTPUT.read_text(encoding="utf-8"))
    if (len(motion["bones"])!=55 or motion["joint_names"]!=names or len(motion["joints"])!=5
        or motion.get("source_frame_indices")!=frames.tolist()
        or motion.get("source_frames_contiguous") is not False
        or motion.get("motion_space_warning")!="BLENDER_LOCAL_EULER_NOT_ICLONE_FK_CHANNELS"):
        raise RuntimeError("Solver output motion does not match Human V3 marker contract")
    report=motion["temporal_joint_tracking"]
    if report["tracker_diagnostics"]["rejected_front"]<1 or report["camera_azimuth_deg"]!=[315,45]:
        raise RuntimeError("Guard flags not preserved in main Solver motion.json")
    if not any("CC_Base_L_ThumbTip" in f["front_inferred"] for f in report["sparse_flags_by_frame"]):
        raise RuntimeError("Injected thumb outlier not recorded")
    out={"status":"MAIN_CLI_INTEGRATION_PASS","frames":len(motion["joints"]),
         "bones":len(motion["bones"]),"markers":len(names),
         "source_blend_sha256":cf["source_blend_sha256"],
         "real_solver_output":str(OUTPUT),
         "guard_rejection":report["tracker_diagnostics"],
         "reprojection_px":motion["reproj_px"],
         "source_detections_sha256":motion["source_detections_sha256"],
         "warning":"Only 40 optimization iterations, provisional human Euler limits. NOT accepted motion quality."}
    (SOLVER/"p1_main_cli_validation.json").write_text(json.dumps(out,indent=2),encoding="utf8")
    print("HUMAN_P1_SOLVER_MAIN_CLI",json.dumps(out),flush=True)
if __name__=="__main__":
    main()
