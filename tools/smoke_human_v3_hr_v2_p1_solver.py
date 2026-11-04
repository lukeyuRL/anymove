"""Short, deterministic P1 human 315/45 Solver FK integration smoke.

No H3 video, no iClone scene edits, no exported motion asset. Uses actual
V3 Nose GT and the generated 55-bone/67-marker Solver contract. Tests the
optional pre-Solver guard with one injected gross single-view outlier.
"""
import json,os,pathlib,sys
import numpy as np
import torch

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
os.environ["MOCAP_CREATURE"]=str((ROOT/"solver").resolve())
os.environ["MOCAP_CAMERA_AZIMUTHS"]="315,45"

from mocap import layout,solve

def main():
    rigpath=ROOT/"solver/rigged/rig_info.json"
    info=json.loads(rigpath.read_text(encoding="utf-8"))
    manifest=json.loads((ROOT/"dataset/manifest.json").read_text(encoding="utf-8"))
    cal=json.loads((ROOT/"reader_v2/full/temporal_skeleton_train_calibration.json").read_text(encoding="utf-8"))
    names=[k["name"] for k in info["keypoints"]]
    if names!=cal["names"] or names!=manifest["joint_names"]:
        raise RuntimeError("Human rig/Reader/GT marker names do not agree")
    if layout.FRONT_AZIMUTH_DEG!=315 or layout.SIDE_AZIMUTH_DEG!=45:
        raise RuntimeError("Solver did not load human camera layout")
    if layout.H!=672 or not np.isclose(layout.M_PER_PX,manifest["metres_per_pixel"]):
        raise RuntimeError("Solver geometry layout differs from render geometry")
    clip=next(c for c in manifest["train_clips"] if "finger_independent_roll" in c)
    motion=json.loads((ROOT/"dataset/motion_gt"/(clip+".json")).read_text(encoding="utf-8"))
    frames=np.asarray([8,15,22,29,36],dtype=int)
    xyz=np.asarray(motion["joint_position_m"],dtype=np.float64)[frames]
    f,s=layout.project(xyz)
    front=f.copy()
    target_idx=names.index("CC_Base_L_ThumbTip")
    front[2,target_idx]+=[200.,-80.]
    d={
      "front_uv":front,
      "side_uv":s.copy(),
      "front_conf":np.ones((len(frames),len(names))),
      "side_conf":np.ones((len(frames),len(names))),
      "frame_indices":frames.copy(),
    }
    guarded,report=solve.apply_temporal_joint_guard(d,names,frames,cal)
    guarded["frame_indices"]=frames.copy()
    original_dist=float(np.linalg.norm(d["front_uv"][2,target_idx]-f[2,target_idx]))
    repaired_dist=float(np.linalg.norm(guarded["front_uv"][2,target_idx]-f[2,target_idx]))
    if original_dist<200 or not repaired_dist<original_dist or not guarded["front_conf"][2,target_idx]<1:
        raise RuntimeError("P1 confidence and joint reassociation were not applied")
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    requested_iters=int(sys.argv[1]) if len(sys.argv)>1 else 10
    if not 1<=requested_iters<=300:raise ValueError('Smoke iteration count must be 1..300')
    fit=solve.fit(guarded,info,iters=requested_iters,dev=device,log=True)
    result_xyz=np.asarray(fit["joints"])
    if result_xyz.shape!=xyz.shape or not np.isfinite(result_xyz).all():
        raise RuntimeError("Torch FK Solver returned invalid joints")
    stats={
        "status":"SMOKE_PASS",
        "source_blend_sha256":cal["source_blend_sha256"],
        "clip":clip,"frames":frames.tolist(),
        "joints":len(names),"bones":len(info["bones"]),
        "device":device,"solver_iterations_smoke":requested_iters,
        "injected_2d_error_px":round(original_dist,4),
        "guarded_2d_error_px":round(repaired_dist,4),
        "injected_joint_confidence_after_guard":float(guarded["front_conf"][2,target_idx]),
        "guard_counts":report["tracker_diagnostics"],
        "fitted_vs_3d_gt_mean_cm":round(float(np.linalg.norm(result_xyz-xyz,axis=-1).mean()*100),5),
        "fit_reprojection":fit["reproj_px"],
        "notes":"Functional end-to-end smoke. Provisional unconstrained human joint limits; not production motion quality."
    }
    path=ROOT/"solver"/"p1_guarded_fk_smoke.json"
    path.write_text(json.dumps(stats,indent=2),encoding="utf-8")
    if not np.isfinite(stats['fitted_vs_3d_gt_mean_cm']) or stats['fitted_vs_3d_gt_mean_cm']>2.0:
        raise RuntimeError('Human FK Solver smoke accuracy exceeds 2cm threshold')
    print("P1_HUMAN_FK_SOLVER_SMOKE",json.dumps(stats),flush=True)
if __name__=="__main__":
    main()
