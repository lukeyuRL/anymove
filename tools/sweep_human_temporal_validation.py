"""Validation-only threshold sensitivity for HR V2 temporal tracker.

Use frozen Reader UV cache and TRAIN-only calibration. This is explicitly
development-set tuning, NEVER evidence of generalization on unseen clips.
"""
import copy,json,pathlib
import numpy as np

from mocap.temporal_skeleton_tracker import track_two_view
from mocap.camera_geometry import triangulate_orthographic

BASE=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
OUT=BASE/"reader_v2/full"
GT=BASE/"dataset/motion_gt"

def main():
    cache=np.load(OUT/"validation_raw_uv_frozen.npz")
    calib=json.loads((OUT/"temporal_skeleton_train_calibration.json").read_text(encoding="utf-8"))
    names=cache["names"].tolist()
    clips=cache["clips"].tolist()
    assert len(clips)==24 and names==calib["names"]
    store={clip:np.asarray(json.loads((GT/(clip+".json")).read_text(encoding="utf-8"))["joint_position_m"],dtype=float)
           for clip in clips}
    results=[]
    for speed_factor in (1.,4.,100000.):
      for spatial_factor in (.7,1.,1.5,2.):
       for bone_guard in (False,True):
            copycal=copy.deepcopy(calib)
            for slot in ("front","side"):
                for gate in copycal["cameras"][slot]:
                    gate["velocity_gate_px_per_frame"]*=speed_factor
                    gate["length_gate_px"]*=spatial_factor
            if not bone_guard:
                copycal["rest_lengths_m"]=[0.0]*len(copycal["rest_lengths_m"])
            err=[]
            count=np.zeros(3,int)
            temporal=[]
            lastgood=[]
            for i,clip in enumerate(clips):
                frame=cache["frames"][i].astype(int)
                f=cache["front_uv"][i]
                s=cache["side_uv"][i]
                tracked=track_two_view(f,s,frame,copycal,names)
                world=store[clip][frame]
                current=tracked["joints_world_m"]
                err.extend((np.linalg.norm(current-world,axis=-1)*100).ravel().tolist())
                count+=np.array([tracked["diagnostics"]["rejected_front"],
                                 tracked["diagnostics"]["rejected_side"],
                                 tracked["diagnostics"]["skeleton_repaired"]])
                temporal.extend((np.linalg.norm(np.diff(current,axis=0)-np.diff(world,axis=0),axis=-1)*100).ravel().tolist())
            e=np.asarray(err)
            t=np.asarray(temporal)
            results.append({
                "speed_factor":speed_factor,"spatial_factor":spatial_factor,"bone_guard":bone_guard,
                "mean_cm":round(float(e.mean()),5),
                "p95_cm":round(float(np.percentile(e,95)),5),
                "over5":int((e>5).sum()),
                "over10":int((e>10).sum()),
                "over50":int((e>50).sum()),
                "max_cm":round(float(e.max()),3),
                "temporal_mean_cm":round(float(t.mean()),5),
                "temporal_over5":int((t>5).sum()),
                "reject_front":int(count[0]),"reject_side":int(count[1]),"skeleton_repair":int(count[2]),
            })
    path=OUT/"temporal_validation_sensitivity.json"
    path.write_text(json.dumps({
        "source":"Validation-only after freezing Reader checkpoint; thresholds exploratory and not Test-selected",
        "results":results},indent=2))
    print("SENSITIVITY_TOP_BY_MEAN",json.dumps(sorted(results,key=lambda r:r["mean_cm"])[:10]),flush=True)
    print("SENSITIVITY_TOP_BY_SPIKES",json.dumps(sorted(results,key=lambda r:(r["over5"],r["mean_cm"]))[:10]),flush=True)
if __name__=="__main__":
    main()
