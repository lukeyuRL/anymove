"""Validate root hip/waist Kabsch initialization, VALIDATION clips only.

No test GT or Solver training. Raw GT marker projection + fixed 1px Gaussian
image noise exercises the root initialization against known world Root quat.
"""
import json,pathlib,zlib
import numpy as np
from scipy.spatial.transform import Rotation
from mocap.camera_geometry import project_orthographic,triangulate_orthographic
from mocap.human_root_initializer import estimate_human_v3_root_axis_angle,_orthonormal_factor

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
DATA=ROOT/"spine_motion_prior_v1"
RIG=json.loads((ROOT/"solver/rigged/rig_info_trainlimits.json").read_text(encoding="utf8"))
MAN=json.loads((DATA/"manifest.json").read_text(encoding="utf8"))
FAMILIES=("pivot_turn_left","walk_turn_right","waist_twist_right",
          "reach_counterrotate","lateral_bend_left","lean_recover")
SRC_SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"

def run():
    if MAN["source_blend_sha256"]!=SRC_SHA:
        raise RuntimeError("Invalid V3 Nose GT")
    names=[p["name"] for p in RIG["keypoints"]]
    rest=np.asarray([p["rest"] for p in RIG["keypoints"]],dtype=float)
    B=_orthonormal_factor(np.asarray(RIG["bones"][0]["matrix_local"])[:3,:3])
    recorded=[]
    for family in FAMILIES:
        row=next(x for x in MAN["rows"] if x["split"]=="validation" and x["family"]==family)
        with np.load(DATA/row["npz"],allow_pickle=False) as d:
            xyz=np.asarray(d["world_joints_m"],dtype=float)
            gtq=np.asarray(d["root_rotation_wxyz"],dtype=float)
        pix=[]
        rng=np.random.default_rng(20261104+zlib.crc32(row["clip"].encode("utf8")))
        for deg in (315,45):
            gt=project_orthographic(xyz,deg,672,672,2.432802200317383/672,.8688581585884094)
            pix.append(gt+rng.normal(0,1,gt.shape))
        ones=np.ones(xyz.shape[:2])
        tri=triangulate_orthographic(pix[0],pix[1],ones,ones,
              front_width=672,side_width=672,height=672,
              metres_per_pixel=2.432802200317383/672,z_center=.8688581585884094,
              front_angle_deg=315,side_angle_deg=45,front_height_weight=1.)
        roots=estimate_human_v3_root_axis_angle(tri,rest,names,B,np.ones(xyz.shape[:2]))
        world=Rotation.from_quat(gtq[:,[1,2,3,0]]).as_matrix()
        expected=Rotation.from_matrix(B.T@world@B)
        estimate=Rotation.from_rotvec(roots)
        e=np.rad2deg((expected.inv()*estimate).magnitude())
        original=np.rad2deg(expected.magnitude())
        result={"family":family,
            "root_gt_orientation_range_deg":row["root_yaw_range_deg"],
            "original_zero_root_mean_error_deg":float(original.mean()),
            "kabsch_root_mean_error_deg":float(e.mean()),
            "kabsch_root_p95_error_deg":float(np.percentile(e,95)),
            "kabsch_root_max_error_deg":float(e.max())}
        print("HUMAN_ROOT_ANCHOR_VALIDATION",json.dumps(result),flush=True)
        recorded.append(result)
    path=ROOT/"solver/spine_so3_research/root_initializer_validation.json"
    path.write_text(json.dumps({
        "schema":"anymove.human_v3_root_anchor_validation/1",
        "status":"VALIDATION_ONLY",
        "clips":recorded,"gt_used_for_scoring_only":True,
        "test_not_accessed":True},indent=2),encoding="utf8")
    print("HUMAN_ROOT_VALIDATION_SUMMARY",json.dumps({
        "clips":len(recorded),
        "avg_original_zero_root_error_deg":float(np.mean([x["original_zero_root_mean_error_deg"] for x in recorded])),
        "avg_kabsch_root_error_deg":float(np.mean([x["kabsch_root_mean_error_deg"] for x in recorded])),
        "worst_p95_kabsch_deg":max(x["kabsch_root_p95_error_deg"] for x in recorded),
        "saved":str(path)
    }),flush=True)

if __name__=="__main__":
    run()
