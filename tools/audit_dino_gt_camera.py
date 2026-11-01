"""Check that Blender GT cameras EXACTLY match AnyMove's orthographic layout.

Use BEFORE calling the Joint Reader/Solver. The Blender-generated JSON
contains each camera's rotation/translation and focal px scale; this
cross-check is independent of the Solver's own project() function.

python -m tools.audit_dino_gt_camera \
    EXPERIMENT/gt_render/joints.json \
    EXPERIMENT/gt_render/cam_front.json EXPERIMENT/gt_render/cam_side.json
"""
import argparse
import json
import numpy as np


def projection_from_camera(points, camera):
    xyz = np.asarray(points, dtype=float)
    R = np.asarray(camera["R"], dtype=float)
    t = np.asarray(camera["t"], dtype=float)
    if (camera.get("type") != "ORTHO" or
            R.shape != (3,3) or t.shape != (3,) or
            not np.isfinite(R).all() or not np.isfinite(t).all()):
        raise ValueError("invalid Blender orthographic camera transform")
    f = float(camera["f"])
    cx, cy = float(camera["cx"]), float(camera["cy"])
    if not np.isfinite([f,cx,cy]).all() or f <= 0:
        raise ValueError("nonfinite or nonpositive camera pixel focal length")
    local = xyz @ R.T + t
    return np.stack([cx + f*local[...,0], cy - f*local[...,1]],-1)


def audit_camera(truth, front, side, camera_geometry, *, tolerance_px=.2):
    J = np.asarray(truth["joints"], dtype=float)
    if J.ndim!=3 or J.shape[2]!=3 or not np.isfinite(J).all():
        raise ValueError("invalid Blender truth world positions")
    H,Wf,Ws,metres,Z=[float(camera_geometry[k]) for k in
                      ("H","W_FRONT","W_SIDE","M_PER_PX","Z_CENTER")]
    if not np.isfinite([H,Wf,Ws,metres,Z]).all() or min(H,Wf,Ws,metres)<=0:
        raise ValueError("invalid AnyMove camera layout")
    if not 0<tolerance_px<=5:
        raise ValueError("invalid camera audit pixel tolerance")
    v=H/2-(J[...,2]-Z)/metres
    projected={
        "front":np.stack([Wf/2+J[...,0]/metres,v],-1),
        "side":np.stack([Ws/2-J[...,1]/metres,v],-1),
    }
    report={
        "profile":"anymove.dino_blender_camera_layout_audit/1",
        "frames":int(J.shape[0]),"keypoints":int(J.shape[1]),
        "tolerance_px":float(tolerance_px),
        "views":{}
    }
    for name,cam,expected in (("front",front,(Wf,H)),("side",side,(Ws,H))):
        if list(cam.get("res",[])) != list(expected):
            raise ValueError(f"{name} rendered resolution does not match Dino layout")
        e = np.linalg.norm(
            projection_from_camera(J,cam)-projected[name],axis=-1)
        report["views"][name] = {
            "mean_pixel_geometry_difference":round(float(e.mean()),6),
            "max_pixel_geometry_difference":round(float(e.max()),6),
        }
        if e.max()>tolerance_px:
            raise ValueError(
                f"{name} Blender camera projection differs from AnyMove "
                f"layout by up to {e.max():.3f}px; do not use invalid GT video")
    report["status"]="blender_cameras_match_anymove_orthographic_layout"
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("blender_truth")
    p.add_argument("front_camera")
    p.add_argument("side_camera")
    p.add_argument("--tolerance-px",type=float,default=.2)
    a=p.parse_args(argv)
    from mocap import layout
    def read(path):
        with open(path,encoding="utf-8") as fd:
            return json.load(fd)
    geometry = {k:getattr(layout,k) for k in
                ("H","W_FRONT","W_SIDE","M_PER_PX","Z_CENTER")}
    report=audit_camera(read(a.blender_truth),read(a.front_camera),
                        read(a.side_camera),geometry,
                        tolerance_px=a.tolerance_px)
    print(json.dumps(report,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
