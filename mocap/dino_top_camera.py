"""Top-view camera contract for optional Dino multi-view 3D fit.

The two-view AnyMove Front/Side layout remains unchanged. For a separately
rendered orthographic Top camera at z=30m, Blender camera local -Z looks down
toward world -Z and local image up is +Y:

    u = W_TOP/2 + X/M_PER_PX
    v = H/2 - Y/M_PER_PX

All camera data MUST also match Blender's exported cam_top.json. Inference
must never synthesize Top detections from Front/Side or Blender ground truth.
"""
import json
import numpy as np

from mocap import layout

TOP_WIDTH = 880
TOP_HEIGHT = 720
PROFILE = "anymove.dino_top_orthographic_camera_v1"


def validate_top_camera(camera):
    if not isinstance(camera, dict) or camera.get("type") != "ORTHO":
        raise ValueError("Top camera must be a Blender ORTHO camera")
    if camera.get("res") != [TOP_WIDTH, TOP_HEIGHT]:
        raise ValueError("unexpected Dino Top camera resolution")
    r = np.asarray(camera.get("R"), dtype=np.float64)
    t = np.asarray(camera.get("t"), dtype=np.float64)
    f = float(camera.get("f", 0))
    cx, cy = float(camera.get("cx", -1)), float(camera.get("cy", -1))
    if (r.shape != (3, 3) or t.shape != (3,) or
            not np.isfinite(r).all() or not np.isfinite(t).all() or
            not np.isfinite([f,cx,cy]).all()):
        raise ValueError("invalid Top camera projection fields")
    if not np.allclose(r, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("Top camera must look down -Z with X right and +Y up")
    if not np.isclose(f, 1/layout.M_PER_PX, rtol=0, atol=1e-4):
        raise ValueError("Dino Top camera scale differs from fixed layout")
    if abs(cx - TOP_WIDTH/2) > 1e-4 or abs(cy - TOP_HEIGHT/2) > 1e-4:
        raise ValueError("Top camera principal point differs from resolution")
    if not np.allclose(t[:2], 0, rtol=0, atol=1e-4):
        raise ValueError("Top camera X/Y target must be world origin")
    if float(t[2]) >= -1.0:
        raise ValueError("Top camera must originate above the Dino")
    return {"profile":PROFILE,"R":r,"t":t,"f":f,"cx":cx,"cy":cy}


def project_top_numpy(joints, camera):
    spec = validate_top_camera(camera)
    pts = np.asarray(joints, np.float64)
    if pts.ndim < 2 or pts.shape[-1] != 3 or not np.isfinite(pts).all():
        raise ValueError("finite 3D joint array required for Top projection")
    local=pts@spec["R"].T+spec["t"]
    return np.stack([spec["cx"] + spec["f"]*local[...,0],
                     spec["cy"] - spec["f"]*local[...,1]],-1)


def project_top_torch(joints, camera):
    """Differentiable projection using the same exact Blender geometry."""
    import torch
    spec=validate_top_camera(camera)
    R=torch.as_tensor(spec["R"],dtype=joints.dtype,device=joints.device)
    translation=torch.as_tensor(spec["t"],dtype=joints.dtype,device=joints.device)
    local=torch.matmul(joints,R.T)+translation
    return torch.stack([spec["cx"]+spec["f"]*local[...,0],
                        spec["cy"]-spec["f"]*local[...,1]],dim=-1)


def camera_from_json(path):
    with open(path,encoding="utf-8") as file:
        camera=json.load(file)
    validate_top_camera(camera)
    return camera
