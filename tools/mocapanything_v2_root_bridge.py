"""MoCapAnything V2 BVH -> AnyMove Dino/world Root bridge.
The V2 predictor uses a root-relative pose and exports a static-root BVH.
This adds a front-view-only X/Z root track using existing AnyMove detections.
The side view is NOT used for fitting. No image/video generation.

Example:
python tools/mocapanything_v2_root_bridge.py --v2-repo F:/mocapanything_v2 \
  --rig-info assets/dino/rigged/rig_info.json \
  --reference-bvh F:/mocapanything_v2/zoo/characters_face_zplus/Dino/Dino_ffs.bvh \
  --source-bvh F:/mocapanything_v2/outputs/anymove_dino_p0/h3front_124f/Dino#H3Front/Dino_Dino_rot6d_pred.bvh \
  --detections runs/dino/clips/test_walk/det.npz \
  --out F:/mocapanything_v2_test/Dino_V2_front_root_124f.bvh
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def similarity_from_rest(reference, target):
    """Fit x_world = scale * x_bvh @ rotation + offset using REST POSE only."""
    source = np.asarray(reference, float)
    dest = np.asarray(target, float)
    if source.shape != dest.shape or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("need matching 3D skeleton heads")
    x = source-source.mean(0)
    y = dest-dest.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    direction = np.eye(3)
    direction[2,2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ direction @ vt
    scale = float(np.sum((x @ rotation)*y) / np.sum(x*x))
    offset = dest.mean(0) - scale*source.mean(0) @ rotation
    fit = scale*source@rotation + offset
    errors=np.linalg.norm(fit-dest,axis=-1)
    return scale,rotation,offset,errors


def read_motion_lines(path, frames):
    """Read a BVH root's channel layout and preserve every non-root channel bytewise."""
    lines=path.read_text(encoding="utf-8").splitlines(keepends=True)
    root=None
    for i,line in enumerate(lines):
        stripped=line.strip()
        if stripped.startswith("ROOT "):
            root=stripped[5:].strip()
            break
    if not root:
        raise ValueError("no ROOT declaration in BVH")
    channels=[]
    for line in lines[i+1:]:
        if "CHANNELS " in line:
            parts=line.split()
            channels=parts[2:2+int(parts[1])]
            break
    if len(channels)<6 or any(f"{axis}position" not in channels for axis in "XYZ"):
        raise ValueError("root motion requires X/Y/Zposition channels")
    frame_ix=next((i for i,line in enumerate(lines) if line.strip().startswith("Frame Time:")),None)
    if frame_ix is None:raise ValueError("missing Frame Time")
    first=frame_ix+1
    if len(lines)-first != frames:
        raise ValueError(f"frame count mismatch in BVH text: {len(lines)-first} != {frames}")
    return lines,first,channels,root


def write_root_positions(lines, first, channels, added_xyz, out):
    """Patch only the root's position channels, never BVH bone rotations."""
    old_positions=[]
    position_channels=[channels.index(f"{ax}position") for ax in "XYZ"]
    for f,disp in enumerate(added_xyz):
        text=lines[first+f]
        fields=text.split()
        if len(fields)<=max(position_channels):
            raise ValueError(f"missing position values at frame {f}")
        old_positions.append([float(fields[i]) for i in position_channels])
        for idx,increment in zip(position_channels,disp):
            fields[idx]=f"{float(fields[idx])+float(increment):.8f}"
        suffix="\r\n" if text.endswith("\r\n") else "\n"
        lines[first+f]=" ".join(fields)+suffix
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("".join(lines),encoding="utf-8")
    return np.asarray(old_positions)


def compute_front_root(det,layout,root_rest):
    """Observed Front hip -> world X and Z. Y must remain fixed at the rig's rest value."""
    names=[str(n) for n in det["names"]]
    root_name="joint:"+layout["root_bone"]+":head"
    if root_name not in names:raise ValueError(f"no exact hip keypoint {root_name}")
    hi=names.index(root_name)
    uv=np.asarray(det["front_uv"][:,hi],float)
    if not np.isfinite(uv).all():raise ValueError("nonfinite hip tracking")
    m=float(layout["M_PER_PX"])
    fw=float(layout["W_FRONT"])
    h=float(layout["H"])
    zc=float(layout["Z_CENTER"])
    if m<=0:raise ValueError("invalid layout M_PER_PX")
    base_u=fw/2+root_rest[0]/m
    base_v=h/2-(root_rest[2]-zc)/m
    target=np.broadcast_to(root_rest,(len(uv),3)).copy()
    target[:,0]+=(uv[:,0]-base_u)*m
    target[:,2]+=(base_v-uv[:,1])*m
    return target,uv,hi


def run(args):
    # MocapAnything's own BVH parser and FK avoid a duplicate, potentially wrong BVH reader.
    sys.path.insert(0,str(Path(args.v2_repo).resolve()))
    from utils import bvh as BVH,animation

    info=json.loads(Path(args.rig_info).read_text(encoding="utf-8"))
    bones=info["bones"]
    root_bones=[i for i,b in enumerate(bones) if b["parent"]<0]
    if root_bones != [0]:raise ValueError("requires one root at bone index 0")
    rig_names=[b["name"] for b in bones]
    rest=np.array([np.asarray(b["matrix_local"],dtype=float)[:3,3] for b in bones])
    camera=info["layout"].copy()
    camera["root_bone"]=rig_names[0]
    if not {"M_PER_PX","W_FRONT","W_SIDE","H","Z_CENTER"}<=camera.keys():
        raise ValueError("missing Dino/AnyMove calibrated camera layout")

    ref, ref_names, _=BVH.load(args.reference_bvh)
    pred, pred_names, fps_interval=BVH.load(args.source_bvh)
    if len(ref_names)!=len(rig_names) or set(ref_names)!=set(rig_names):
        raise ValueError("reference BVH bones must equal rig bones")
    if len(pred_names)!=len(rig_names) or set(pred_names)!=set(rig_names):
        raise ValueError("prediction BVH bones must equal rig bones")
    ref_head=animation.positions_global(ref)[0,[ref_names.index(n) for n in rig_names]]
    scale,rotation,offset,err=similarity_from_rest(ref_head,rest)
    if np.max(err)>0.002:
        raise ValueError(f"rest basis mismatch: max {max(err):.6f}m")
    if not (0.98<scale<1.02):
        raise ValueError(f"unexpected BVH/world units: fitted scale {scale}")

    det=np.load(args.detections)
    T=len(pred)
    if len(det["front_uv"])!=T:
        raise ValueError(f"video/detection/BVH frames mismatch: {len(det['front_uv'])} vs {T}")
    source_fps=1.0/float(fps_interval)
    if abs(source_fps-float(det["fps"]))>0.02:
        raise ValueError(f"fps mismatch: {source_fps} vs {det['fps']}")
    target,observed,root_keypoint_idx=compute_front_root(det,camera,rest[0])

    # Use actual root FK, not template root OFFSET; exported BVHs can differ.
    root_model=animation.positions_global(pred)[:,pred_names.index(rig_names[0])]
    source_world=scale*root_model@rotation
    delta_model=(target-source_world)@rotation.T/scale
    original_bvh=Path(args.source_bvh)
    dest=Path(args.out)
    if dest.resolve()==original_bvh.resolve():
        raise ValueError("output must not replace the source prediction")
    lines,first,channels,root_bvh=read_motion_lines(original_bvh,T)
    if root_bvh!=rig_names[0]:
        raise ValueError(f"BVH root is {root_bvh}, rig root is {rig_names[0]}")
    old_xyz=write_root_positions(lines,first,channels,delta_model,dest)

    # Strict round-trip and FK validation.
    result,result_names,dt=BVH.load(str(dest))
    if result_names!=pred_names:raise AssertionError("bone order changed")
    if len(result)!=T or abs(dt-fps_interval)>1e-8:raise AssertionError("frames or FPS changed")
    fk_before=animation.positions_global(pred)
    fk_after=animation.positions_global(result)
    diff=(fk_after-fk_before)
    shifted=diff - delta_model[:,None,:]
    fk_roundtrip_max=float(np.abs(shifted).max())
    if fk_roundtrip_max>2e-5:
        raise AssertionError(f"non-root transforms changed: max {fk_roundtrip_max}")
    world_after=scale*fk_after[:,pred_names.index(rig_names[0])]@rotation
    root_error=float(np.max(np.linalg.norm(world_after-target,axis=-1)))
    if root_error>2e-5:raise AssertionError(f"root target mismatch: {root_error}")
    from_model=scale*fk_after[:,pred_names.index(rig_names[0])]@rotation
    proj_u=camera["W_FRONT"]/2+from_model[:,0]/camera["M_PER_PX"]
    proj_v=camera["H"]/2-(from_model[:,2]-camera["Z_CENTER"])/camera["M_PER_PX"]
    hip_reproj=float(np.max(np.linalg.norm(np.stack((proj_u,proj_v),1)-observed,axis=-1)))
    if hip_reproj>0.002:raise AssertionError(f"root pixel error {hip_reproj}")

    movement=target-target[0]
    report={"status":"PASS","source_bvh":str(original_bvh),"corrected_bvh":str(dest),
        "frames":T,"fps":float(det["fps"]),"bone_count":len(pred_names),
        "root_tracking":"FRONT_ONLY_XZ_FROM_ANYMOVE_READER__Y_FIXED_TO_REST",
        "side_held_out":True,"used_new_video_generation":False,
        "camera_layout":{k:camera[k] for k in ("H","W_FRONT","W_SIDE","M_PER_PX","Z_CENTER")},
        "basis_rotation_bvh_to_world":rotation.tolist(),"basis_scale":float(scale),
        "template_basis_offset_world_m":offset.tolist(),
        "rest_fit_max_error_m":float(err.max()),"rest_fit_rmse_m":float(np.sqrt(np.mean(err**2))),
        "root_initial_world_m":target[0].tolist(),"root_last_world_m":target[-1].tolist(),
        "root_delta_last_m":movement[-1].tolist(),
        "root_max_travel_from_initial_m":float(np.linalg.norm(movement,axis=-1).max()),
        "root_movement_total_path_m":float(np.linalg.norm(np.diff(target,axis=0),axis=-1).sum()),
        "roundtrip_fk_max_m":fk_roundtrip_max,"root_target_max_error_m":root_error,
        "root_front_hip_max_reprojection_px":hip_reproj,
        "limitations":["Root Y is unobservable from Front camera; held fixed",
                       "Root X/Z comes from creature-trained AnyMove Reader, not MoCapAnything V2 itself",
                       "This does not validate body pose or side-view reprojection"]}
    report_path=Path(args.report) if args.report else dest.with_suffix(".report.json")
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser()
    for arg in ("v2-repo","rig-info","reference-bvh","source-bvh","detections","out"):
        ap.add_argument("--"+arg,required=True)
    ap.add_argument("--report",default="")
    run(ap.parse_args())


if __name__=="__main__":
    main()
