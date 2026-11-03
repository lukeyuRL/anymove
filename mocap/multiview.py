"""View-indexed orthographic camera observation contract for AnyMove.

No global Front/Side constants, hardcoded Creature bone names, or video tiles.
The contract accepts 1..N orthographic cameras; a 3D solve needs geometric
rank >=3 from observed camera axes. All per-view visibility labels are
metadata *not* automatically trusted as probabilistic Reader confidence.

Packet (NPZ, no pickle): uv[V,T,J,2], confidence[V,T,J],
valid[V,T,J], view_ids[V], names[J], world_to_camera[V,4,4],
image_size_px[V,2], meters_per_px[V], source_kind, frame_ids[T],
fps, independent_stills. 3D Ground Truth MUST be in a separate file.
"""
from pathlib import Path
import numpy as np

PROFILE = "anymove.multiview_observations/1"
EXPECTED = ("view_ids","names","uv","confidence","valid",
            "world_to_camera","image_size_px","meters_per_px",
            "frame_ids","fps","independent_stills","source_kind")


def project_np(world, world_to_camera, image_size_px, meters_per_px):
    """Return [V,T,J,2] pixel projection of world xyz [T,J,3]."""
    xyz=np.asarray(world,dtype=float)
    matrix=np.asarray(world_to_camera,dtype=float)
    sz=np.asarray(image_size_px,dtype=float)
    m=np.asarray(meters_per_px,dtype=float)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("world points must be finite T,J,3")
    v=len(matrix)
    if matrix.shape!=(v,4,4) or sz.shape!=(v,2) or m.shape!=(v,):
        raise ValueError("invalid camera projection parameters")
    uv=np.einsum("vij,tkj->vtki",matrix[:,:3,:3],xyz)+matrix[:,None,None,:3,3]
    return np.stack((sz[:,None,None,0]/2+uv[...,0]/m[:,None,None],
                     sz[:,None,None,1]/2-uv[...,1]/m[:,None,None]),axis=-1)


def linear_cameras(world_to_camera,image_size_px,meters_per_px):
    """Return u,v = A xyz + b, with A [V,2,3], b [V,2]."""
    R=np.asarray(world_to_camera,dtype=np.float64)
    size=np.asarray(image_size_px,dtype=np.float64)
    meters=np.asarray(meters_per_px,dtype=np.float64)
    inv=1/meters
    a=np.stack((R[:,0,:3]*inv[:,None],-R[:,1,:3]*inv[:,None]),axis=1)
    b=np.stack((size[:,0]/2+R[:,0,3]*inv,
                size[:,1]/2-R[:,1,3]*inv),axis=1)
    return a,b


def check_packet(p, *, rig_names=None):
    """Validate archive contents fail-closed, including camera extrinsics."""
    if any(key not in p for key in EXPECTED):
        raise ValueError("multiview packet missing required field")
    arr={key:np.asarray(p[key]) for key in EXPECTED}
    views=[str(x) for x in arr["view_ids"]]
    names=[str(x) for x in arr["names"]]
    if not 1<=len(views)<=12 or len(set(views))!=len(views) or any(not v.strip() for v in views):
        raise ValueError("invalid or duplicate View IDs")
    if len(names)<1 or len(names)!=len(set(names)) or any(not n.strip() for n in names):
        raise ValueError("invalid or duplicate Keypoint names")
    if rig_names is not None and names != [str(x) for x in rig_names]:
        raise ValueError("ordered Reader names differ from Rig")
    V=len(views);J=len(names)
    uv=arr["uv"]
    if uv.ndim!=4 or uv.shape[0]!=V or uv.shape[2:]!=(J,2):
        raise ValueError("UV must be V,T,J,2")
    T=uv.shape[1]
    if T<1:raise ValueError("empty frame sequence")
    conf=arr["confidence"];valid=arr["valid"]
    if conf.shape!=(V,T,J) or valid.shape!=(V,T,J):
        raise ValueError("confidence / valid shape incompatible")
    if valid.dtype != np.bool_:
        raise ValueError("valid mask must be bool")
    if not np.isfinite(uv).all() or not np.isfinite(conf).all() or (conf<0).any():
        raise ValueError("invalid UV or Reader confidence")
    # Raw peak amplitudes are not P(visible) and are allowed above 1.
    W=arr["world_to_camera"];size=arr["image_size_px"];mpp=arr["meters_per_px"]
    if (W.shape!=(V,4,4) or size.shape!=(V,2) or mpp.shape!=(V,)
            or not all(np.isfinite(x).all() for x in (W,size,mpp))
            or not (size>1).all() or not (mpp>0).all()):
        raise ValueError("invalid camera extrinsics, dimensions or metre scale")
    if not np.allclose(W[:,3,:],np.array([0,0,0,1]),atol=1e-7):
        raise ValueError("camera homogeneous coordinates invalid")
    rr=W[:,:3,:3]
    if (not np.allclose(rr@np.swapaxes(rr,-2,-1),
                         np.broadcast_to(np.eye(3),(V,3,3)),atol=2e-3)
            or not np.allclose(np.linalg.det(rr),1,atol=2e-3)):
        raise ValueError("camera must use proper rigid rotation matrices")
    fps=arr["fps"]
    independent=arr["independent_stills"]
    if fps.shape!=() or not np.isfinite(fps) or fps<=0:
        raise ValueError("fps must be one positive finite scalar")
    if independent.shape!=() or independent.dtype!=np.bool_:
        raise ValueError("independent_stills must be bool scalar")
    fid=arr["frame_ids"]
    if fid.shape!=(T,) or not np.issubdtype(fid.dtype,np.integer) or len(set(fid.tolist()))!=T:
        raise ValueError("invalid frame IDs")
    source=str(arr["source_kind"].item()) if arr["source_kind"].shape==() else ""
    if source not in ("blender_gt_projection","synthetic_noisy","reader_inference","legacy_reader"):
        raise ValueError("unsupported Observation source kind")
    if "joint_surface_part_visible" in p:
        raise ValueError("GT mesh surface visibility must be stored only in a separate evaluation file")
    if "joints_world" in p or "gt_world" in p:
        raise ValueError("3D GT must never be stored inside Solver input packet")
    optional_quality=("visibility_score","expected_error_px")
    if any(key in p for key in optional_quality):
        allowed_semantics={"synthetic_cutout_only_uncalibrated",
                           "mesh_surface_part_visibility_v1_uncalibrated"}
        if ("quality_semantics" not in p or
                str(np.asarray(p["quality_semantics"]).item()) not in allowed_semantics):
            raise ValueError("experimental Reader Quality Head must declare explicitly uncalibrated visibility semantics")
        for key in optional_quality:
            if key not in p:
                raise ValueError("experimental Reader Quality Head must provide both visibility and error estimates")
            q=np.asarray(p[key],dtype=float)
            if q.shape!=(V,T,J) or not np.isfinite(q).all():
                raise ValueError("invalid Reader Quality Head shape/values: "+key)
            if key=="visibility_score" and (np.any(q<0) or np.any(q>1)):
                raise ValueError("visibility score outside [0,1]")
            if key=="expected_error_px" and np.any(q<0):
                raise ValueError("negative predicted localization error")
    return {"V":V,"T":T,"J":J,"view_ids":views,"names":names,
            "fps":float(fps),"independent_stills":bool(independent),
            "source_kind":source}


def save_packet(path,p, *, rig_names=None):
    dest=Path(path)
    if dest.exists():raise FileExistsError("refusing to overwrite observations: "+str(dest))
    meta=check_packet(p,rig_names=rig_names)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,**p)
    return meta


def load_packet(path, *, rig_names=None):
    with np.load(path,allow_pickle=False) as raw:
        p={key:raw[key].copy() for key in raw.files}
    meta=check_packet(p,rig_names=rig_names)
    return p,meta


def triangulate(p, *, index_views=None, min_views=2, anchor=None,
                ridge_m=1.0, cond_max=1e9):
    """Independent weighted orthographic linear ray intersection.

    Does NOT require or access 3D GT. When geometric rank is deficient,
    optional anchor can fill underconstrained points but these are explicitly
    reported as rank_deficient; this is NOT a valid triangulated 3D truth.
    """
    meta=check_packet(p)
    views=np.arange(meta["V"]) if index_views is None else np.asarray(index_views,dtype=int)
    if (len(views)<min_views or len(set(views.tolist()))!=len(views) or
            np.any(views<0) or np.any(views>=meta["V"])):
        raise ValueError("invalid/insufficient selection of views")
    A,b=linear_cameras(p["world_to_camera"][views],
                       p["image_size_px"][views],p["meters_per_px"][views])
    obs=np.asarray(p["uv"],float)[views] - b[:,None,None,:]
    w=np.asarray(p["confidence"],float)[views]
    valid=np.asarray(p["valid"],bool)[views]
    weights=(w.clip(0,1)**2)*valid
    T=meta["T"];J=meta["J"]
    anchor_positions=np.zeros((T,J,3),float) if anchor is None else np.broadcast_to(
        np.asarray(anchor,dtype=float),(T,J,3)).copy()
    out=anchor_positions.copy()
    cond=np.full((T,J),np.inf)
    enough=np.zeros((T,J),bool)
    for t in range(T):
        for j in range(J):
            indices=np.nonzero(weights[:,t,j]>.001)[0]
            if len(indices)<min_views:
                continue
            mat=A[indices].reshape(-1,3)
            wg=np.repeat(np.sqrt(weights[indices,t,j]),2)
            rhs=obs[indices,t,j].reshape(-1)
            mat=mat*wg[:,None]
            rhs=rhs*wg
            singular=np.linalg.svd(mat,compute_uv=False)
            # Relative singular-value rank test: opposite cameras are nearly
            # identical projectors with a null depth axis, not a valid 3D fix.
            if singular[-1]<=1e-6*singular[0] or singular[0]/singular[-1]>cond_max:
                continue
            out[t,j]=np.linalg.lstsq(mat,rhs,rcond=None)[0]
            cond[t,j]=singular[0]/singular[-1]
            enough[t,j]=True
    return out, {"triangulable":enough,"condition_number":cond,
                 "triangulable_count":int(enough.sum()),"points_total":T*J,
                 "geometric_view_indices":views.tolist()}
