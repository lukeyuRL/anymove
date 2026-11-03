"""Software orthographic z-buffer for visibility of SKINNED SURFACE PARTS.

Pure NumPy, independent of Blender and trained Reader. Expects triangles
already transformed to each camera's LOCAL coordinates (x right, y up,
-z forward). The origin is the camera optical center.

Outputs true visible-surface part IDs and camera depth for a posed mesh,
plus amodal surface-part projected coverage. It does NOT measure whether
a buried joint center is visible, and does NOT infer real H3 occlusion.
"""
import numpy as np


BACKGROUND = -1
UNLABELED_SURFACE = -2


def dominant_triangle_parts(vertex_bone_weights, faces, *, min_weight=.45,
                            min_margin=.08):
    """Assign each deformed triangle to its strongest skinning bone region.

    Ambiguous/unweighted triangles remain opaque but receive part ID -2.
    Skinning weights, not bone naming conventions, define the regions.
    """
    weights=np.asarray(vertex_bone_weights,dtype=np.float64)
    f=np.asarray(faces,dtype=np.int64)
    if (weights.ndim!=2 or f.ndim!=2 or f.shape[1]!=3 or
            not len(weights) or weights.shape[1]<1 or not len(f) or
            np.any(f<0) or np.any(f>=len(weights)) or
            not np.isfinite(weights).all() or np.any(weights<0) or
            not 0<=min_weight<=1 or not 0<=min_margin<=1):
        raise ValueError("invalid skinning weights or triangle indices")
    mean=(weights[f[:,0]]+weights[f[:,1]]+weights[f[:,2]])/3.
    winner=np.argmax(mean,axis=1)
    greatest=np.max(mean,axis=1)
    second=np.partition(mean,-2,axis=1)[:,-2] if mean.shape[1]>1 else np.zeros(len(mean))
    known=(greatest>=min_weight)&((greatest-second)>=min_margin)
    result=np.where(known,winner,UNLABELED_SURFACE).astype(np.int16)
    return result, {"num_triangles":len(f),"assigned_triangles":int(known.sum()),
                    "ambiguous_triangles":int((~known).sum()),
                    "parts_with_assigned_surface":int(len(set(result[result>=0].tolist())))}


def rasterize_camera_mesh(vertices_camera, faces, face_parts, *,
                          width, height, meters_per_px, n_parts,
                          near=.1, far=500.0):
    """Orthographic triangle rasterization with closest-surface depth testing.

    Camera +X is image right, +Y up, -Z forward. Depth is -camera_Z.
    Amodal coverage for each part is all projected triangle pixels without
    other parts occluding it; visible coverage is the frontmost part ID.
    """
    vertices=np.asarray(vertices_camera,dtype=np.float64)
    face=np.asarray(faces,dtype=np.int64)
    part=np.asarray(face_parts,dtype=np.int32)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or face.ndim!=2 or
            face.shape[1]!=3 or len(face)!=len(part) or
            not len(vertices) or not len(face) or
            not np.isfinite(vertices).all() or np.any(face<0) or
            np.any(face>=len(vertices)) or
            np.any((part<0)&(part!=UNLABELED_SURFACE)) or
            np.any(part>=n_parts) or not 0<n_parts<32760 or
            not isinstance(width,int) or not isinstance(height,int) or
            width<1 or height<1 or not np.isfinite(meters_per_px) or
            meters_per_px<=0 or near<=0 or far<=near):
        raise ValueError("invalid camera mesh or raster parameters")
    xy=np.stack((width/2+vertices[:,0]/meters_per_px,
                 height/2-vertices[:,1]/meters_per_px),axis=1)
    depth=-vertices[:,2]
    depth_buffer=np.full((height,width),np.inf,dtype=np.float64)
    part_map=np.full((height,width),BACKGROUND,dtype=np.int16)
    amodal=np.zeros((n_parts,height,width),dtype=np.bool_)
    visible_triangles=0
    clipped=0
    for triangle,pid in zip(face,part):
        d=depth[triangle]
        if np.any(d<=near) or np.any(d>=far):
            clipped+=1
            continue
        u0,v0=xy[triangle[0]]
        u1,v1=xy[triangle[1]]
        u2,v2=xy[triangle[2]]
        det=(v1-v2)*(u0-u2)+(u2-u1)*(v0-v2)
        if abs(det)<1e-9:
            continue
        x0=max(0,int(np.ceil(min(u0,u1,u2)-.5)))
        x1=min(width-1,int(np.floor(max(u0,u1,u2)-.5)))
        y0=max(0,int(np.ceil(min(v0,v1,v2)-.5)))
        y1=min(height-1,int(np.floor(max(v0,v1,v2)-.5)))
        if x0>x1 or y0>y1:
            continue
        gx=(np.arange(x0,x1+1,dtype=np.float64)+.5)[None,:]
        gy=(np.arange(y0,y1+1,dtype=np.float64)+.5)[:,None]
        w0=((v1-v2)*(gx-u2)+(u2-u1)*(gy-v2))/det
        w1=((v2-v0)*(gx-u2)+(u0-u2)*(gy-v2))/det
        w2=1-w0-w1
        inside=(w0>=-1e-8)&(w1>=-1e-8)&(w2>=-1e-8)
        if not np.any(inside):
            continue
        visible_triangles+=1
        if pid>=0:
            region=amodal[int(pid),y0:y1+1,x0:x1+1]
            region[inside]=True
        z=np.asarray(w0*d[0]+w1*d[1]+w2*d[2])
        region_depth=depth_buffer[y0:y1+1,x0:x1+1]
        nearer=inside&(z<region_depth)
        if np.any(nearer):
            region_depth[nearer]=z[nearer]
            region_part=part_map[y0:y1+1,x0:x1+1]
            region_part[nearer]=pid
    count_amodal=amodal.sum(axis=(1,2)).astype(np.int32)
    count_visible=np.bincount(
        part_map[part_map>=0].astype(np.int64),minlength=n_parts).astype(np.int32)
    if np.any(count_visible>count_amodal):
        raise AssertionError("visible area exceeds amodal area")
    fractions=np.divide(count_visible,count_amodal,
                        out=np.full(n_parts,np.nan,dtype=np.float64),
                        where=count_amodal>0)
    depth_image=depth_buffer.astype(np.float32)
    depth_image[~np.isfinite(depth_buffer)]=np.nan
    return {
        "depth_m":depth_image,
        "part_id":part_map,
        "amodal_pixel_count":count_amodal,
        "visible_pixel_count":count_visible,
        "surface_visible_fraction":fractions.astype(np.float32),
        "total_surface_pixels":int(np.isfinite(depth_buffer).sum()),
        "labeled_surface_pixels":int(np.count_nonzero(part_map>=0)),
        "ambiguous_surface_pixels":int(np.count_nonzero(part_map==UNLABELED_SURFACE)),
        "rasterized_triangles":visible_triangles,
        "near_far_clipped_triangles":clipped,
    }


def part_visibility_targets(raster, joint_bone_indices, *, min_area_px=12,
                            min_visible_fraction=.025):
    """Return binary visible/occluded and unknown=-1 for part-owned keypoints.

    This is an explicitly defined *surface part* label, not internal-joint
    ray visibility. Parts with tiny or no projected area are UNKNOWN.
    """
    bone=np.asarray(joint_bone_indices)
    area=np.asarray(raster["amodal_pixel_count"])
    fraction=np.asarray(raster["surface_visible_fraction"])
    if (bone.ndim!=1 or not np.issubdtype(bone.dtype,np.integer) or
            np.any(bone<0) or np.any(bone>=len(area)) or
            min_area_px<1 or not 0<=min_visible_fraction<=1):
        raise ValueError("invalid part-to-keypoint mapping or thresholds")
    labels=np.full(len(bone),-1,dtype=np.int8)
    valid=area[bone]>=min_area_px
    labels[valid]=(fraction[bone[valid]]>=min_visible_fraction).astype(np.int8)
    return labels


def camera_axes_consistency(xy_front,xy_back,width,*,atol_px=.02):
    """Check ideal 180�X orthographic camera pair across a fixed body point."""
    a=np.asarray(xy_front)
    b=np.asarray(xy_back)
    return bool(np.allclose(a[...,0]+b[...,0],width,atol=atol_px))
