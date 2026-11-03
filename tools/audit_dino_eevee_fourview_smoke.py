"""Audit the live five-pose Dino SkinMax41 Eevee smoke render.

Checks pixel dimensions, source FK, 41-color RGB saturation, Part-ID/depth,
mask IoU, orthographic opposite-mirror symmetry, input hashes and resume.
Creates one JSON report for a user-reviewed stop/go decision.
"""
import hashlib,json,pathlib,numpy as np,cv2,sys
ROOT=pathlib.Path(r"F:\anymove")
sys.path.insert(0,str(ROOT))
from blender.fourview_render_contract import sha256_file,sample_is_complete,VIEWS
SRC=ROOT/"runs/dino/dataset_hr_v1"
OUT=ROOT/"runs/dino/skinmax_camera_aware_v1/dataset_fourview_color41_full_v1_eevee"
INFO=json.loads((ROOT/"assets/dino/rigged/rig_info.json").read_text())
CONTRACT=json.loads((OUT/"render_contract.json").read_text())
from blender.fourview_render_contract import contract_sha256
SIG=contract_sha256(CONTRACT)
assert CONTRACT["engine"]=="BLENDER_EEVEE",CONTRACT["engine"]
print("CONTRACT",SIG,CONTRACT["skinmax_blend_sha256"])
POSES=(1,2,3,25,50)
result={"schema":"anymove.dino_skinmax41_eevee_smoke_qa/1",
        "engine":"BLENDER_EEVEE","pose_ids":list(POSES),
        "contract_sha256":SIG,"poses":[]}
MIN_IOU=0.91
for pose in POSES:
    source=SRC/f"{pose:05d}_meta.json"
    report=OUT/f"{pose:05d}_visibility_meta.json"
    s=json.loads(source.read_text())
    r=json.loads(report.read_text())
    assert r["max_fk_vs_source_gt_error_m"]<=.001,r["max_fk_vs_source_gt_error_m"]
    assert r["render_engine"]=="BLENDER_EEVEE"
    assert len(r["artifacts"])==8
    assert np.max(abs(np.asarray(r["joints_world"])-np.asarray(s["joints_world"])))<=.001
    assert sample_is_complete(OUT,pose,sha256_file(source),SIG,CONTRACT["dimensions"])
    p={"pose":pose,"fk_max_3d_error_m":r["max_fk_vs_source_gt_error_m"],"views":{}}
    data={}
    for view in VIEWS:
        png=OUT/f"{pose:05d}_{view}.png"
        npz=OUT/f"{pose:05d}_{view}_labels.npz"
        rgba=cv2.imread(str(png),cv2.IMREAD_UNCHANGED)
        with np.load(npz) as f:
            part=f["part_id"]
            depth=f["depth_m"]
            uv=f["joints_uv"]
            jdepth=f["joints_depth_m"]
        h,w=rgba.shape[:2]
        assert rgba.shape==(h,w,4)
        assert part.shape==depth.shape==(h,w),(part.shape,depth.shape)
        assert uv.shape==(len(INFO["keypoints"]),2)
        assert jdepth.shape==(len(INFO["keypoints"]),)
        assert part.dtype==np.int16 and depth.dtype==np.float32
        assert np.isfinite(uv).all() and np.isfinite(jdepth).all()
        assert part.min()>=-1 and part.max()<len(INFO["bones"])
        surface=np.isfinite(depth)
        alpha=rgba[:,:,3]>8
        intersection=np.count_nonzero(surface & alpha)
        union=np.count_nonzero(surface | alpha)
        iou=float(intersection/union) if union else 0.
        rgb=rgba[:,:,:3][alpha]
        rich=float(np.mean((rgb.max(axis=1)-rgb.min(axis=1))>35)) if len(rgb) else 0.
        assert len(rgb)>1000 and rich>0.18,(pose,view,"colors not visible",rich)
        assert iou>=MIN_IOU,(pose,view,"depth/RGB silhouette divergence",iou)
        assert np.all(depth[surface]>0),(pose,view,"invalid visible depth")
        assert np.all(part[~surface]==-1),(pose,view,"part id leaks into BG")
        count=int(np.count_nonzero(part>=0))
        assert count>1000,(pose,view,"empty skin labels")
        data[view]=uv
        p["views"][view]={"rgb_size":[w,h],"silhouette_iou":round(iou,5),
                          "fraction_saturated_pixels":round(rich,5),
                          "visible_surface_pixels":int(surface.sum()),
                          "part_covered_surface_pixels":count,
                          "visible_part_classes":int(len(np.unique(part[part>=0]))),
                          "joints":len(uv),"rgba":True,
                          "artifact_sha_verified":True}
    for a,b,width in (("front","back",data["front"].shape[0]*0+CONTRACT["dimensions"]["front"][0]),
                      ("side","opposite_side",CONTRACT["dimensions"]["side"][0])):
        one,two=data[a],data[b]
        err=max(float(np.abs(one[:,0]+two[:,0]-width).max()),
                float(np.abs(one[:,1]-two[:,1]).max()))
        assert err<0.02,(pose,a,b,err)
        p.setdefault("mirror_max_error_px",{})[a+"_"+b]=round(err,5)
    result["poses"].append(p)
    print("PASS",pose,"IoU",[(v,p["views"][v]["silhouette_iou"]) for v in VIEWS],
          "color frac",[(v,p["views"][v]["fraction_saturated_pixels"]) for v in VIEWS],
          "FK-m",p["fk_max_3d_error_m"],flush=True)
result["min_iou"]=round(min(x["silhouette_iou"] for pose in result["poses"] for x in pose["views"].values()),5)
result["status"]="PASS"
p=OUT/"smoke_5pose_eevee_geometry_color_audit.json"
if p.exists():
    raise FileExistsError("refuse to overwrite existing smoke audit")
p.write_text(json.dumps(result,indent=2),encoding="utf8")
print("SMOKE_QA_PASS",str(p),"min_iou",result["min_iou"],flush=True)
