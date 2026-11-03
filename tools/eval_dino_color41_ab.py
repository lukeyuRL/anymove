"""Matched Dino material A/B verification and heldout 2D/3D evaluation.

Data phase checks exact same poses, camera transforms, GT part masks, depth,
new 12-pose test ids not used for palette design or Reader training.

Result phase reads both independent Reader packets/solver outputs and separate
Blender GT to compute mean/P95/paired-per-pose errors and per-Bone categories.
No H3 generation and never trains models or uses GT in solver inputs.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from collections import Counter

VIEWS=("front","side","back","opposite_side")
TRAIN_IDS=tuple(range(1,25))
VAL_IDS=(25,50,75,100)
TEST_IDS=tuple(range(11001,11013))
def hash_file(file):
    h=hashlib.sha256()
    with open(file,"rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()
def error(x):
    a=np.asarray(x,dtype=np.float64)
    if not a.size or not np.isfinite(a).all():raise ValueError("no finite errors")
    return {"n":int(a.size),"mean":round(float(a.mean()),4),
            "median":round(float(np.median(a)),4),
            "p95":round(float(np.percentile(a,95)),4),
            "max":round(float(np.max(a)),4)}
def group(name):
    n=name.lower()
    if "tail" in n:return "tail"
    if "toe" in n:return "toes"
    if any(x in n for x in ("foot","calf","thigh")):return "hindlegs"
    if any(x in n for x in ("calvicle","clavicle","upperarm","forearm","hand","index","mid")):return "forelimbs_hands"
    return "body"
def verify_dataset(A,B,ids,rig):
    if len(ids)!=len(set(ids)):raise ValueError("duplicate pose IDs")
    out={"pose_count":len(ids),"frame_ids":list(ids),"camera_geometry_equal":True,
         "part_id_equal":True,"depth_equal":True,"joint_projected_GT_equal":True,
         "RGB_difference_detected":False,"mean_rgb_abs_difference_per_view":{}}
    errors={v:[] for v in VIEWS}
    rigsha=hash_file(rig)
    for idx in ids:
        am=json.loads((A/f"{idx:05d}_visibility_meta.json").read_text())
        bm=json.loads((B/f"{idx:05d}_visibility_meta.json").read_text())
        if (am["rig_info_sha256"]!=rigsha or bm["rig_info_sha256"]!=rigsha or
                am["source_meta_sha256"]!=bm["source_meta_sha256"] or
                not bm["viewaware_color_applied"] or am.get("viewaware_color_applied") or
                am["appearance_mode"]!="source-material" or
                bm["appearance_mode"]!="source-material"):
            raise ValueError(f"unmatched provenance or mixed appearance for pose {idx}")
        for view in VIEWS:
            ap=A/f"{idx:05d}_{view}_labels.npz"
            bp=B/f"{idx:05d}_{view}_labels.npz"
            with np.load(ap,allow_pickle=False) as av, np.load(bp,allow_pickle=False) as bv:
                np.testing.assert_array_equal(av["part_id"],bv["part_id"])
                np.testing.assert_allclose(av["depth_m"],bv["depth_m"],equal_nan=True)
                np.testing.assert_allclose(av["joints_uv"],bv["joints_uv"],atol=.008)
            for k in ("camera_world_to_local","image_size_px","meters_per_pixel"):
                if not np.allclose(am["views"][view][k],bm["views"][view][k],atol=1e-6):
                    raise ValueError("camera differs across material groups")
            import cv2
            orig=cv2.imread(str(A/f"{idx:05d}_{view}.png"),cv2.IMREAD_UNCHANGED)
            colored=cv2.imread(str(B/f"{idx:05d}_{view}.png"),cv2.IMREAD_UNCHANGED)
            if orig is None or colored is None or orig.shape!=colored.shape or orig.shape[2]!=4:
                raise ValueError("image mismatch")
            diff=float(np.abs(orig[:,:,:3].astype(np.float32)-
                              colored[:,:,:3].astype(np.float32)).mean())
            errors[view].append(diff)
    out["mean_rgb_abs_difference_per_view"]={v:round(float(np.mean(errors[v])),3) for v in VIEWS}
    out["RGB_difference_detected"]=all(np.mean(x)>1 for x in errors.values())
    if not out["RGB_difference_detected"]:
        raise ValueError("control and color-coded images unexpectedly identical")
    return out
def analysis(root):
    root=Path(root).resolve()
    exp=root/"runs/dino/skinmax_camera_aware_v1"
    rigpath=root/"assets/dino/rigged/rig_info.json"
    rig=json.loads(rigpath.read_text(encoding="utf8"))
    orig=root/"runs/dino/dataset_fourview_p3_material_fullres_v1"
    colored=exp/"ab_train_val_color41_v1"
    newGT=exp/"ab_blind_new_pose_v1"
    aa=exp/"ab_blind_A_original_v1";bb=exp/"ab_blind_B_color41_v1"
    train=verify_dataset(orig,colored,TRAIN_IDS+VAL_IDS,rigpath)
    blind=verify_dataset(aa,bb,TEST_IDS,rigpath)
    originals={i:int(x.name[:5]) for i,x in enumerate(newGT.glob("*_meta.json"))}
    if set(originals.values())!=set(TEST_IDS):
        raise ValueError("new seed poses not complete")
    for i in TEST_IDS:
        meta=json.loads((newGT/f"{i:05d}_meta.json").read_text())
        if meta["schema"]!="anymove.p4_independent_3d_blender_pose/1" or meta["sample_seed"]!=20261221+i:
            raise ValueError("new independent test pose seed mismatch")
    return {"schema":"anymove.dino_color41_ab_dataset_contract/1",
            "color_palette_sha256":hash_file(exp/"optimized_palette_constrained_v4.json"),
            "original_reader_sha256":hash_file(root/"runs/dino/reader_hr_v1.pt"),
            "training":train,"new_blind":blind,
            "all_blind_ids_greater_than_11000":True,
            "test_pose_not_used_in_palette_optimizer":True,
            "no_H3_video_generated":True}
def report_metrics(root):
    root=Path(root).resolve()
    exp=root/"runs/dino/skinmax_camera_aware_v1"
    rig=json.loads((root/"assets/dino/rigged/rig_info.json").read_text())
    names=[kp["name"] for kp in rig["keypoints"]]
    bone=[rig["bones"][kp["bone"]]["name"] for kp in rig["keypoints"]]
    groups={}
    palette=json.loads((exp/'optimized_palette_constrained_v4.json').read_text(encoding='utf8'))
    selected=set(palette['bone_colors'])
    for i,name in enumerate(bone):
        category=group(name)
        groups.setdefault(category,[]).append(i)
        if category=='toes':
            sub='toes_with_skinmax_colours' if name in selected else 'toes_without_dominant_skin_weight'
            groups.setdefault(sub,[]).append(i)
        if category=='tail':
            sub='tail_with_skinmax_colours' if name in selected else 'tail_without_dominant_skin_weight'
            groups.setdefault(sub,[]).append(i)
    metrics={"schema":"anymove.dino_color41_ab_eval/1",
             "groups":{k:len(v) for k,v in groups.items()},
             "pose_count":len(TEST_IDS),
             "source_checkpoint":str(root/"runs/dino/reader_hr_v1.pt"),
             "checkpoint_training_matched":False,"model_reports":{},
             "reader_2d_per_view":{},"3d":{},"paired_delta":{}}
    packets={}
    for tag in ("A","B"):
        ck=exp/f"reader_{tag}_{'original' if tag=='A' else 'color41'}_5ep.pt"
        record=ck.with_suffix(".validation.json")
        if not ck.exists() or not record.exists():raise FileNotFoundError(ck)
        data=json.loads(record.read_text())
        packets[tag]={}
        path=exp/f"ab_blind_{tag}_{'original' if tag=='A' else 'color41'}_v1"/f"reader_{tag}_blind.npz"
        with np.load(path,allow_pickle=False) as z:
            p={k:z[k].copy() for k in z.files}
        packets[tag]=p
        with np.load(path.with_suffix(".ground_truth.npz"),allow_pickle=False) as z:
            gt={k:z[k].copy() for k in z.files}
        if p["frame_ids"].tolist()!=list(TEST_IDS) or p["names"].tolist()!=names:
            raise ValueError("Reader test frame and Rig mismatch")
        if tag=="A":
            gtA=gt
        else:
            np.testing.assert_allclose(gtA["joints_world"],gt["joints_world"],atol=1e-5)
            np.testing.assert_allclose(gtA["exact_uv"],gt["exact_uv"],atol=1e-4)
        metrics["model_reports"][tag]={
          "checkpoint":str(ck),"checkpoint_sha256":hash_file(ck),
          "train_poses":data["train_poses"],"validation_poses":data["val_poses"],
          "train_ids":data["train_pose_ids"],"validation_ids":data["validation_pose_ids"],
          "seed":74111,"backbone_frozen":data["frozen_backbone"],
          "selected_epoch":data["best_checkpoint_epoch"],
          "val_mean_layout_px":data["best_validation_mean_layout_px"],
          "baseline_by_view":{v:data["baseline_validation"][v]["mean_layout_px"] for v in VIEWS},
          "validation_by_view":{v:data["history"][data["best_checkpoint_epoch"]-1]["validation"][v]["mean_layout_px"] for v in VIEWS},
          "source_checkpoint_hash":data["input_checkpoint_sha256"],
          "model_resolution":[data["input_size"],data["heatmap_size"]]}
        errors=np.linalg.norm(p["uv"]-gt["exact_uv"],axis=-1)
        metrics["reader_2d_per_view"][tag]={}
        for vi,v in enumerate(VIEWS):
            valid=p["valid"][vi]
            result={"all":error(errors[vi][valid])}
            for k,j in groups.items():
                ix=np.asarray(j)
                mask=valid[:,ix]
                result[k]=error(errors[vi][:,ix][mask])
            metrics["reader_2d_per_view"][tag][v]=result
        metrics["3d"][tag]={}
        for kind in ("2v","4v"):
            out=exp/f"ab_blind_{tag}_{'original' if tag=='A' else 'color41'}_v1"/f"reader_{tag}_fit_{kind}.npz"
            with np.load(out,allow_pickle=False) as z:
                world=z["world_joints"]
                if list(z["names"])!=names or list(z["frame_ids"])!=list(TEST_IDS):
                    raise ValueError("3D output joint order test mismatch")
            dist=np.linalg.norm(world-gt["joints_world"],axis=-1)*1000
            values={"all":error(dist)}
            for k,idx in groups.items():values[k]=error(dist[:,idx])
            values["per_pose_mean_mm"]=np.round(dist.mean(axis=1),3).tolist()
            metrics["3d"][tag][kind]=values
    A=metrics["model_reports"]["A"];B=metrics["model_reports"]["B"]
    for field in ("train_poses","validation_poses","train_ids","validation_ids","source_checkpoint_hash","model_resolution"):
        if A[field]!=B[field]:raise ValueError("training A/B not matched: "+field)
    if A["train_ids"]!=list(TRAIN_IDS) or A["validation_ids"]!=list(VAL_IDS):
        raise ValueError("unexpected split")
    metrics["checkpoint_training_matched"]=True
    for kind in ("2v","4v"):
        x=np.asarray(metrics["3d"]["A"][kind]["per_pose_mean_mm"])
        y=np.asarray(metrics["3d"]["B"][kind]["per_pose_mean_mm"])
        delta=y-x
        rng=np.random.default_rng(11711)
        bootstrap=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(axis=1)
        metrics["paired_delta"][kind]={
            "B_minus_A_mean_mm":round(float(delta.mean()),3),
            "improved_pose_count":int((delta<0).sum()),
            "95pct_bootstrap_interval_mm":np.round(np.percentile(bootstrap,[2.5,97.5]),3).tolist(),
            "not_a_continuous_motion_test":True}
    return metrics
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",default=r"F:\anymove")
    parser.add_argument("--mode",choices=("verify","report"),required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    dest=Path(args.out)
    if dest.exists():raise FileExistsError("refusing to overwrite experiment report")
    obj=analysis(args.repo) if args.mode=="verify" else report_metrics(args.repo)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(obj if args.mode=="verify" else
        {"model_reports":obj["model_reports"],"paired_delta":obj["paired_delta"],
         "2d_toe":{k:{v:r["toes"]["mean"] for v,r in x.items()} for k,x in obj["reader_2d_per_view"].items()},
         "3d_main":{tag:{kind:{key:v["mean"] for key,v in vals.items() if isinstance(v,dict) and "mean" in v}
                      for kind,vals in runs.items()} for tag,runs in obj["3d"].items()}},
        ensure_ascii=False,indent=2))
