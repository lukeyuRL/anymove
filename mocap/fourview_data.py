"""Four-view Reader supervision from existing Blender rendered poses.

This dataset deliberately keeps poses (not individual views) as the unit
of train/validation split. Appearance is tracked explicitly: source-material
for color-coded Reader training, neutral-geometry for old gray-only audits.
Visibility targets are explicitly Skinning
SURFACE PART visibility, not skeleton center ray-cast or real H3 occlusion.
Unknown (-1) part visibility is masked from training and metrics.

The new P3 model uses the existing AnyMove 672/192 DINOv2 architecture;
no modification to the original two-view checkpoints or mocap.train.
"""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mocap.reader_geometry import Letterbox
from mocap.train import photometric,backdrop

VIEWS=("front","side","back","opposite_side")
SCHEMA="anymove.four_view_surface_part_gt/1"


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def split_poses(dataset):
    """Use original Reader val rule: index %25 ==0. All four views stay together."""
    path=Path(dataset)
    files=sorted(path.glob("*_visibility_meta.json"))
    ids=[int(p.name[:5]) for p in files]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate 4-view pose IDs")
    train=tuple(i for i in ids if i%25!=0)
    val=tuple(i for i in ids if i%25==0)
    if not train or not val or set(train)&set(val):
        raise ValueError("requires separate nonempty pose-disjoint Train/Validation")
    return train,val


class FourViewDataset(Dataset):
    def __init__(self, dataset, rig_info, pose_ids, *, train, input_size=672,
                 seed=20261101, include_quality=True, views=None):
        self.root=Path(dataset)
        self.rig_path=Path(rig_info)
        self.rig=json.loads(self.rig_path.read_text(encoding="utf-8"))
        self.names=[p["name"] for p in self.rig["keypoints"]]
        self.pose_ids=tuple(int(x) for x in pose_ids)
        if not self.pose_ids or len(set(self.pose_ids))!=len(self.pose_ids):
            raise ValueError("empty/duplicate pose list")
        self.train=bool(train)
        self.seed=int(seed)
        self.epoch=0
        self.input_size=int(input_size)
        self.include_quality=include_quality
        self.views=tuple(VIEWS if views is None else views)
        if not self.views or len(set(self.views))!=len(self.views) or any(v not in VIEWS for v in self.views):
            raise ValueError('invalid selected four-view dataset views')
        self.rows=[]
        self.appearance_modes=set()
        digest=sha256(self.rig_path)
        for i in self.pose_ids:
            m=self.root/f"{i:05d}_visibility_meta.json"
            meta=json.loads(m.read_text(encoding="utf-8"))
            if (meta.get("schema_version")!=SCHEMA or
                    int(meta["sample_index"])!=i or
                    meta.get("rig_info_sha256")!=digest):
                raise ValueError("4-view GT metadata, sample ID or rig hash mismatch")
            appearance=meta.get("appearance_mode","neutral-geometry")
            if appearance not in ("neutral-geometry","source-material"):
                raise ValueError("unrecognized four-view renderer appearance")
            self.appearance_modes.add(appearance)
            pose_src=Path(meta["source_pose_metadata"])
            if not pose_src.is_file() or sha256(pose_src)!=meta["source_meta_sha256"]:
                raise ValueError("4-view source pose integrity check failed")
            for vi,view in enumerate(VIEWS):
                if view not in self.views:
                    continue
                detail=meta["views"].get(view)
                if detail is None:
                    raise ValueError("four-view sample incomplete")
                png=self.root/f"{i:05d}_{view}.png"
                label=self.root/f"{i:05d}_{view}_labels.npz"
                if not png.is_file() or not label.is_file():
                    raise ValueError("4-view missing image/visibility npz")
                self.rows.append((i,vi,png,label,detail))
        if len(self.appearance_modes)!=1:
            raise ValueError("mixed gray and textured four-view samples are not a valid training set")
        self.appearance_mode=next(iter(self.appearance_modes))
        self.sample_keys=tuple((p,v) for p in self.pose_ids for v in self.views)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        pose,vi,img_path,label_path,detail=self.rows[index]
        rng=np.random.default_rng(self.seed+100003*self.epoch+9967*index)
        raw=cv2.imread(str(img_path),cv2.IMREAD_UNCHANGED)
        w,h=map(int,detail["image_size_px"])
        if raw is None or raw.shape!=(h,w,4):
            raise ValueError(f"invalid RGBA four-view image {img_path}")
        with np.load(label_path,allow_pickle=False) as npz:
            kp=np.asarray(npz["joints_uv"],dtype=np.float32)
            labels=np.asarray(npz["joint_surface_part_visible"],dtype=np.int8)
        if kp.shape!=(len(self.names),2) or labels.shape!=(len(self.names),):
            raise ValueError("keypoint/visibility label dimensions mismatch")
        if not np.isin(labels,(-1,0,1)).all():
            raise ValueError("unrecognized surface visibility label")
        # Render images have transparency. Match the current independent
        # P2 Reader gray-background benchmark for a fair held-out comparison.
        rgba=raw[:,:,[2,1,0,3]].astype(np.float32)
        alpha=rgba[...,3:4]/255.
        bg=np.empty((h,w,3),np.float32)
        if self.train:
            bg[:]=rng.uniform(105,175,size=3)
            # Texture/domain variation, not a claim of real-H3 matching.
            bg+=np.linspace(-8,8,h,dtype=np.float32)[:,None,None]
        else:
            bg[:]=(134,130,128) # corresponds to BGR [128,130,134] in P2 Reader benchmark
        img=rgba[...,:3]*alpha+(1-alpha)*bg
        pad_fill=tuple(float(v) for v in bg[0,0])
        geo=Letterbox(w,h,self.input_size)
        A=np.asarray(geo.affine,dtype=np.float32)
        if self.train:
            angle=float(rng.uniform(-3,3))
            sc=float(rng.uniform(.92,1.08))
            R=cv2.getRotationMatrix2D((self.input_size/2,self.input_size/2),angle,sc)
            R[:,2]+=rng.uniform(-.025,.025,size=2)*self.input_size
            A=(R@np.vstack((A,[0.,0.,1.]))).astype(np.float32)
        mapped=kp@A[:,:2].T+A[:,2]
        square=cv2.warpAffine(img,A,(self.input_size,self.input_size),
                            flags=cv2.INTER_AREA if self.input_size/geo.side<1 else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,borderValue=pad_fill)
        if self.train:
            square=photometric(square,rng)
        square=np.clip(square,0,255).astype(np.uint8)
        pixel_valid=(mapped[:,0]>=0)&(mapped[:,0]<self.input_size)&(
                     mapped[:,1]>=0)&(mapped[:,1]<self.input_size)
        return {
            "image":torch.from_numpy(np.ascontiguousarray(square)).permute(2,0,1).float()/255.,
            "keypoints":torch.from_numpy(mapped.astype(np.float32)),
            "visibility":torch.from_numpy(labels.astype(np.int64)),
            "inside":torch.from_numpy(pixel_valid),
            "geo":torch.tensor([geo.ox,geo.oy,geo.side,float(detail["meters_per_pixel"])],
                               dtype=torch.float32),
            "source_uv":torch.from_numpy(kp),
            "view_index":torch.tensor(vi,dtype=torch.int64),
            "pose_id":torch.tensor(pose,dtype=torch.int64),
        }


def expected_visibility_and_error_summary(scores, visible_labels, predicted_error, actual_error):
    """Compute known surface-part reliability and localization error only.

    Uses observed vs expected MEAN pixel localization error; does not claim
    Gaussian sigma calibration or independent H3 occlusion truth.
    """
    from mocap.quality_metrics import error_calibration_report
    labels=np.asarray(visible_labels).ravel()
    prob=np.asarray(scores,dtype=float).ravel()
    known=labels>=0
    result={"known_surface_part_samples":int(known.sum()),
            "unknown_surface_part_samples":int((~known).sum())}
    if known.any():
        s=prob[known]
        y=labels[known].astype(float)
        result.update({
            "occluded_surface_part_samples":int((y==0).sum()),
            "visible_surface_part_samples":int((y==1).sum()),
            "brier":round(float(np.mean((s-y)**2)),5),
            "mean_predicted_visibility":round(float(s.mean()),5),
            "true_visible_fraction":round(float(y.mean()),5),
            "ece_8bins":None,
            "auc":None,
        })
        # Fixed bins are test-only. The model is not calibrated on real video.
        expected=0.
        for b in range(8):
            lo=b/8;hi=(b+1)/8
            ids=(s>=lo)&((s<hi) if b<7 else (s<=hi))
            if ids.any():
                expected+=np.sum(ids)*abs(float(s[ids].mean()-y[ids].mean()))
        result["ece_8bins"]=round(float(expected/max(len(s),1)),5)
        if (y==0).any() and (y==1).any():
            # Tie-correct binary ranking ROC AUC.
            order=np.argsort(s,kind="stable")
            ranks=np.empty(len(order),dtype=float)
            sorted_scores=s[order]
            pos=0
            while pos<len(order):
                stop=pos+1
                while stop<len(order) and sorted_scores[stop]==sorted_scores[pos]:
                    stop+=1
                ranks[order[pos:stop]]=(pos+stop+1)/2.
                pos=stop
            n1=float(y.sum());n0=len(y)-n1
            result["auc"]=round(float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)),5)
    predicted=np.asarray(predicted_error,dtype=float).ravel()
    actual=np.asarray(actual_error,dtype=float).ravel()
    finite=np.isfinite(predicted)&np.isfinite(actual)&(predicted>=0)&(actual>=0)
    if finite.sum()>3:
        result["localization_error_calibration"]=error_calibration_report(
            predicted[finite],actual[finite],bins=8)
    else:
        result["localization_error_calibration"]=None
    return result
