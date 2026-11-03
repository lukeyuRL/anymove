"""Dino Top-specific Joint Reader SMOKE fine-tuning, kept separate from production.

python -m mocap.train_dino_top \
  runs/dino/dataset_top_smoke_v1 assets/dino/rigged/rig_info.json \
  runs/dino/reader_hr_v1.pt runs/dino/top_reader_smoke_v1.pt \
  --epochs 2 --batch 2

Trains the existing heatmap decoder (neck, upsample, out) while freezing
DINOv2 Backbone, then writes a NEW checkpoint with trained_views=["top"].
No synthetic cutout visibility claim, no original Reader file changed.
A 30-pose smoke (28 train/2 val) is NOT a production-quality model.
"""
import argparse
import hashlib
import json
import math
import pathlib
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from mocap import layout
from mocap.model import JointReader,decode
from mocap.reader_config import ReaderSpec
from mocap.reader_geometry import Letterbox
from mocap.reader_warmstart import validate_base_reader
from mocap.train import heatmaps,photometric

PROFILE="anymove.dino_top_reader_head_only_smoke_v1"


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as fd:
        for chunk in iter(lambda:fd.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


class TopSamples(Dataset):
    def __init__(self,metas,names,*,train,size=672):
        self.metas=list(metas)
        self.names=list(names)
        self.train=bool(train)
        self.size=int(size)
        if self.size%14:
            raise ValueError("Top Reader input size must be divisible by 14")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self,index):
        file=self.metas[index]
        meta=json.loads(file.read_text(encoding="utf-8"))
        if (meta.get("profile")!="anymove.dino_top_same_pose_static_v1"
                or meta["names"]!=self.names
                or meta.get("render_size")!=[880,720]):
            raise ValueError("Top training sample is not from matching Dino rig/camera")
        image=str(file).replace("_top_meta.json","_top.png")
        rgba=cv2.imread(image,cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape!=(720,880,4):
            raise ValueError("missing Top view sample "+image)
        rgb=rgba[..., [2,1,0,3]].astype(np.float32)
        alpha=rgb[...,3:]/255
        rng=np.random.default_rng() if self.train else np.random.default_rng(index)
        if self.train:
            bg=np.full((720,880,3),rng.uniform(100,165),np.float32)
        else:
            bg=np.empty((720,880,3),np.float32);bg[:]=layout.BG
        composite=rgb[...,:3]*alpha+bg*(1-alpha)
        kp=np.asarray(meta["top_uv"],np.float32)
        if kp.shape!=(len(self.names),2) or not np.isfinite(kp).all() or np.any(
                (kp[:,0]<0)|(kp[:,0]>=880)|(kp[:,1]<0)|(kp[:,1]>=720)):
            raise ValueError("Top metadata contains cropped or invalid keypoints")
        geo=Letterbox(880,720,self.size)
        A=np.asarray(geo.affine,dtype=np.float32)
        k=self.size/geo.side
        if self.train:
            # Same modest photometric / 2D affine augmentation as existing
            # Front/Side Reader. No new pose labels are fabricated.
            scale=rng.uniform(.90,1.10)
            rot=rng.uniform(-4.,4.)
            jitter=cv2.getRotationMatrix2D((self.size/2,self.size/2),rot,scale)
            jitter[:,2]+=rng.uniform(-.035,.035,2)*self.size
            A=(jitter @ np.vstack([A,[0,0,1]])).astype(np.float32)
        x=cv2.warpAffine(composite,A,(self.size,self.size),
                          flags=cv2.INTER_AREA if k<1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=tuple(float(z) for z in bg[0,0]))
        targets=kp @ A[:,:2].T + A[:,2]
        if self.train:
            x=photometric(x,rng)
        img=torch.from_numpy(np.ascontiguousarray(x)).permute(2,0,1).float()/255
        meta_geo=torch.tensor([geo.ox,geo.oy,geo.side],dtype=torch.float32)
        return img,torch.from_numpy(targets.astype(np.float32)),meta_geo


def validate(model,loader,spec,names,device,amp_dtype):
    errors=[]
    tail_idx=[i for i,n in enumerate(names) if "Tail" in n.split(":")[1]]
    if not tail_idx:
        raise ValueError("Dino Top training requires anatomical Tail group")
    tail_errors=[]
    model.eval()
    with torch.no_grad():
        for x,kp,geo in loader:
            with torch.autocast("cuda",dtype=amp_dtype):
                pred=model(x.to(device)).float()
            uv_hm,_=decode(pred)
            uv=(uv_hm.cpu()*spec.input_size/spec.heatmap_size)
            err=((uv-kp)*geo[:,None,2:3]/spec.input_size).norm(dim=-1)
            errors.extend(err.flatten().tolist())
            tail_errors.extend(err[:,tail_idx].flatten().tolist())
    return {
        "val_mean_2d_error_px":round(float(np.mean(errors)),5),
        "val_p95_2d_error_px":round(float(np.percentile(errors,95)),5),
        "val_tail_mean_2d_error_px":round(float(np.mean(tail_errors)),5),
        "val_tail_p95_2d_error_px":round(float(np.percentile(tail_errors,95)),5),
        "val_images":len(loader.dataset),
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("top_dataset")
    p.add_argument("rig_info")
    p.add_argument("base_reader")
    p.add_argument("output_checkpoint")
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--input-size",type=int,default=672)
    p.add_argument("--heatmap-size",type=int,default=192)
    a=p.parse_args(argv)
    if (a.epochs<1 or a.epochs>30 or a.batch<1 or
            a.batch>8 or not torch.cuda.is_available()):
        raise ValueError("Top head-only smoke needs CUDA and valid epochs/batch")
    spec=ReaderSpec(a.input_size,a.heatmap_size)
    output=pathlib.Path(a.output_checkpoint).resolve()
    if output.exists() or output in {pathlib.Path(a.base_reader).resolve(),
                                      pathlib.Path(a.rig_info).resolve()}:
        raise FileExistsError("will not overwrite any original Reader/Rig")
    dataset=pathlib.Path(a.top_dataset)
    src_audit=dataset/"dataset_top_audit.json"
    if not src_audit.is_file():
        raise FileNotFoundError("Top same-pose dataset provenance missing")
    meta=json.loads(src_audit.read_text(encoding="utf-8"))
    if (meta.get("profile")!="anymove.dino_top_same_pose_static_v1"
        or meta["rig_info_sha256"]!=digest(a.rig_info)):
        raise ValueError("Top dataset Rig checksum/profile mismatch")
    metas=sorted(dataset.glob("*_top_meta.json"))
    if len(metas)!=meta["count"] or len(metas)<3:
        raise ValueError("incomplete Top training dataset")
    train=[x for x in metas if int(x.name[:5])%25!=0]
    val=[x for x in metas if int(x.name[:5])%25==0]
    if not train or not val:
        raise ValueError("Top training requires held-out sample indices divisible by 25")
    rig=json.loads(pathlib.Path(a.rig_info).read_text(encoding="utf-8"))
    names=[k["name"] for k in rig["keypoints"]]
    if len(names)!=92:
        raise ValueError("Dino Reader must have exactly 92 joints")
    original=torch.load(a.base_reader,map_location="cpu",weights_only=True)
    state=validate_base_reader(original,names,spec)
    model=JointReader(len(names),input_size=spec.input_size,
                      heatmap_size=spec.heatmap_size,quality_head=False).cuda()
    model.load_state_dict(state,strict=True)
    model.backbone.set_attn_implementation("sdpa")
    model.backbone.requires_grad_(False)
    head=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(head,lr=3e-4,weight_decay=.02)
    amp=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=amp==torch.float16)
    loader=DataLoader(TopSamples(train,names,train=True,size=spec.input_size),
                      batch_size=a.batch,shuffle=True,num_workers=0)
    validation=DataLoader(TopSamples(val,names,train=False,size=spec.input_size),
                          batch_size=a.batch,shuffle=False,num_workers=0)
    history=[]
    started=time.time()
    print("TOP READER SMOKE",len(train),"train",len(val),"val",
          "AMP",str(amp),"frozen backbone","epochs",a.epochs,flush=True)
    initial=validate(model,validation,spec,names,"cuda",amp)
    print("TOP reader initial held-out:",initial,flush=True)
    for epoch in range(a.epochs):
        model.train()
        model.backbone.eval()
        for i,(x,kp,_) in enumerate(loader):
            x,kp=x.cuda(non_blocking=True),kp.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=amp):
                pred=model(x)
            loss=F.mse_loss(
                pred.float(),heatmaps(kp,spec.input_size,spec.heatmap_size))*1000
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head,1.0)
            scaler.step(optimizer)
            scaler.update()
            if i%5==0:
                print("TOP EPOCH",epoch,"step",i,"loss",
                      round(float(loss.item()),5),flush=True)
        validation_stats=validate(model,validation,spec,names,"cuda",amp)
        history.append({"epoch":epoch,**validation_stats})
        print("TOP EPOCH",epoch,"VAL",validation_stats,
              "VRAM GiB",round(torch.cuda.max_memory_allocated()/2**30,3),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint={
        "state_dict":model.state_dict(),
        "names":names,
        "reader_spec":dict(spec.checkpoint_metadata(),quality_head=False),
        "epoch":a.epochs-1,
        "val":{"top":history[-1]["val_mean_2d_error_px"]},
        "top_view_training":{
            "profile":PROFILE,
            "trained_views":["top"],
            "source_pose_policy":"Original Dino HR 2200-pose sample metadata, not a new motion corpus",
            "dataset_sha256":digest(src_audit),
            "base_reader_sha256":digest(a.base_reader),
            "frozen_backbone":True,
            "epochs":a.epochs,
            "train_images":len(train),"val_images":len(val),
            "smoke_only":len(metas)<300,
            "initial_validation":initial,
            "validation_history":history,
        },
    }
    temp=output.with_suffix(output.suffix+".tmp")
    if temp.exists():
        raise FileExistsError("Top Reader atomic checkpoint temp already exists")
    torch.save(checkpoint,temp)
    temp.replace(output)
    report={
        "profile":PROFILE,
        "checkpoint":str(output),
        "checkpoint_sha256":digest(output),
        "top_val_initial":initial,"history":history,
        "elapsed_seconds":round(time.time()-started,2),
        "warning":"30 images/2 held-out validation poses are pipeline smoke only, NOT Top Reader generalization or 3D Motion truth",
    }
    output.with_suffix(".training.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
