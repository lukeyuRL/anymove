"""Train native 672/192 Human Reader V2 with strict clip-split validation.

Local NVIDIA RTX 3070. Frozen pretrained DINOv2 with registers, trainable
heatmap neck. No test data used to tune or choose checkpoints.
Uses only human V3 Nose 315/45 dataset with explicit source SHA provenance.
"""
import argparse, hashlib, json, pathlib, random, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mocap.reader_hr_v2 import (
    HumanReaderHRV2, INPUT_SIZE, HEATMAP_SIZE, decode_hr
)

DATA=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45/dataset")
OUT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45/reader_v2")
SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"
SEED=20261104

def load_manifest():
    audited=json.loads((DATA/"audit.json").read_text(encoding="utf-8"))
    record=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
    if audited.get("status")!="PASS" or audited.get("image_count")!=5376:
        raise RuntimeError("Dataset V2 release audit is not PASS")
    if record.get("source_blend_sha256")!=SHA or record["status"]!="complete":
        raise RuntimeError("Invalid V3 Nose model provenance")
    if record["camera_angles_deg"]!=[315,45] or record["width"]!=672 or record["height"]!=672:
        raise RuntimeError("Only audited 315/45 native 672 dataset is supported")
    if len(record["joint_names"])!=67 or len(record["samples"])!=5376:
        raise RuntimeError("Dataset must have exactly 67 markers and 5376 images")
    parts=[set(record[s+"_clips"]) for s in ("train","validation","test")]
    if [len(x) for x in parts]!=[144,24,24] or any(parts[i]&parts[j] for i in range(3) for j in range(i+1,3)):
        raise RuntimeError("Train/validation/test clip leakage")
    return record

class V2Images(Dataset):
    def __init__(self,manifest,split,limit=None,augment=False):
        self.samples=[r for r in manifest["samples"] if r["split"]==split]
        expect={"train":4032,"validation":672,"test":672}[split]
        if len(self.samples)!=expect:
            raise RuntimeError(f"Bad {split} image count {len(self.samples)} != {expect}")
        if limit is not None:
            self.samples=self.samples[:limit]
        self.augment=augment
        self.epoch=0
        self.names=manifest["joint_names"]
        self.index={n:i for i,n in enumerate(self.names)}
        self.groups={
            "finger":[i for i,n in enumerate(self.names) if any(t in n for t in ("Pinky","Ring","Mid","Index","Thumb"))],
            "toe":[i for i,n in enumerate(self.names) if "ToeTip" in n],
            "leg":[i for i,n in enumerate(self.names) if any(t in n for t in ("Thigh","Calf","Foot","ToeBase","ToeTip"))],
            "arm":[i for i,n in enumerate(self.names) if any(t in n for t in ("Upperarm","Forearm","_Hand"))],
            "head":[i for i,n in enumerate(self.names) if "Head" in n or "NeckTwist" in n],
        }
        if len(self.groups["finger"])!=40 or len(self.groups["toe"])!=2:
            raise RuntimeError("Human fullbody/finger group count invalid")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,index):
        row=self.samples[index]
        path=DATA/row["file"]
        rgba=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape!=(672,672,4):
            raise RuntimeError("Invalid source PNG "+str(path))
        alpha=rgba[...,3,None].astype(np.float32)/255.
        rgb=rgba[...,[2,1,0]].astype(np.float32)
        img=rgb*alpha+(1-alpha)*np.array([105.,107.,111.],dtype=np.float32)
        if self.augment:
            rng=np.random.default_rng(SEED+int(index)*131+int(self.epoch)*30011)
            img=np.clip(img*rng.uniform(.93,1.07)+rng.uniform(-4.,4.),0,255)
        image=torch.from_numpy(np.ascontiguousarray(img)).permute(2,0,1).float()/255.
        pts=np.asarray(row["uv"],dtype=np.float32)
        if pts.shape!=(67,2) or not np.isfinite(pts).all():
            raise RuntimeError("Invalid 67 keypoint UV labels")
        return image,torch.from_numpy(pts),int(index)

@torch.inference_mode()
def evaluate(model,loader):
    model.eval()
    stats={"all":[],"finger":[],"toe":[],"leg":[],"arm":[],"head":[]}
    for image,gt,indices in loader:
        with torch.autocast("cuda",dtype=torch.float16):
            logits=model(image.cuda(non_blocking=True))
        pred,_=decode_hr(logits.float())
        residual=torch.linalg.norm(pred.cpu()-gt,dim=-1)
        stats["all"].extend(residual.numpy().ravel().tolist())
        for key in ("finger","toe","leg","arm","head"):
            stats[key].extend(residual[:,loader.dataset.groups[key]].numpy().ravel().tolist())
    return {key:{"mean_px":round(float(np.mean(vals)),4),
                  "p95_px":round(float(np.percentile(vals,95)),4)}
            for key,vals in stats.items()}

def train(epochs,batch,smoke,smoke_steps):
    record=load_manifest()
    if not torch.cuda.is_available():
        raise RuntimeError("Local NVIDIA CUDA GPU is required")
    train_ds=V2Images(record,"train",limit=smoke_steps*batch if smoke else None,augment=True)
    val_ds=V2Images(record,"validation",limit=min(16,672) if smoke else None,augment=False)
    train_loader=DataLoader(train_ds,batch_size=batch,shuffle=True,num_workers=0,pin_memory=True,
                            generator=torch.Generator().manual_seed(SEED))
    val_loader=DataLoader(val_ds,batch_size=batch,shuffle=False,num_workers=0,pin_memory=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    model=HumanReaderHRV2(67).cuda()
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=0.0005,weight_decay=0.01)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    directory=OUT/("smoke" if smoke else "full")
    directory.mkdir(parents=True,exist_ok=True)
    summary={
        "schema":"anymove.human_reader_hr_v2_training/1",
        "training_status":"SMOKE" if smoke else "RUNNING",
        "camera_angles_deg":[315,45],
        "source_blend_sha256":SHA,
        "input_resolution":INPUT_SIZE,"heatmap_resolution":HEATMAP_SIZE,
        "architecture":"DINOv2 with registers base, frozen backbone; native high resolution heatmap",
        "seed":SEED,"batch":batch,"epochs_planned":epochs,
        "train_images":len(train_ds),"validation_images":len(val_ds),
        "test_images_untouched":672,
        "train_clips":len(record["train_clips"]),
        "validation_clips":len(record["validation_clips"]),
        "test_clips":len(record["test_clips"]),
        "train_only_crop_hints":"Not used as inference inputs",
        "visibility_proxy":"Not trained as independent visibility head",
        "history":[]}
    print("READER_HRV2_SETUP",json.dumps({k:v for k,v in summary.items() if k!="history"}),flush=True)
    best=float("inf")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        begin=time.monotonic()
        train_ds.epoch=epoch
        model.train()
        losses=[]
        for step,(image,uv,indices) in enumerate(train_loader):
            image=image.cuda(non_blocking=True)
            y=(uv.cuda(non_blocking=True)*(HEATMAP_SIZE/INPUT_SIZE)).round().long().clamp(0,HEATMAP_SIZE-1)
            target=(y[...,1]*HEATMAP_SIZE+y[...,0]).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                heat=model(image)
                logits=heat.float().reshape(-1,HEATMAP_SIZE*HEATMAP_SIZE)
                loss=F.cross_entropy(logits,target)
            if not torch.isfinite(loss):raise RuntimeError("Non-finite heatmap CE")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params,2.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            if (step+1)%40==0:
                print("READER_HRV2_STEP",epoch,step+1,len(train_loader),
                      "loss",round(float(np.mean(losses[-40:])),4),
                      "allocated_GiB",round(float(torch.cuda.max_memory_allocated()/2**30),2),flush=True)
        metrics=evaluate(model,val_loader)
        entry={"epoch":epoch,"train_CE":round(float(np.mean(losses)),5),
               "validation":metrics,"duration_s":round(time.monotonic()-begin,1),
               "peak_gpu_gib":round(float(torch.cuda.max_memory_allocated()/2**30),3)}
        summary["history"].append(entry)
        print("READER_HRV2_EPOCH",json.dumps(entry),flush=True)
        if metrics["all"]["mean_px"]<best:
            best=metrics["all"]["mean_px"]
            state={
                "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "names":record["joint_names"],
                "model":"HumanReaderHRV2",
                "camera_azimuth_deg":[315,45],
                "source_blend_sha256":SHA,
                "input_size":INPUT_SIZE,"heatmap_size":HEATMAP_SIZE,
                "train_clips":record["train_clips"],
                "validation_clips":record["validation_clips"],
                "test_clips":record["test_clips"],
                "epoch":epoch,"seed":SEED,"validation":metrics,
            }
            torch.save(state,directory/"best.pt")
        summary["best_validation_px"]=best
        summary["training_status"]="SMOKE_COMPLETE" if smoke else ("COMPLETE" if epoch==epochs-1 else "IN_PROGRESS")
        (directory/"training_report.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("READER_HRV2_COMPLETE",json.dumps({
        "status":summary["training_status"],"epochs":epochs,
        "best_validation_px":best,"checkpoint":str(directory/"best.pt"),
        "peak_gpu_gib":round(float(torch.cuda.max_memory_allocated()/2**30),3)
    }),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--batch",type=int,default=1)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--smoke-steps",type=int,default=4)
    a=ap.parse_args()
    if a.epochs<1 or a.batch<1 or a.smoke_steps<1:raise ValueError("Positive epochs/batch/steps required")
    train(a.epochs,a.batch,a.smoke,a.smoke_steps)
