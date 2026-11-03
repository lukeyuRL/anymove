"""Strict 67-keypoint V3-nose fullbody Reader camera A/B, local GPU only.

16 train, 4 validation, 4 *unseen* test clips; no shared frames or
identical first/last rest poses in split evaluation. Includes 30 finger
phalange-head joints, 10 fingertips, 2 toe endpoints, legs/arms/torso.
Checkpoints chosen on validation ONLY and evaluated later by separate script.
"""
import argparse,hashlib,json,pathlib,random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

from mocap.model import JointReader,SIZE,HM,decode,square
from mocap.camera_geometry import triangulate_orthographic

ROOT=pathlib.Path("runs/human/fullbody_finger_v3_nose")
SOURCE_SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"
PAIR_NAMES=("baseline_0_90","symmetric_315_45")
SEED=20261104

def manifest(pair):
    p=ROOT/"dataset"/pair/"manifest.json"
    data=json.loads(p.read_text(encoding="utf-8"))
    if data.get("status")!="complete" or data.get("source_blend_sha256")!=SOURCE_SHA:
        raise RuntimeError("Unverified V3 nose fullbody dataset")
    if len(data["joint_names"])!=67 or len(set(data["joint_names"]))!=67:
        raise RuntimeError("67-point fullbody contract missing")
    for split,n in (("train",16),("validation",4),("test",4)):
        clips=data[split+"_clips"]
        if len(clips)!=n or len(set(clips))!=n:raise RuntimeError("Wrong clip count")
    if (set(data["train_clips"]) & set(data["test_clips"]) or
        set(data["train_clips"]) & set(data["validation_clips"]) or
        set(data["test_clips"]) & set(data["validation_clips"])):
        raise RuntimeError("Train/validation/test clip leakage")
    if any(row["camera_azimuth_deg"]!=data["angles_deg"][0 if row["slot"]=="front" else 1]
           for row in data["samples"]):
        raise RuntimeError("Wrong camera angle/image slot labeling")
    return data

class FullBodyImages(Dataset):
    def __init__(self,pair,data,split,train=False):
        self.base=ROOT/"dataset"
        self.rows=[r for r in data["samples"] if r["split"]==split and r["frame"] not in (0,106)]
        self.train=train
        self.epoch=0
        self.names=data["joint_names"]
        if not self.rows or len(self.names)!=67:raise ValueError("Missing images or joints")
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        path=self.base/row["path"]
        rgba=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape!=(448,448,4):
            raise RuntimeError("Invalid image "+str(path))
        rgb=rgba[..., [2,1,0]].astype(np.float32)
        alpha=rgba[..., 3:4].astype(np.float32)/255.
        fill=np.array((105.,107.,111.),dtype=np.float32)
        image=rgb*alpha+fill*(1-alpha)
        if self.train:
            gen=np.random.default_rng(SEED+i*17+self.epoch*100003)
            image=np.clip(image*gen.uniform(.91,1.09)+gen.uniform(-7,7),0,255)
        image,ox,oy,side=square(image,fill)
        if side!=SIZE:raise RuntimeError("Unexpected squared image")
        gt=np.asarray(row["uv"],np.float32)+np.array((ox,oy),np.float32)
        if gt.shape!=(67,2) or not np.isfinite(gt).all():
            raise RuntimeError("Invalid fullbody labels")
        return (torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).float()/255.,
                torch.from_numpy(gt),i)

def init_reader(n=67):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    net=JointReader(n).cuda()
    for p in net.backbone.parameters():p.requires_grad_(False)
    net.backbone.eval()
    return net

@torch.inference_mode()
def eval_valid(model,dl,data):
    model.eval()
    errors=[]
    view_errors=defaultdict(list)
    for x,kp,ids in dl:
        with torch.autocast("cuda",dtype=torch.float16):
            heat=model(x.cuda(non_blocking=True))
        uv,_=decode(heat.float())
        uv=(uv.float().cpu().numpy()*(SIZE/HM))
        truth=kp.numpy()
        err=np.linalg.norm(uv-truth,axis=-1)
        errors.extend(err.ravel().tolist())
        for i,idx in enumerate(ids.tolist()):
            row=dl.dataset.rows[int(idx)]
            view_errors[row["slot"]].extend(err[i].tolist())
    e=np.asarray(errors,dtype=np.float64)
    return {"mean_2d_px":round(float(np.mean(e)),4),
            "p95_2d_px":round(float(np.percentile(e,95)),4),
            "front_2d_px":round(float(np.mean(view_errors["front"])),4),
            "side_2d_px":round(float(np.mean(view_errors["side"])),4),
            "validation_images":len(dl.dataset)}

def train_pair(pair,epochs,batch):
    data=manifest(pair)
    train=FullBodyImages(pair,data,"train",train=True)
    valid=FullBodyImages(pair,data,"validation")
    test=FullBodyImages(pair,data,"test")
    if len(train)!=16*12*2 or len(valid)!=4*6*2 or len(test)!=4*6*2:
        raise RuntimeError("Dynamic-frame split size unexpected")
    run=ROOT/"reader_models_strict"/pair
    run.mkdir(parents=True,exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    net=init_reader(67)
    param=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(param,lr=5e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda")
    loader=DataLoader(train,batch_size=batch,shuffle=True,num_workers=0,pin_memory=True,
                      generator=torch.Generator().manual_seed(SEED))
    vl=DataLoader(valid,batch_size=batch,shuffle=False,num_workers=0,pin_memory=True)
    history=[]
    best=float("inf")
    torch.cuda.reset_peak_memory_stats()
    print("FULLBODY_TRAIN_READY",json.dumps({
        "pair":pair,"angles":data["angles_deg"],
        "tracked_points":67,"train_images":len(train),
        "validation_images":len(valid),"unseen_test_images":len(test),
        "train_clips":data["train_clips"],
        "validation_clips":data["validation_clips"],
        "heldout_test_clips":data["test_clips"],
        "epochs":epochs}),flush=True)
    for epoch in range(epochs):
        train.epoch=epoch
        net.train();net.backbone.eval()
        losses=[]
        for step,(x,kp,idx) in enumerate(loader):
            x=x.cuda(non_blocking=True)
            pixels=(kp.cuda(non_blocking=True)*(HM/SIZE)).round().long().clamp(0,HM-1)
            truth=(pixels[...,1]*HM+pixels[...,0]).reshape(-1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                logit=net(x).float().reshape(-1,HM*HM)
                loss=F.cross_entropy(logit,truth)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite Reader loss")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(param,2.)
            scaler.step(opt);scaler.update()
            losses.append(float(loss.item()))
            if (step+1)%24==0:
                print("FULLBODY_TRAIN_STEP",pair,epoch,step+1,
                      round(float(np.mean(losses[-24:])),4),flush=True)
        measurements=eval_valid(net,vl,data)
        measurements.update({"epoch":epoch,"train_loss":round(float(np.mean(losses)),5),
                    "peak_vram_gib":round(float(torch.cuda.max_memory_allocated()/2**30),4)})
        history.append(measurements)
        print("FULLBODY_VALIDATION",pair,json.dumps(measurements),flush=True)
        if measurements["mean_2d_px"]<best:
            best=measurements["mean_2d_px"]
            ck={"state_dict":{k:v.detach().cpu() for k,v in net.state_dict().items()},
                "names":data["joint_names"],"joint_groups":data["joint_groups"],
                "camera_azimuth_deg":data["angles_deg"],
                "source_blend_sha256":SOURCE_SHA,
                "train_clips":data["train_clips"],
                "validation_clips":data["validation_clips"],
                "test_clips":data["test_clips"],
                "epoch":epoch,"seed":SEED,
                "training_scope":"DINOv2-with-registers-base frozen + 67 output heatmaps",
                "selected_using":"validation 2D only",
                "validation_metrics":measurements}
            torch.save(ck,run/"best.pt")
            print("FULLBODY_CHECKPOINT_SAVED",pair,"epoch",epoch,flush=True)
    result={"schema":"anymove.human_v3_nose_fullbody_finger_train/1",
        "state":"COMPLETE_HELDOUT_TEST_UNTOUCHED",
        "checkpoint_selection":"lowest 2D validation mean",
        "pair":pair,"angles_deg":data["angles_deg"],
        "source_blend_sha256":SOURCE_SHA,
        "train_images":len(train),"validation_images":len(valid),
        "heldout_images_unseen":len(test),
        "train_clips":data["train_clips"],"val_clips":data["validation_clips"],
        "test_clips":data["test_clips"],
        "history":history,
        "best_epoch":min(history,key=lambda q:q["mean_2d_px"])["epoch"],
        "best_val_2d_px":best}
    (run/"validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    del net
    torch.cuda.empty_cache()
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--batch",type=int,default=4)
    ap.add_argument("--only-pair",choices=PAIR_NAMES,default=None)
    args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("Requires local NVIDIA GPU")
    torch.set_num_threads(4)
    pairs=[args.only_pair] if args.only_pair else list(PAIR_NAMES)
    outcomes={p:train_pair(p,args.epochs,args.batch) for p in pairs}
    print("FULLBODY_AB_TRAIN_DONE",json.dumps({p:{
        "epoch":r["best_epoch"],"validation_2d_px":r["best_val_2d_px"]}
        for p,r in outcomes.items()}),flush=True)
