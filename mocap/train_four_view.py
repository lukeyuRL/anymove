"""Train and validate a 4-view AnyMove high-resolution Reader + geometry
surface-part Visibility/Uncertainty head on existing Blender static poses.

GPU smoke example:
python -m mocap.train_four_view runs/dino/dataset_fourview_p3_smoke_v1 \
  assets/dino/rigged/rig_info.json --init-reader runs/dino/reader_hr_v1.pt \
  --out runs/dino/reader_fourview_p3_smoke_v1.pt --epochs 2 --batch 2

Baseline held-out evaluation:
python -m mocap.train_four_view runs/dino/dataset_fourview_p3_smoke_v1 \
  assets/dino/rigged/rig_info.json --eval-checkpoint runs/dino/reader_hr_v1.pt \
  --report runs/dino/fourview_p3_baseline.json

This trains a separate *experimental* checkpoint. It never overwrites the
original Reader, and does not generate H3 video. The 4 validation poses are
independent by pose id, not a random split of their four views.
"""
import argparse
import json
import math
import os
import pathlib
import time
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mocap.fourview_data import (
    FourViewDataset,VIEWS,expected_visibility_and_error_summary,split_poses,sha256)
from mocap.model import JointReader,decode,quality_at
from mocap.reader_config import ReaderSpec,spec_from_checkpoint
from mocap.reader_warmstart import validate_base_reader
from mocap.train import heatmaps
from mocap.training_checkpoint import atomic_torch_save

PROFILE="anymove.four_view_reader_mesh_visibility_p3/1"
QUALITY_TAG="mesh_surface_part_visibility_v1_uncalibrated"


def _new_model(names,spec,*,quality):
    model=JointReader(len(names),input_size=spec.input_size,
                      heatmap_size=spec.heatmap_size,quality_head=quality).cuda()
    model.backbone.set_attn_implementation("sdpa")
    return model


def prepare_from_checkpoint(checkpoint_path,names,*,train_new_quality):
    """Never silently alter the original 2-view Reader's weights."""
    source=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    spec=spec_from_checkpoint(source)
    if list(source.get("names",[]))!=list(names):
        raise ValueError("Reader checkpoint bone/keypoint order mismatch")
    old_quality=source.get("quality_spec")
    if train_new_quality:
        state=validate_base_reader(source,names,spec)
        model=_new_model(names,spec,quality=True)
        load=model.load_state_dict(state,strict=False)
        if (set(load.missing_keys)!={"quality_out.weight","quality_out.bias"} or
                load.unexpected_keys):
            raise RuntimeError("warm-start differs from expected new quality head")
    else:
        quality=bool(old_quality)
        if quality and old_quality.get("version") not in (1,2):
            raise ValueError("unsupported checkpoint quality semantics")
        model=_new_model(names,spec,quality=quality)
        model.load_state_dict(source["state_dict"],strict=True)
    return model,spec,source


def evaluate(model,dataset,*,batch,workers):
    """No gradient; compare same held-out Blender pose IDs for all four views."""
    model.eval()
    J=len(dataset.names)
    size=model.input_size
    hm=model.heatmap_size
    val_loader=DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=workers,
                          pin_memory=True)
    all_err=defaultdict(list)
    visible_label=defaultdict(list)
    vis_pred=defaultdict(list)
    expected=defaultdict(list)
    actual=defaultdict(list)
    with torch.inference_mode():
        for sample in val_loader:
            x=sample["image"].cuda(non_blocking=True)
            with torch.autocast("cuda",dtype=torch.float16):
                if model.quality_head:
                    hm_pred,qual=model(x,return_quality=True)
                else:
                    hm_pred=model(x)
            uv_hm,_=decode(hm_pred.float())
            uv_model=uv_hm * (size/hm)
            g=sample["geo"].float()
            uv_view=(uv_model.cpu() * g[:,None,2:3]/size - g[:,None,:2])
            real=sample["source_uv"]
            scale=(g[:,3:4] * 0+1.) # keep dimensions, scale derives from meters_per_px below
            # Known layout M_PER_PX derived from rig source frame, stored in
            # dataset's Rig camera layout. Image scale = GT layout MPP / image MPP.
            rig_mpp=float(dataset.rig["layout"]["M_PER_PX"])
            image_scale=rig_mpp/g[:,3]
            error=torch.linalg.vector_norm(uv_view-real,dim=-1)/image_scale[:,None]
            mask=sample["inside"] # excludes unobserved/out-of-image labels
            if model.quality_head:
                logits,log_error=quality_at(qual.float(),uv_hm)
                vscore=torch.sigmoid(logits).cpu()
                # map predicted heatmap-space localization error to original
                # unscaled camera layout pixels (same units as 2D Reader audits)
                error_px=torch.expm1(log_error.float().clamp(0,6)).cpu()
                expected_layout=error_px*(g[:,None,2]/hm)/image_scale[:,None]
            for b in range(len(x)):
                vi=int(sample["view_index"][b])
                tag=VIEWS[vi]
                valid=mask[b].numpy().astype(bool)
                if valid.any():
                    all_err[tag].extend(error[b][valid].tolist())
                if model.quality_head:
                    label=sample["visibility"][b].numpy()
                    keep=valid&(label>=0)
                    if keep.any():
                        visible_label[tag].extend(label[keep].tolist())
                        vis_pred[tag].extend(vscore[b][keep].tolist())
                    if valid.any():
                        actual[tag].extend(error[b][valid].tolist())
                        expected[tag].extend(expected_layout[b][valid].tolist())
    out={}
    for view in VIEWS:
        e=np.asarray(all_err[view],dtype=float)
        if len(e)<1:raise RuntimeError("missing heldout validation for "+view)
        stats={
            "num_keypoints_in_frame":len(e),
            "mean_layout_px":round(float(e.mean()),5),
            "median_layout_px":round(float(np.median(e)),5),
            "p95_layout_px":round(float(np.percentile(e,95)),5),
        }
        if model.quality_head:
            stats["quality"]=expected_visibility_and_error_summary(
                vis_pred[view],visible_label[view],
                expected[view],actual[view])
        out[view]=stats
    out["all_views_mean_layout_px"]=round(float(np.mean(
        [e for v in VIEWS for e in all_err[v]])),5)
    out["mean_back_opposite_layout_px"]=round(
        (out["back"]["mean_layout_px"]+out["opposite_side"]["mean_layout_px"])/2,5)
    out["mean_front_side_layout_px"]=round(
        (out["front"]["mean_layout_px"]+out["side"]["mean_layout_px"])/2,5)
    return out


def loss_terms(model,sample,spec,*,quality_strength):
    x=sample["image"].cuda(non_blocking=True)
    kp=sample["keypoints"].cuda(non_blocking=True)
    known=sample["visibility"].cuda(non_blocking=True)
    inside=sample["inside"].cuda(non_blocking=True)
    with torch.autocast("cuda",dtype=torch.float16):
        heat,qual=model(x,return_quality=True)
    heat=heat.float()
    # Same heatmap target/loss as the pretrained high-res Reader.
    target=heatmaps(kp,spec.input_size,spec.heatmap_size)
    map_loss=F.mse_loss(heat,target)*1000.
    # Quality supervision is sampled at the model's PREDICTED 2D location,
    # and is separate from the source Blender GT keypoint itself.
    with torch.no_grad():
        uv_hm,_=decode(heat.detach())
    vis_logits,log_error=quality_at(qual.float(),uv_hm)
    visible=(known==1).float()
    vis_mask=inside&(known>=0)
    bce=F.binary_cross_entropy_with_logits(vis_logits,visible,reduction="none")
    bce=bce*torch.where(visible<.5,3.,1.)
    visibility_loss=(bce*vis_mask).sum()/vis_mask.sum().clamp(min=1)
    kp_hm=kp*(spec.heatmap_size/spec.input_size)
    with torch.no_grad():
        gt_error=torch.linalg.vector_norm(uv_hm-kp_hm,dim=-1)
        log_gt=torch.log1p(gt_error.clamp(max=128))
    err_loss=(F.smooth_l1_loss(log_error.clamp(-1,6),log_gt,reduction="none")*
              inside).sum()/inside.sum().clamp(min=1)
    loss=map_loss+quality_strength*(visibility_loss+err_loss)
    return loss,(float(map_loss.detach()),float(visibility_loss.detach()),
                 float(err_loss.detach()))


def run(args):
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    if not torch.cuda.is_available():
        raise RuntimeError("P3 Reader training/evaluation requires an installed CUDA GPU")
    rig=json.loads(pathlib.Path(args.rig_info).read_text(encoding="utf8"))
    names=[kp["name"] for kp in rig["keypoints"]]
    if args.split_manifest:
        split=json.loads(pathlib.Path(args.split_manifest).read_text(encoding='utf8'))
        train_ids=tuple(int(i) for i in split['train_ids'])
        val_ids=tuple(int(i) for i in split['val_ids'])
        if not train_ids or not val_ids:
            raise ValueError('P4 custom train/validation cannot be empty')
        existing=set(int(p.name[:5]) for p in pathlib.Path(args.dataset_dir).glob('*_visibility_meta.json'))
        if not set(train_ids+val_ids)<=existing or len(set(train_ids))!=len(train_ids) or len(set(val_ids))!=len(val_ids):
            raise ValueError('P4 split IDs unavailable or duplicated')
        if set(train_ids)&set(val_ids):
            raise ValueError('P4 train/validation leakage')
        test_ids=set(int(i) for i in split.get('test_ids',[]))
        if test_ids & (set(train_ids)|set(val_ids)):
            raise ValueError('P4 blind-test IDs leaked into training/validation')
    else:
        train_ids,val_ids=split_poses(args.dataset_dir)
    if set(train_ids)&set(val_ids):
        raise AssertionError("same four-view pose leaked across train/validation")
    val_set=FourViewDataset(args.dataset_dir,args.rig_info,val_ids,
                             train=False,seed=args.seed,input_size=args.input_size)
    chosen_train_views=tuple(v.strip() for v in args.train_views.split(','))
    if len(set(chosen_train_views))!=len(chosen_train_views) or any(v not in VIEWS for v in chosen_train_views):
        raise ValueError('invalid P4 trained camera list')
    train_set=FourViewDataset(args.dataset_dir,args.rig_info,train_ids,
                             train=True,seed=args.seed,input_size=args.input_size,
                             views=chosen_train_views)
    if set(train_set.sample_keys)&set(val_set.sample_keys):
        raise AssertionError("per-view Train/Validation sample keys overlap")
    # P1 originally used Workbench SINGLE gray for geometry debugging.
    # Reject that domain as Reader training by default. Original 2-view
    # Reader used the creature's textured / limb-coloured materials.
    if (not args.allow_neutral_geometry and
            (train_set.appearance_mode!="source-material" or
             val_set.appearance_mode!="source-material")):
        raise ValueError("P3 Reader train/eval requires --appearance source-material images; "
                         "rerender in a NEW folder or explicitly use --allow-neutral-geometry "
                         "for legacy gray-domain ablation")
    training=not args.eval_checkpoint
    checkpoint_path=pathlib.Path(args.init_reader if training else args.eval_checkpoint)
    if training and not args.out:
        raise ValueError("training requires --out for separate P3 checkpoint")
    model,spec,source=prepare_from_checkpoint(checkpoint_path,names,
                                             train_new_quality=training)
    if spec.input_size!=args.input_size or spec.heatmap_size!=args.heatmap_size:
        raise ValueError("P3 four-view Reader checkpoint resolution mismatch")
    original_sha=sha256(checkpoint_path)
    info={"profile":PROFILE,"creature_rig":str(pathlib.Path(args.rig_info).resolve()),
          "dataset":str(pathlib.Path(args.dataset_dir).resolve()),
          "input_checkpoint":str(checkpoint_path.resolve()),
          "input_checkpoint_sha256":original_sha,
          "train_pose_ids":list(train_ids),"validation_pose_ids":list(val_ids),
          "train_poses":len(train_ids),"val_poses":len(val_ids),
          "views":list(VIEWS),"input_size":spec.input_size,
          "heatmap_size":spec.heatmap_size,
          "visibility_semantics":("geometry_skin_surface_part_labels_visible_occluded_unknown"
                                   " NOT skeleton joint center and NOT H3-calibrated"),
          "validation_independent_by_pose":True,
          "train_views":list(chosen_train_views),
          "checkpoint_selection_metric":args.selection_metric,
          "split_manifest":str(pathlib.Path(args.split_manifest).resolve()) if args.split_manifest else None,
          "split_manifest_sha256":sha256(args.split_manifest) if args.split_manifest else None,
          "independent_static_poses_not_motion":True,
          "no_h3_video_generation":True,
          "training_appearance":train_set.appearance_mode,
          "validation_appearance":val_set.appearance_mode,
          "legacy_gray_appearance_explicitly_allowed":bool(args.allow_neutral_geometry),
          "frozen_backbone":args.unfreeze_last_blocks==0,
          "unfreeze_last_blocks":args.unfreeze_last_blocks}
    before=evaluate(model,val_set,batch=args.batch,workers=args.workers)
    info["baseline_validation"]=before
    print("P3 BASELINE",json.dumps({
        v:before[v]["mean_layout_px"] for v in VIEWS},indent=2),flush=True)
    if not training:
        path=pathlib.Path(args.report)
        if path.exists():raise FileExistsError("P3 baseline report would overwrite existing result")
        path.parent.mkdir(parents=True,exist_ok=True)
        info["status"]="BASELINE_EVALUATED_NO_TRAINING"
        path.write_text(json.dumps(info,indent=2),encoding="utf8")
        print("P3_BASELINE_REPORT",path,flush=True)
        return info
    # Preserve frozen DINOv2 pretrained features; update only the 4-view
    # heatmap decoder and newly-added mesh-visibility/error head.
    model.backbone.requires_grad_(False)
    if args.unfreeze_last_blocks:
        blocks=model.backbone.encoder.layer
        if args.unfreeze_last_blocks>len(blocks):
            raise ValueError('requested last Backbone blocks exceed DINOv2 depth')
        for block in blocks[-args.unfreeze_last_blocks:]:block.requires_grad_(True)
    train_backbone=[p for p in model.backbone.parameters() if p.requires_grad]
    decoder=list(model.neck.parameters())+list(model.up1.parameters())+\
            list(model.up2.parameters())+list(model.out.parameters())
    quality=list(model.quality_out.parameters())
    optimizer=torch.optim.AdamW([
        {"params":decoder,"lr":args.lr},
        {"params":quality,"lr":args.lr*2}]+(
        [{"params":train_backbone,"lr":args.lr*.1}] if train_backbone else []),weight_decay=.01)
    amp_dtype=torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    best=float("inf")
    best_val=None
    best_epoch=0
    history=[]
    out=pathlib.Path(args.out)
    report=out.with_suffix(".validation.json")
    if out.exists() or report.exists():
        raise FileExistsError("Refusing to overwrite trained P3 Reader checkpoint")
    out.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    for epoch in range(args.epochs):
        train_set.epoch=epoch
        loader=DataLoader(train_set,batch_size=args.batch,shuffle=True,
                          generator=torch.Generator().manual_seed(args.seed+epoch),
                          num_workers=args.workers,pin_memory=True)
        model.eval()
        if args.unfreeze_last_blocks:
            for block in model.backbone.encoder.layer[-args.unfreeze_last_blocks:]:block.train()
        model.neck.train();model.up1.train();model.up2.train()
        model.out.train();model.quality_out.train()
        losses=[];qvis=[];qerr=[];step=0
        torch.cuda.reset_peak_memory_stats()
        for sample in loader:
            loss,components=loss_terms(model,sample,spec,
                                       quality_strength=args.quality_strength)
            if not torch.isfinite(loss):
                raise FloatingPointError("P3 four-view Reader training diverged")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder+quality+train_backbone,max_norm=1.)
            scaler.step(optimizer);scaler.update()
            step+=1
            losses.append(components[0]);qvis.append(components[1]);qerr.append(components[2])
            if step==1 or step%20==0 or step==len(loader):
                print("P3 TRAIN", "ep",epoch+1,"step",step,"/",len(loader),
                      "hm",round(components[0],5),
                      "vis",round(components[1],5),
                      "error",round(components[2],5),
                      "seconds",round(time.perf_counter()-start,1),flush=True)
        val=evaluate(model,val_set,batch=args.batch,workers=args.workers)
        candidate=float(val['mean_back_opposite_layout_px'] if args.selection_metric=='back_opposite' else val['all_views_mean_layout_px'])
        h={"epoch":epoch+1,"train_heatmap_loss":round(float(np.mean(losses)),5),
           "train_visibility_bce":round(float(np.mean(qvis)),5),
           "train_expected_error_huber":round(float(np.mean(qerr)),5),
           "peak_gpu_allocated_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
           "validation":val}
        history.append(h)
        print("P3 VAL EPOCH",epoch+1,
              json.dumps({v:val[v]["mean_layout_px"] for v in VIEWS}),
              "total",candidate,"peak_vram_gib",h["peak_gpu_allocated_gib"],
              flush=True)
        if candidate<best:
            best=candidate
            best_epoch=epoch+1
            best_val=val
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            checkpoint={
                "state_dict":state,"names":names,
                "epoch":epoch+1,
                "val":{v:val[v]["mean_layout_px"] for v in VIEWS},
                "reader_spec":dict(spec.checkpoint_metadata(),quality_head=True),
                "quality_spec":{"version":2,
                    "target":"mesh_surface_part_visibility",
                    "visibility_semantics":QUALITY_TAG,
                    "error_unit":"heatmap_pixels",
                    "not_validated_real_h3":True},
                "four_view_spec":{"schema_version":PROFILE,
                    "view_ids":list(VIEWS),
                    "train_poses":len(train_ids),"val_poses":len(val_ids),
                    "frozen_backbone":args.unfreeze_last_blocks==0,
                    "unfreeze_last_blocks":args.unfreeze_last_blocks,
                    "trained_views":list(chosen_train_views),
                    "selection_metric":args.selection_metric,
                    "source_checkpoint_sha256":original_sha},
            }
            # Only the experiment output, never original Reader. A second
            # epoch may replace the previously saved BEST experimental model.
            temp=out.with_name(out.name+".tmp")
            torch.save(checkpoint,temp)
            temp.replace(out)
            print("P3_SAVED_BEST",out,"epoch",epoch+1,flush=True)
    info.update({
        "mode":"train_four_view_decoder_and_quality_head",
        "status":"SMOKE_TRAINED_NOT_PRODUCTION",
        "iterations_epochs":args.epochs,"batch":args.batch,
        "lr":args.lr,"quality_strength":args.quality_strength,
        "best_validation_mean_layout_px":best_val['all_views_mean_layout_px'],
        "best_checkpoint_selection_value_px":best,
        "best_checkpoint_epoch":best_epoch,
        "history":history,
        "trained_checkpoint":str(out.resolve()),
        "trained_checkpoint_sha256":sha256(out),
        "elapsed_seconds":round(time.perf_counter()-start,3)})
    report.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding="utf8")
    print("P3_REPORT",report,flush=True)
    return info


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("rig_info")
    ap.add_argument("--init-reader",default="")
    ap.add_argument("--eval-checkpoint",default="")
    ap.add_argument("--out",default="")
    ap.add_argument("--report",default="")
    ap.add_argument("--epochs",type=int,default=2)
    ap.add_argument("--batch",type=int,default=2)
    ap.add_argument("--workers",type=int,default=0)
    ap.add_argument('--train-views',default='front,side,back,opposite_side')
    ap.add_argument('--unfreeze-last-blocks',type=int,default=0,
                    help='P4 experimental partial DINOv2 finetuning; original Reader remains untouched')
    ap.add_argument('--selection-metric',choices=('all','back_opposite'),default='all')
    ap.add_argument('--split-manifest',default='')
    ap.add_argument("--allow-neutral-geometry",action="store_true",
                    help="Reproduce OLD gray Workbench P3 smoke only; do not promote as normal colorized Reader training")
    ap.add_argument("--input-size",type=int,default=672)
    ap.add_argument("--heatmap-size",type=int,default=192)
    ap.add_argument("--lr",type=float,default=0.0002)
    ap.add_argument("--quality-strength",type=float,default=.2)
    ap.add_argument("--seed",type=int,default=20261101)
    args=ap.parse_args()
    if ((bool(args.init_reader)==bool(args.eval_checkpoint)) or
            args.epochs<1 or args.batch<1 or args.workers<0 or
            args.unfreeze_last_blocks<0 or
            args.lr<=0 or args.quality_strength<0):
        raise ValueError("select exactly one of --init-reader / --eval-checkpoint and valid hyperparameters")
    if args.eval_checkpoint and not args.report:
        raise ValueError("evaluation requires --report")
    run(args)


if __name__=="__main__":
    main()
