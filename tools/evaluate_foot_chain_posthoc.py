"""Post-hoc guarded Reader comparison on the ALREADY-INSPECTED Test.

This is a diagnostic sanity check, NOT a fresh unbiased acceptance test.
Thresholds/matrices are frozen in training+validation calibration JSON.
"""
import hashlib,json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from mocap.model import JointReader, SIZE, HM, decode
from mocap.camera_geometry import triangulate_orthographic
from mocap.toe_tip_guard import ToeBaseGuardConfig, ToeTipGuardConfig, guard_foot_chain
from tools.train_human_v3_fullbody_finger_reader_ab import ROOT,PAIR_NAMES,manifest,FullBodyImages

def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def stats(a):
    return {"mean":float(np.mean(a)),"p95":float(np.percentile(a,95)),
            "over_10px_fraction":float(np.mean(np.asarray(a)>10))}

@torch.inference_mode()
def run_one(pair):
    config=manifest(pair)
    folder=ROOT/"reader_models_strict"/pair
    model_path=folder/"best.pt"
    contract=json.loads((folder/"foot_chain_guard_train_val.json").read_text(encoding="utf-8"))
    if (contract["checkpoint_sha256"]!=file_hash(model_path) or
        contract["angles_deg"]!=config["angles_deg"]):
        raise RuntimeError("Guard calibration not frozen to this checkpoint/angle pair")
    base={k:ToeBaseGuardConfig.from_dict(v) for k,v in contract["base_calibration"].items()}
    tip={k:ToeTipGuardConfig.from_dict(v) for k,v in contract["tip_calibration"].items()}
    ck=torch.load(model_path,map_location="cpu",weights_only=True)
    model=JointReader(len(config["joint_names"])).cuda().eval()
    model.load_state_dict(ck["state_dict"])
    data=FullBodyImages(pair,config,"test",False)
    dl=DataLoader(data,batch_size=4,shuffle=False,num_workers=0)
    names=config["joint_names"]
    all_uv=defaultdict(lambda:{"raw":{},"guard":{}})
    e2=defaultdict(list)
    flags=defaultdict(int)
    for x,kp,ids in dl:
        with torch.autocast("cuda",dtype=torch.float16):
            heat=model(x.cuda())
        predicted,_=decode(heat.float())
        uv=predicted.cpu().numpy()*(SIZE/HM)
        true=kp.numpy()
        for i,j in enumerate(ids.tolist()):
            row=data.rows[j]
            new,flag=guard_foot_chain(uv[i],names,row["slot"],base,tip,config["angles_deg"])
            key=(row["clip"],int(row["frame"]))
            all_uv[key]["raw"][row["slot"]]=uv[i]
            all_uv[key]["guard"][row["slot"]]=new
            flags["base_repaired"]+=flag["base_repaired"]
            flags["tip_repaired"]+=flag["tip_repaired"]
            flags["base_untrusted"]+=flag["base_untrusted"]
            flags["tip_untrusted"]+=flag["tip_untrusted"]
            for side in ("L","R"):
                for kind in ("ToeBase","ToeTip"):
                    idx=names.index("CC_Base_"+side+"_"+kind)
                    label=kind+"_"+side
                    e2[label+"_raw"].append(float(np.linalg.norm(uv[i,idx]-true[i,idx])))
                    e2[label+"_guard"].append(float(np.linalg.norm(new[idx]-true[i,idx])))
    idx_toe=[names.index("CC_Base_"+s+"_ToeTip") for s in ("L","R")]
    idx_base=[names.index("CC_Base_"+s+"_ToeBase") for s in ("L","R")]
    idx_leg=[i for i,n in enumerate(names) if any(t in n for t in ("Thigh","Calf","Foot","ToeBase","ToeTip"))]
    err3=defaultdict(list)
    clip_cache={}
    for (clip,frame),views in all_uv.items():
        if clip not in clip_cache:
            clip_cache[clip]=json.loads((ROOT/"dataset"/"motion_gt"/(clip+".json")).read_text(encoding="utf-8"))
        world=np.asarray(clip_cache[clip]["joint_position_m"][frame],dtype=np.float64)
        for choice in ("raw","guard"):
            v=views[choice]
            result=triangulate_orthographic(
                v["front"],v["side"],np.ones(67),np.ones(67),
                front_width=448,side_width=448,height=448,
                metres_per_pixel=config["metres_per_pixel"],z_center=config["z_center"],
                front_angle_deg=config["angles_deg"][0],side_angle_deg=config["angles_deg"][1],
                front_height_weight=1.0)
            err=np.linalg.norm(result-world,axis=-1)*100
            err3[choice+"_full"].extend(err)
            err3[choice+"_toe_tip"].extend(err[idx_toe])
            err3[choice+"_toe_base"].extend(err[idx_base])
            err3[choice+"_leg"].extend(err[idx_leg])
    report={
        "schema":"anymove.fullbody_foot_chain_posthoc_old_test/1",
        "status":"POSTHOC_DIAGNOSTIC_NOT_INDEPENDENT_ACCEPTANCE",
        "pair":pair,"angles_deg":config["angles_deg"],
        "checkpoint_sha256":file_hash(model_path),
        "source_calibration":"TRAIN-only, quality checked on VALIDATION",
        "test_clips_previously_inspected":config["test_clips"],
        "views":len(data),"foot_flags":dict(flags),
        "2d_toe_px":{k:stats(v) for k,v in e2.items()},
        "3d_cm":{k:{"mean":float(np.mean(v)),"p95":float(np.percentile(v,95))}
                 for k,v in err3.items()},
        "caution":"This TEST already exposed the failures; use a fresh never-viewed test for unbiased acceptance."
    }
    path=folder/"foot_chain_posthoc_old_test.json"
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("POSTHOC_FOOT_RESULT",pair,json.dumps({
        "flags":report["foot_flags"],
        "raw_3d_toe_cm":report["3d_cm"]["raw_toe_tip"]["mean"],
        "guard_3d_toe_cm":report["3d_cm"]["guard_toe_tip"]["mean"],
        "raw_full_3d_cm":report["3d_cm"]["raw_full"]["mean"],
        "guard_full_3d_cm":report["3d_cm"]["guard_full"]["mean"],
        "raw_toebase_3d_cm":report["3d_cm"]["raw_toe_base"]["mean"],
        "guard_toebase_3d_cm":report["3d_cm"]["guard_toe_base"]["mean"]
    }),flush=True)
    del model
    torch.cuda.empty_cache()

if __name__=="__main__":
    torch.set_num_threads(4)
    for p in PAIR_NAMES:run_one(p)
