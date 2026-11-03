"""Fresh untouched Human V3 fullbody Reader/Foot-Chain Guard evaluation.

Runs original, frozen 67-keypoint Reader and immutable TRAIN-derived guard.
No training, checkpoint selection or threshold updates; whole new clip seeds
were generated only after the guard was frozen using Validation.
"""
import hashlib,json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from mocap.model import JointReader,SIZE,HM,decode
from mocap.camera_geometry import triangulate_orthographic
from mocap.toe_tip_guard import (
    ToeBaseGuardConfig,ToeTipGuardConfig,guard_foot_chain
)
from tools.train_human_v3_fullbody_finger_reader_ab import (
    ROOT,PAIR_NAMES,SOURCE_SHA,FullBodyImages
)

FRESH=ROOT/"fresh_test_20261103"
TRAIN=ROOT/"dataset"

class FreshImages(FullBodyImages):
    def __init__(self,manifest):
        self.base=FRESH
        self.rows=[r for r in manifest["samples"] if r["split"]=="test"]
        self.train=False
        self.epoch=0
        self.names=manifest["joint_names"]
        if len(self.names)!=67 or len(self.rows)!=56:
            raise RuntimeError("Unexpected fresh test data")
    # inherits same image loading and 67 keypoint GT transformation.

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()

def categorise(names):
    regions={"fingers":[],"legs_feet_toes":[],"arms_hands":[],"torso_root":[],"head_neck":[]}
    for i,n in enumerate(names):
        if any(t in n for t in ("Pinky","Ring","Mid","Index","Thumb")):
            key="fingers"
        elif any(t in n for t in ("Thigh","Calf","Foot","ToeBase","ToeTip")):
            key="legs_feet_toes"
        elif any(t in n for t in ("Clavicle","Upperarm","Forearm","_Hand")):
            key="arms_hands"
        elif any(t in n for t in ("Head","NeckTwist")):
            key="head_neck"
        else:
            key="torso_root"
        regions[key].append(i)
    if list(map(len,regions.values()))!=[40,10,8,6,3]:
        raise RuntimeError("Anatomical region partition mismatch")
    return regions

def metrics(values):
    x=np.asarray(values,dtype=np.float64)
    return {"n":int(x.size),"mean":round(float(x.mean()),4),
            "p95":round(float(np.percentile(x,95)),4),
            "over_10":int(np.sum(x>10))}

@torch.inference_mode()
def evaluate(pair):
    manifest=json.loads((FRESH/pair/"manifest.json").read_text(encoding="utf-8"))
    original=json.loads((TRAIN/pair/"manifest.json").read_text(encoding="utf-8"))
    ckpath=ROOT/"reader_models_strict"/pair/"best.pt"
    contract=json.loads((ROOT/"reader_models_strict"/pair/"foot_chain_guard_train_val.json").read_text(encoding="utf-8"))
    if (manifest["source_blend_sha256"]!=SOURCE_SHA or
        manifest["angles_deg"]!=original["angles_deg"] or
        manifest["joint_names"]!=original["joint_names"] or
        contract["checkpoint_sha256"]!=digest(ckpath) or
        contract["angles_deg"]!=manifest["angles_deg"]):
        raise RuntimeError("Fresh evaluation provenance/angles mismatch")
    ck=torch.load(ckpath,map_location="cpu",weights_only=True)
    seen=set(ck["train_clips"]+ck["validation_clips"]+ck["test_clips"])
    if seen.intersection(manifest["test_clips"]):
        raise RuntimeError("Fresh test clip was previously used in Reader training or validation")
    base={k:ToeBaseGuardConfig.from_dict(v) for k,v in contract["base_calibration"].items()}
    tip={k:ToeTipGuardConfig.from_dict(v) for k,v in contract["tip_calibration"].items()}
    data=FreshImages(manifest)
    dl=DataLoader(data,batch_size=4,shuffle=False,num_workers=0,pin_memory=True)
    model=JointReader(len(manifest["joint_names"])).cuda().eval()
    model.load_state_dict(ck["state_dict"])
    names=manifest["joint_names"]
    regions=categorise(names)
    region_keys=dict(regions)
    region_keys["all_67"]=list(range(len(names)))
    region_keys["toe_2_tips"]=[names.index("CC_Base_"+s+"_ToeTip") for s in ("L","R")]
    region_keys["toe_2_bases"]=[names.index("CC_Base_"+s+"_ToeBase") for s in ("L","R")]
    region_keys["finger_10_tips"]=[names.index("CC_Base_"+s+"_"+digit+"Tip")
                                    for s in ("L","R") for digit in ("Pinky","Ring","Mid","Index","Thumb")]
    errs2=defaultdict(list)
    flags=defaultdict(int)
    output=defaultdict(lambda:{"raw":{},"guard":{}})
    for images,kp,indices in dl:
        with torch.autocast("cuda",dtype=torch.float16):
            heat=model(images.cuda(non_blocking=True))
        uv,_=decode(heat.float())
        uv=uv.cpu().numpy()*(SIZE/HM)
        gt=kp.numpy()
        for i,j in enumerate(indices.tolist()):
            row=data.rows[j]
            corrected,flag=guard_foot_chain(uv[i],names,row["slot"],base,tip,manifest["angles_deg"])
            flags["base_repaired"]+=flag["base_repaired"]
            flags["base_untrusted"]+=flag["base_untrusted"]
            flags["tip_repaired"]+=flag["tip_repaired"]
            flags["tip_untrusted"]+=flag["tip_untrusted"]
            for label,points in (("raw",uv[i]),("guard",corrected)):
                err=np.linalg.norm(points-gt[i],axis=-1)
                for region,idx in region_keys.items():
                    errs2[label+"_"+region].extend(err[idx])
            key=(row["clip"],int(row["frame"]))
            output[key]["raw"][row["slot"]]=uv[i]
            output[key]["guard"][row["slot"]]=corrected
    gt_cache={clip:json.loads((FRESH/"motion_gt"/(clip+".json")).read_text(encoding="utf-8"))
              for clip in manifest["test_clips"]}
    errs3=defaultdict(list)
    perclip=defaultdict(lambda:defaultdict(list))
    for (clip,frame),views in output.items():
        gt=np.asarray(gt_cache[clip]["joint_position_m"][frame],dtype=np.float64)
        for method in ("raw","guard"):
            v=views[method]
            if set(v)!={"front","side"}:
                raise RuntimeError("Missing one camera in paired frame")
            triangulated=triangulate_orthographic(
                v["front"],v["side"],np.ones(67),np.ones(67),
                front_width=448,side_width=448,height=448,
                metres_per_pixel=manifest["metres_per_pixel"],
                z_center=manifest["z_center"],front_angle_deg=manifest["angles_deg"][0],
                side_angle_deg=manifest["angles_deg"][1],front_height_weight=1.0)
            err=np.linalg.norm(triangulated-gt,axis=-1)*100
            for region,ids in region_keys.items():
                errs3[method+"_"+region].extend(err[ids])
                perclip[clip][method+"_"+region].extend(err[ids])
    report={
        "schema":"anymove.human_fullbody_fresh_guard_comparison_v1",
        "status":"FRESH_TEST_COMPLETE_NO_RETRAINING",
        "pair":pair,"angles_deg":manifest["angles_deg"],
        "source_blend_sha256":SOURCE_SHA,
        "fixed_checkpoint_sha256":digest(ckpath),
        "calibration_source":"TRAIN GT only; QA done on separate Validation clips before Fresh Test generation",
        "new_clips":manifest["test_clips"],
        "fresh_test_images":len(data),"fresh_test_frames":len(output),
        "markers":67,
        "repair_flags":dict(flags),
        "2d_px":{key:metrics(x) for key,x in errs2.items()},
        "3d_cm":{key:metrics(x) for key,x in errs3.items()},
        "perclip_3d_cm":{clip:{key:metrics(vals) for key,vals in methods.items()}
                         for clip,methods in perclip.items()},
        "limitations":[
            "Procedural Blender GT, not natural human MoCap",
            "Only four new seed clips; larger domain tests are required",
            "No temporal filtering, no full FK Solver, no H3 video evaluation",
        ],
    }
    for method in ("raw","guard"):
        score=np.mean([report["3d_cm"][method+"_"+k]["mean"] for k in regions])
        report["3d_cm"][method+"_5region_equal_score_cm"]=float(score)
    dest=FRESH/pair/"reader_guard_evaluation.json"
    dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("FRESH_AB_PAIR",pair,json.dumps({
        "flags":dict(flags),
        "3d_full_raw_cm":report["3d_cm"]["raw_all_67"]["mean"],
        "3d_full_guard_cm":report["3d_cm"]["guard_all_67"]["mean"],
        "3d_toe_tip_raw_cm":report["3d_cm"]["raw_toe_2_tips"]["mean"],
        "3d_toe_tip_guard_cm":report["3d_cm"]["guard_toe_2_tips"]["mean"],
        "5region_guard_cm":report["3d_cm"]["guard_5region_equal_score_cm"],
    }),flush=True)
    del model
    torch.cuda.empty_cache()
    return report

def main():
    results={p:evaluate(p) for p in PAIR_NAMES}
    summary={
        "schema":"anymove.human_fullbody_fresh_guard_ab/1",
        "status":"NEW_SEEDS_HELDOUT_AND_GUARD_FROZEN",
        "test_samples_each":results[PAIR_NAMES[0]]["fresh_test_images"],
        "new_test_clips":results[PAIR_NAMES[0]]["new_clips"],
        "metrics":{}}
    for pair,report in results.items():
        summary["metrics"][pair]={
            "raw_3d_full_cm":report["3d_cm"]["raw_all_67"]["mean"],
            "guard_3d_full_cm":report["3d_cm"]["guard_all_67"]["mean"],
            "raw_3d_toetip_cm":report["3d_cm"]["raw_toe_2_tips"]["mean"],
            "guard_3d_toetip_cm":report["3d_cm"]["guard_toe_2_tips"]["mean"],
            "raw_equal_5region_cm":report["3d_cm"]["raw_5region_equal_score_cm"],
            "guard_equal_5region_cm":report["3d_cm"]["guard_5region_equal_score_cm"],
            "base_repaired":report["repair_flags"].get("base_repaired",0),
            "tip_repaired":report["repair_flags"].get("tip_repaired",0),
            "untrusted_count":report["repair_flags"].get("base_untrusted",0)+report["repair_flags"].get("tip_untrusted",0),
            "finger_tip_3d_cm":report["3d_cm"]["guard_finger_10_tips"]["mean"],
        }
    (FRESH/"fresh_ab_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("FRESH_AB_SUMMARY",json.dumps(summary),flush=True)

if __name__=="__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Fresh test requires local CUDA for the pretrained Reader")
    torch.set_num_threads(4)
    main()
