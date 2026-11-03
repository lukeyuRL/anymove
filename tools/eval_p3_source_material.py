"""Independent acceptance gate for properly textured full-resolution P3 rerun.

Never a training script, never reads GT in Solver. Existing P2 N-camera FK
outputs and independent GT evaluations must come from SAME original four
heldout pose IDs. Old gray 0.5-scale results are explicitly NOT substituted.
"""
import argparse,hashlib,json
from pathlib import Path

VIEWS=("front","side","back","opposite_side")
def sha(path):
    dig=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):dig.update(chunk)
    return dig.hexdigest()


def require(cond,msg):
    if not cond:raise ValueError("P3 material provenance failed: "+msg)


def eval_material_run(creature,repo):
    root=Path(repo).resolve()
    dataset=root/"runs"/creature/"dataset_fourview_p3_material_fullres_v1"
    output=root/"runs"/creature
    train=json.loads((output/"reader_fourview_p3_material_fullres_v1.validation.json").read_text())
    base2d=json.loads((output/"fourview_p3_material_baseline_v1.json").read_text())
    tuned2d=json.loads((output/"fourview_p3_material_posttrain_eval_v1.json").read_text())
    base3d=json.loads((dataset/"p3_material_baseline_fk_3d_eval_v1.json").read_text())
    tuned3d=json.loads((dataset/"p3_material_trained_fk_3d_eval_v1.json").read_text())
    manifestbase=json.loads((dataset/"p3_material_baseline_heldout.manifest.json").read_text())
    manifesttuned=json.loads((dataset/"p3_material_trained_heldout.manifest.json").read_text())
    require(train["training_appearance"]=="source-material" and train["validation_appearance"]=="source-material","not source material")
    require(not train["legacy_gray_appearance_explicitly_allowed"],"gray override enabled")
    require(Path(train["dataset"]).resolve()==dataset,"unexpected training dataset")
    require(train["train_pose_ids"]==list(range(1,25)),"different train pose IDs")
    require(train["validation_pose_ids"]==[25,50,75,100],"different heldout pose IDs")
    require(set(train["train_pose_ids"]).isdisjoint(train["validation_pose_ids"]),"pose leaked into validation")
    require(sha(train["input_checkpoint"])==train["input_checkpoint_sha256"],"original Reader hash changed")
    require(sha(train["trained_checkpoint"])==train["trained_checkpoint_sha256"],"trained checkpoint hash changed")
    require(train["trained_checkpoint"]!=train["input_checkpoint"],"overwrote original Reader")
    require(base2d["training_appearance"]=="source-material" and tuned2d["training_appearance"]=="source-material","2D appearance mismatch")
    require(manifestbase["frame_ids"]==manifesttuned["frame_ids"]==[25,50,75,100],"inference input different")
    require(manifestbase["mode"]==manifesttuned["mode"]=="reader","not Reader inference")
    require(manifestbase["rig_sha256"]==manifesttuned["rig_sha256"],"rig changed")
    require(manifestbase["camera_gt_reprojection_max_error_px"]<.05 and manifesttuned["camera_gt_reprojection_max_error_px"]<.05,"camera GT projection wrong")
    for m in (base3d,tuned3d):
        require(m["frame_ids"]==[25,50,75,100] and m["independent_stills"],"3D GT changed")
    mbase=base2d["baseline_validation"]
    mtune=tuned2d["baseline_validation"]
    quality={}
    vstats={}
    for v in VIEWS:
        q=mtune[v].get("quality",{})
        ecal=q.get("localization_error_calibration") or {}
        quality[v]={
            "auc":q.get("auc"),"brier":q.get("brier"),"ece":q.get("ece_8bins"),
            "known":q.get("known_surface_part_samples"),
            "occluded":q.get("occluded_surface_part_samples"),
            "expected_error_calibration_gap_px":ecal.get("mean_error_calibration_gap_px")
        }
        vstats[v]={"original_2d_layout_px":mbase[v]["mean_layout_px"],
                   "fine_tuned_2d_layout_px":mtune[v]["mean_layout_px"],
                   "delta_layout_px":round(mtune[v]["mean_layout_px"]-mbase[v]["mean_layout_px"],4)}
    baseline_3d=base3d["paired_3d_comparison"]
    tuned_3d=tuned3d["paired_3d_comparison"]
    gates={
        "2d_mean_better_all_views":mtune["all_views_mean_layout_px"]<mbase["all_views_mean_layout_px"],
        "no_front_side_2d_regression":all(vstats[v]["delta_layout_px"]<=0 for v in ("front","side")),
        "2view_3d_no_regression":tuned_3d["two_view_mean_3d_mm"]<=baseline_3d["two_view_mean_3d_mm"],
        "4view_3d_beats_original_2view_3d":tuned_3d["four_view_mean_3d_mm"]<baseline_3d["two_view_mean_3d_mm"],
        "4view_3d_beats_trained_2view_3d":tuned_3d["four_view_mean_3d_mm"]<tuned_3d["two_view_mean_3d_mm"],
        "mesh_visibility_auc_ece":all((quality[v]["auc"] or 0)>=.8 and
                    (quality[v]["ece"] is not None and quality[v]["ece"]<=.1) for v in VIEWS),
        "uncertainty_gap_lt_5px":all(quality[v]["expected_error_calibration_gap_px"] is not None and
                       quality[v]["expected_error_calibration_gap_px"]<=5 for v in VIEWS)
    }
    return {
        "schema":"anymove.p3_source_material_fullres_acceptance/1",
        "creature":creature,
        "dataset":str(dataset),
        "train_pose_count":len(train["train_pose_ids"]),
        "validation_pose_count":len(train["validation_pose_ids"]),
        "appearance":"source-material",
        "resolution_mode":"original_blender_layout_scale_1",
        "trained_checkpoint":train["trained_checkpoint"],
        "original_checkpoint":train["input_checkpoint"],
        "checkpoint_hash_integrity":"PASS",
        "source_pose_gt_stored_separately":True,
        "old_gray_experiment_not_valid_as_only_comparison":True,
        "per_view_2d":vstats,
        "all_views_mean_layout_px":{"baseline":mbase["all_views_mean_layout_px"],
             "trained":mtune["all_views_mean_layout_px"]},
        "per_view_mesh_surface_visibility":quality,
        "independent_blender_gt_3d_mm":{
             "baseline_2view":baseline_3d["two_view_mean_3d_mm"],
             "baseline_4view":baseline_3d["four_view_mean_3d_mm"],
             "trained_2view":tuned_3d["two_view_mean_3d_mm"],
             "trained_4view":tuned_3d["four_view_mean_3d_mm"],
        },
        "gates":gates,
        "production_promotion_allowed":all(gates.values()),
        "status":"PASS_ALL_GATES" if all(gates.values()) else "EXPERIMENT_VERIFIED_NOT_PRODUCTION",
        "limits":"Only 4 Blender independent static heldout poses, not synchronized Creature motion or H3 video."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--creature",choices=("dino","dragon"),required=True)
    p.add_argument("--repo",default=".")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError("preserve existing P3 acceptance report")
    result=eval_material_run(a.creature,a.repo)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
