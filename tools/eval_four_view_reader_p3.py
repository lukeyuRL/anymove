"""P3 four-view Reader/Quality gate: independent, no Solver/model training.

Consumes P3 training, independent 2D validation, and 3D FK comparisons.
Never promotes a checkpoint if extra cameras or original two-view quality
regresses. Limits/thresholds are engineering screening only (no H3 claims).
"""
import argparse,json,hashlib
from pathlib import Path


VIEWS=("front","side","back","opposite_side")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            h.update(block)
    return h.hexdigest()


def summarize(creature,repo):
    if creature not in ("dino","dragon"):
        raise ValueError("P3 smoke currently has Dino and Dragon fixtures")
    base=repo/"runs"/creature
    data=base/"dataset_fourview_p3_smoke_v1"
    training=json.loads((base/"reader_fourview_p3_smoke_v1.validation.json").read_text())
    orig2d=json.loads((base/"fourview_p3_baseline_smoke_v1.json").read_text())["baseline_validation"]
    newer=json.loads((base/"fourview_p3_posttrain_eval_v1.json").read_text())["baseline_validation"]
    orig3d=json.loads((data/"p3_baseline_fk_3d_eval.json").read_text())
    new3d=json.loads((data/"p3_trained_fk_3d_eval.json").read_text())
    gtmeta=(data/"p3_trained_heldout_views.manifest.json")
    meta=json.loads(gtmeta.read_text())
    if (orig3d["frame_ids"]!=new3d["frame_ids"] or
            new3d["frame_ids"]!=training["validation_pose_ids"] or
            not orig3d["independent_stills"] or
            not new3d["independent_stills"] or
            training["trained_checkpoint_sha256"]!=sha(training["trained_checkpoint"]) or
            training["input_checkpoint_sha256"]!=sha(training["input_checkpoint"])):
        raise RuntimeError("P3 mismatch: source data/checkpoint/heldout poses")
    if meta["mode"]!="reader" or meta["frame_ids"]!=training["validation_pose_ids"]:
        raise ValueError("P3 Reader/GT provenance mismatch")
    res={}
    for v in VIEWS:
        before=orig2d[v]["mean_layout_px"]
        after=newer[v]["mean_layout_px"]
        q=newer[v].get("quality") or {}
        calib=q.get("localization_error_calibration") or {}
        res[v]={
            "baseline_2d_px":before,"fine_tuned_2d_px":after,
            "delta_2d_px":round(after-before,5),
            "mesh_visibility_auc":q.get("auc"),
            "mesh_visibility_brier":q.get("brier"),
            "mesh_visibility_ece":q.get("ece_8bins"),
            "mesh_visibility_known_labels":q.get("known_surface_part_samples"),
            "mesh_visibility_occluded_labels":q.get("occluded_surface_part_samples"),
            "expected_error_calibration_gap_px":calib.get("mean_error_calibration_gap_px"),
        }
    base3=orig3d["paired_3d_comparison"]
    p3_3=new3d["paired_3d_comparison"]
    two_view_no_regression=(p3_3["two_view_mean_3d_mm"]<=base3["two_view_mean_3d_mm"])
    four_better_than_two=(p3_3["four_view_mean_3d_mm"]<p3_3["two_view_mean_3d_mm"])
    full_2d_gain=(newer["all_views_mean_layout_px"]<orig2d["all_views_mean_layout_px"])
    vis_gate=all((res[v]["mesh_visibility_auc"] or 0)>=.80 and
                 (res[v]["mesh_visibility_ece"] or 1)<=.10 for v in VIEWS)
    error_gate=all(res[v]["expected_error_calibration_gap_px"] is not None and
                    res[v]["expected_error_calibration_gap_px"]<=5 for v in VIEWS)
    result={
        "schema":"anymove.p3_four_view_quality_gate/1",
        "creature":creature,
        "status":"ARCHITECTURE_PASS_MODEL_NOT_PROMOTED",
        "train_pose_count":training["train_poses"],
        "validation_pose_count":training["val_poses"],
        "original_checkpoint_sha256":training["input_checkpoint_sha256"],
        "fine_tuned_checkpoint":training["trained_checkpoint"],
        "fine_tuned_checkpoint_sha256":training["trained_checkpoint_sha256"],
        "visibility_semantics":"Skinning mesh SURFACE PART known/occluded/unknown, not bone center nor calibrated H3",
        "inference_gt_leakage":False,
        "validation":"Blender independent still poses; same exact four camera matrices; NOT continuous motion",
        "2d_layout_px_by_view":res,
        "2d_all_views_mean_layout_px":{
            "original":orig2d["all_views_mean_layout_px"],
            "fine_tuned":newer["all_views_mean_layout_px"],
            "delta":round(newer["all_views_mean_layout_px"]-orig2d["all_views_mean_layout_px"],5)},
        "independent_3d_fk_mm":{
            "original_two_view":base3["two_view_mean_3d_mm"],
            "original_four_view":base3["four_view_mean_3d_mm"],
            "fine_tuned_two_view":p3_3["two_view_mean_3d_mm"],
            "fine_tuned_four_view":p3_3["four_view_mean_3d_mm"],
            "four_view_gain_from_training_mm":round(
                p3_3["four_view_mean_3d_mm"]-base3["four_view_mean_3d_mm"],5)},
        "gates":{
            "2d_all_views_improved":full_2d_gain,
            "no_two_view_3d_regression":two_view_no_regression,
            "trained_four_view_beats_trained_two_view_3d":four_better_than_two,
            "all_views_mesh_visibility_auc_ge_0_80_and_ece_le_0_10":vis_gate,
            "all_views_expected_error_calibration_gap_le_5px":error_gate,
        },
        "production_promotion_allowed":bool(full_2d_gain and two_view_no_regression and
                  four_better_than_two and vis_gate and error_gate),
        "cautions":["The engineering thresholds above are screening criteria, not theoretical truths",
                    "A few Blender static poses are insufficient for reliable H3 generalization",
                    "No true synchronized video or novel Creature test has been run"],
    }
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--creature",required=True)
    ap.add_argument("--repo",default=".")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    out=Path(args.out).resolve()
    if out.exists():raise FileExistsError("P3 audit already exists; use new report path")
    obj=summarize(args.creature,Path(args.repo).resolve())
    out.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps({k:obj[k] for k in
            ("creature","status","train_pose_count","validation_pose_count",
             "2d_all_views_mean_layout_px","independent_3d_fk_mm","gates",
             "production_promotion_allowed")},indent=2))


if __name__=="__main__":
    main()
