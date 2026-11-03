"""Independent 2-vs-4 N-camera 3D error audit against held-out Blender GT.

Not imported by the fit solver. Exact independent 3D Blender world joint
data is available ONLY here, in *.ground_truth.npz, not in Solver input.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap.multiview import load_packet,triangulate,project_np
from mocap.solve import Rig


def error_stat(x):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all():raise ValueError("nonfinite ground truth error")
    return {"mean":round(float(x.mean()),5),
            "median":round(float(np.median(x)),5),
            "p95":round(float(np.percentile(x,95)),5),
            "max":round(float(x.max()),5)}


def data_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(2**20),b""):h.update(x)
    return h.hexdigest()


def evaluate_packet(packet_path,gt_path,rig_info,solutions):
    info=json.loads(Path(rig_info).read_text(encoding="utf8"))
    packet,meta=load_packet(packet_path,rig_names=[k["name"] for k in info["keypoints"]])
    with np.load(gt_path,allow_pickle=False) as file:
        world=np.asarray(file["joints_world"],dtype=float)
        exact_uv=np.asarray(file["exact_uv"],dtype=float)
        names=list(map(str,file["names"]))
        frame_ids=np.asarray(file["frame_ids"],dtype=int)
        visibility=np.asarray(file["part_visibility"],dtype=np.int8)
    if (world.shape!=(meta["T"],meta["J"],3) or
            exact_uv.shape!=(meta["V"],meta["T"],meta["J"],2) or
            visibility.shape!=(meta["V"],meta["T"],meta["J"]) or
            names!=meta["names"] or
            not np.array_equal(frame_ids,packet["frame_ids"])):
        raise ValueError("3D GT evidence does not match observation packet schema")
    projected=project_np(world,packet["world_to_camera"],
                         packet["image_size_px"],packet["meters_per_px"])
    deviation=float(np.max(np.linalg.norm(projected-exact_uv,axis=-1)))
    if deviation>.08:
        raise ValueError("independent GT camera reprojection mismatch")
    root_idx=meta["names"].index(info["bones"][0]["head_joint"])
    obs_errors=np.linalg.norm(np.asarray(packet["uv"])-exact_uv,axis=-1)
    per_view={}
    for vi,name in enumerate(meta["view_ids"]):
        valid=np.asarray(packet["valid"])[vi]
        per_view[name]={
            "all_reader_input_error_px":error_stat(obs_errors[vi][valid]),
            "visible_part_reader_error_px":error_stat(obs_errors[vi][valid & (visibility[vi]==1)]) if
                 (valid & (visibility[vi]==1)).any() else None,
            "hidden_part_reader_error_px":error_stat(obs_errors[vi][valid & (visibility[vi]==0)]) if
                 (valid & (visibility[vi]==0)).any() else None,
            "num_valid":int(valid.sum()),
            "visible_part_count":int((visibility[vi]==1).sum()),
            "hidden_part_count":int((visibility[vi]==0).sum()),
            "unknown_part_count":int((visibility[vi]==-1).sum()),
        }
    result={"schema_version":"anymove.nview_gt_eval/1","source_kind":meta["source_kind"],
        "packet":str(packet_path),"packet_sha256":data_sha(packet_path),
        "independent_gt_eval_only":str(gt_path),"gt_sha256":data_sha(gt_path),
        "rig":str(rig_info),"frames":meta["T"],"joint_count":meta["J"],
        "frame_ids":frame_ids.tolist(),"independent_stills":meta["independent_stills"],
        "all_camera_source_3d_gt_reprojection_max_px":deviation,
        "per_view_2d_input_quality":per_view,
        "two_vs_four":{}}

    for tag,views in (("two_view",["front","side"]),
                      ("four_view",["front","side","back","opposite_side"])):
        selected=np.array([meta["view_ids"].index(v) for v in views])
        tri,tri_meta=triangulate(packet,index_views=selected)
        # Out-of-frame Reader joints can be rank-deficient: report them
        # explicitly and evaluate raw triangulation ONLY on valid points.
        # FK output is still evaluated on ALL keypoints using independent GT.
        mask=tri_meta["triangulable"]
        if not mask.any():
            raise ValueError("no rank-valid ray intersection for GT audit")
        tri_error=np.linalg.norm(tri-world,axis=-1)*1000
        root_ok=mask[:,root_idx]
        entry={
            "selected_views":views,"triangulation_error_mm":error_stat(tri_error[mask]),
            "triangulation_metric_scope":"RANK_VALID_POINTS_ONLY",
            "triangulation_root_error_mm":error_stat(tri_error[:,root_idx][root_ok]) if root_ok.any() else None,
            "triangulated_rank_ok":tri_meta["triangulable_count"],
            "triangulated_rank_deficient_count":int((~mask).sum()),
        }
        solution=Path(solutions[tag])
        if solution.is_file():
            manifest_path=solution.with_suffix('.json')
            if not manifest_path.is_file():
                raise ValueError('missing N-view Solver provenance manifest')
            solve_manifest=json.loads(manifest_path.read_text(encoding='utf8'))
            if (solve_manifest.get('schema_version')!='anymove.multiview_fk_solver_experimental/1' or
                    Path(solve_manifest.get('source_packet','')).resolve()!=Path(packet_path).resolve() or
                    solve_manifest.get('active_views')!=views or
                    solve_manifest.get('frames')!=meta['T'] or
                    solve_manifest.get('joint_count')!=meta['J']):
                raise ValueError('Solver provenance does not match GT test packet/views')
            recorded_sha=solve_manifest.get('source_packet_sha256')
            if recorded_sha is not None and recorded_sha!=data_sha(packet_path):
                raise ValueError('source NPZ changed since N-camera Solver fit')
            with np.load(solution,allow_pickle=False) as z:
                got={k:z[k].copy() for k in z.files}
            if (got["world_joints"].shape!=world.shape or
                    list(map(str,got["names"]))!=meta["names"] or
                    not np.array_equal(got["frame_ids"],frame_ids)):
                raise ValueError("N-view Solver results conflict with GT frame/joint provenance")
            if list(map(str,got["view_ids"]))!=views:
                raise ValueError("selected views differ from saved Solver prediction")
            fit=np.asarray(got["world_joints"],float)
            error=np.linalg.norm(fit-world,axis=-1)*1000
            with torch.no_grad():
                # Rig-forward consistency is evaluated on CPU, same FK definition
                r=Rig(info,dev="cpu")
                actual,_=r.forward(
                    torch.tensor(got["trans"],dtype=torch.float64),
                    torch.tensor(got["root_rotmat"],dtype=torch.float64),
                    torch.tensor(got["euler"],dtype=torch.float64))
                fk_error=float(np.linalg.norm(actual.cpu().numpy()-fit,axis=-1).max())
            if fk_error>.0003:
                raise ValueError(f"saved optimized 3D joints do not match FK: {fk_error}")
            all_views_project=project_np(fit,packet["world_to_camera"],
                                         packet["image_size_px"],packet["meters_per_px"])
            residual_observed=np.linalg.norm(all_views_project-packet["uv"],axis=-1)
            reproj_truth=np.linalg.norm(all_views_project-exact_uv,axis=-1)
            entry["solver_error_mm"]=error_stat(error)
            entry["solver_root_error_mm"]=error_stat(error[:,root_idx])
            entry["solver_fk_roundtrip_max_m"]=round(fk_error,9)
            entry["per_frame_solver_mean_error_mm"]=np.round(error.mean(axis=1),5).tolist()
            entry["per_view_solver_reproj_to_input_px"]={
                v:error_stat(residual_observed[i]) for i,v in enumerate(meta["view_ids"])}
            entry["per_view_solver_reproj_to_exact_gt_px"]={
                v:error_stat(reproj_truth[i]) for i,v in enumerate(meta["view_ids"])}
            entry["joint_ground_truth_3d_error_is_independent_of_solver_2d_loss"]=True
        else:
            entry["solver_status"]="NOT_RUN"
        result["two_vs_four"][tag]=entry
    if all("solver_error_mm" in e for e in result["two_vs_four"].values()):
        a=result["two_vs_four"]["two_view"]["solver_error_mm"]["mean"]
        b=result["two_vs_four"]["four_view"]["solver_error_mm"]["mean"]
        result["paired_3d_comparison"]={
            "two_view_mean_3d_mm":a,"four_view_mean_3d_mm":b,
            "four_minus_two_mm":round(b-a,5),
            "four_view_improves_this_small_sample":b<a,
            "not_proof_of_generalization":True}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--packet",required=True)
    parser.add_argument("--ground-truth",required=True)
    parser.add_argument("--rig",required=True)
    parser.add_argument("--two",required=True)
    parser.add_argument("--four",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    output=Path(args.out)
    if output.exists():raise FileExistsError("do not overwrite existing GT evaluation")
    results=evaluate_packet(args.packet,args.ground_truth,args.rig,
                {"two_view":args.two,"four_view":args.four})
    output.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps({"packet":results["packet"],"3d":results.get("paired_3d_comparison"),
            "2d_input":{k:v["all_reader_input_error_px"]["mean"]
                         for k,v in results["per_view_2d_input_quality"].items()},
            "triangulation_mm":{
                k:v["triangulation_error_mm"]["mean"] for k,v in
                results["two_vs_four"].items()}},indent=2))


if __name__=="__main__":
    main()
