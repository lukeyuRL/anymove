"""Audit what mesh occlusion and P2 error-gates actually detect vs Blender 2D GT.

A flagged observation can be truly positionally wrong WITHOUT being occluded.
This independently tests whether those proposed masks select high-2D-error
samples, with no oracle labels supplied to the mask generators themselves.

Usage:
python -m tools.audit_dino_occlusion_sources RAW_DET.npz GT_JOINTS.json \
  GEOMETRY.npz P2_DET.npz NEW_REPORT.json --threshold-px 10

The 2D GT is used ONLY for post-hoc evaluation, never to build mesh/P2 masks.
"""
import argparse
import json
import pathlib

import numpy as np
from mocap import layout
from tools.evaluate_dino_full_body import _groups


def rate(selected,bad):
    selected=np.asarray(selected,bool)
    bad=np.asarray(bad,bool)
    if selected.shape!=bad.shape:
        raise ValueError("mask/bad shape mismatch")
    tp=int(np.count_nonzero(selected & bad))
    flagged=int(np.count_nonzero(selected))
    positive=int(np.count_nonzero(bad))
    return {
        "flagged_count":flagged,
        "bad_observations":positive,
        "true_positive_count":tp,
        "precision_for_large_2d_error":None if not flagged else round(tp/flagged,6),
        "recall_of_large_2d_error":None if not positive else round(tp/positive,6),
    }


def evaluate(raw,truth,geometry,quality,threshold_px=10):
    names=[str(n) for n in raw["names"]]
    if truth["names"]!=names or [str(n) for n in geometry["names"]]!=names or names!=[str(n) for n in quality["names"]]:
        raise ValueError("joint name/order mismatch")
    points=np.asarray(truth["joints"],float)
    T,J=points.shape[:2]
    if J!=92 or points.shape!=(T,J,3) or T<3 or not np.isfinite(points).all():
        raise ValueError("invalid Blender 3D joint truth")
    if not np.isfinite(threshold_px) or threshold_px<=0:
        raise ValueError("invalid high-error threshold")
    front,side=layout.project(points)
    labels,groups=_groups(names)
    report={"profile":"anymove.dino_mesh_vs_error_observation_selection_audit/1",
            "frames":T,"keypoints":J,"threshold_px":float(threshold_px),
            "note":"Large 2D error is NOT an anatomical occlusion ground truth; this is source selection quality, evaluated post hoc only.",
            "views":{}}
    for view,uvgt in (("front",front),("side",side)):
        uv=np.asarray(raw[view+"_uv"],float)
        if uv.shape!=(T,J,2):
            raise ValueError("invalid raw Reader observation shape")
        err=np.linalg.norm(uv-uvgt,axis=-1)
        bad=err>=threshold_px
        blocked=np.asarray(geometry[view+"_mesh_blocked_proxy"])
        support=np.asarray(geometry[view+"_mesh_support"])
        pred=np.asarray(quality[view+"_expected_error_px"],float)
        if (blocked.shape!=(T,J) or support.shape!=(T,J) or pred.shape!=(T,J)
                or not np.isfinite(pred).all() or (pred<0).any()):
            raise ValueError("invalid geometry/P2 profile")
        sources={
            "mesh_proxy":blocked,
            "p2_error_threshold":pred>=threshold_px,
            "union":blocked | (pred>=threshold_px),
        }
        info={
            "raw_mean_2d_error_px":round(float(err.mean()),5),
            "raw_max_2d_error_px":round(float(err.max()),5),
            "true_high_2d_error_count":int(bad.sum()),
            "unknown_mesh_support_count":int((~support).sum()),
            "sources":{key:rate(mask,bad) for key,mask in sources.items()},
            "by_group":{},
        }
        for group,indices in groups.items():
            if not len(indices):continue
            info["by_group"][group]={
                "joints":int(len(indices)),
                "raw_mean_2d_error_px":round(float(err[:,indices].mean()),5),
                "bad_observations":int(bad[:,indices].sum()),
                "sources":{key:rate(mask[:,indices],bad[:,indices])
                           for key,mask in sources.items()},
            }
        report["views"][view]=info
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw")
    parser.add_argument("truth")
    parser.add_argument("geometry")
    parser.add_argument("quality")
    parser.add_argument("output")
    parser.add_argument("--threshold-px",type=float,default=10.)
    args=parser.parse_args(argv)
    target=pathlib.Path(args.output)
    if target.exists() or target.resolve() in {
            pathlib.Path(p).resolve() for p in
            (args.raw,args.truth,args.geometry,args.quality)}:
        raise FileExistsError("do not overwrite occlusion source inputs/output")
    def npz(path):
        with np.load(path,allow_pickle=False) as data:
            return {key:data[key] for key in data.files}
    raw=npz(args.raw)
    geom=npz(args.geometry)
    q=npz(args.quality)
    gt=json.loads(pathlib.Path(args.truth).read_text(encoding="utf-8"))
    report=evaluate(raw,gt,geom,q,args.threshold_px)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
