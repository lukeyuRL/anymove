"""P3 controlled hybrid Reader ablation, no training or GT leakage.

Front and Side keep the validated original 2-view Reader.
Back and Opposite Side use the experimental four-view fine-tuned Reader.
This preserves existing performance as much as possible while testing
whether only OOD-view adaptation adds useful information.

The output is a new 2D OBSERVATION packet, no 3D GT, no material/occlusion
oracle, no automatic use of uncalibrated quality probabilities.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from mocap.multiview import load_packet,save_packet

VIEWS=("front","side","back","opposite_side")


def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(2**20),b""):
            h.update(b)
    return h.hexdigest()


def blend_packets(base_path,adapted_path,output):
    old,meta=load_packet(base_path)
    new,meta2=load_packet(adapted_path)
    if meta["source_kind"]!="reader_inference" or meta2["source_kind"]!="reader_inference":
        raise ValueError("hybrid requires actual Reader observations")
    if meta["view_ids"]!=meta2["view_ids"] or meta["view_ids"]!=list(VIEWS):
        raise ValueError("view order mismatch")
    for k in ("view_ids","names","world_to_camera","image_size_px",
              "meters_per_px","frame_ids","fps","independent_stills"):
        if not np.array_equal(old[k],new[k]):
            raise ValueError("hybrid camera or Rig differs: "+k)
    if meta["T"]<1 or meta["J"]<1:
        raise ValueError("empty hybrid data")
    packet={k:v.copy() for k,v in old.items()
            if k not in ("visibility_score","expected_error_px","quality_semantics")}
    for key in ("uv","confidence","valid"):
        packet[key][2:]=new[key][2:]
    # Explicit per-View provenance is allowed as metadata but has no effect
    # on Solver loss. No GT or mesh-surface visibility labels are included.
    packet["reader_source_by_view"]=np.asarray(("original_front_side",
                 "original_front_side","new_four_view_back",
                 "new_four_view_opposite_side"))
    dest=Path(output)
    if dest.exists() or dest.with_suffix(".manifest.json").exists():
        raise FileExistsError("hybrid output path already exists")
    meta_out=save_packet(dest,packet)
    report={
        "schema":"anymove.p3_four_view_hybrid_reader/1",
        "source_original":str(Path(base_path).resolve()),
        "original_input_sha256":sha(base_path),
        "source_finetuned":str(Path(adapted_path).resolve()),
        "finetuned_input_sha256":sha(adapted_path),
        "output":str(dest.resolve()),
        "view_ids":meta_out["view_ids"],
        "frame_ids":packet["frame_ids"].tolist(),
        "reader_source_by_view":packet["reader_source_by_view"].tolist(),
        "no_gt_leakage":True,
        "no_visibility_oracle":True,
        "no_calibrated_quality_in_loss":True,
        "no_motion_generation_or_h3":True,
        "not_a_validated_production_model":True,
    }
    dest.with_suffix(".manifest.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--original",required=True)
    ap.add_argument("--finetuned",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    report=blend_packets(args.original,args.finetuned,args.out)
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
