"""Plan large, pose-disjoint P4 source-material training without sampling
previous P3 IDs or blurred source images. No GPU and no generated video.

The full plan is 240 train + 36 validation from existing 2200 stored
static poses per creature. The explicit *pilot* is a smaller deterministic
subset to verify the P4-B specialized Reader before spending the cost of
rendering the full plan. Independent novel procedural test pose IDs 9001..9012
are NOT included in either train or model-selection validation.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()


def select(source,creature,*,seed=41111,train_count=240,val_count=36):
    root=Path(source)
    allmeta=sorted(root.glob("*_meta.json"))
    if len(allmeta)<2200:
        raise ValueError("P4 source is not the complete original 2200-pose dataset")
    candidates=[]
    for p in allmeta:
        i=int(p.name[:5])
        if i<=100:
            continue # repeatedly examined P3 samples are excluded
        obj=json.loads(p.read_text(encoding="utf8"))
        if obj.get("blur") or float(obj.get("scale",-1))!=1.:
            continue # skin-surface GT labels apply to a single sharp pose
        paths=[root/f"{i:05d}_{v}.png" for v in ("front","side")]
        if not all(x.is_file() for x in paths):
            continue
        candidates.append(i)
    train=[i for i in candidates if i%25!=0]
    val=[i for i in candidates if i%25==0]
    rng=np.random.default_rng(seed)
    train=rng.permutation(train).tolist()
    val=rng.permutation(val).tolist()
    if len(train)<train_count or len(val)<val_count:
        raise ValueError("not enough unblurred pose-disjoint original GT")
    full_train=sorted(train[:train_count]);full_val=sorted(val[:val_count])
    # Pilot deliberately chooses a strict subset of the full plan.
    pilot_train=sorted(train[:64]);pilot_val=sorted(val[:12])
    novel_test=list(range(9001,9013))
    if set(full_train)&set(full_val) or set(novel_test)&set(full_train+full_val):
        raise AssertionError("P4 split overlap")
    for group in (full_train,full_val):
        if any(i<=100 for i in group):
            raise AssertionError("old repeatedly-tested P3 sample reused")
    return {
        "schema":"anymove.p4_pose_selection/1",
        "creature":creature,
        "source":str(root.resolve()),
        "source_original_metadata_count":len(allmeta),
        "source_unblurred_eligible":len(candidates),
        "selection_seed":seed,
        "selection_protocol":"exclude P3 1..100, use only nonblurred old stored GT; i%25!=0 train / i%25==0 validation; independent novel indices 9001..9012 not included",
        "train_ids":full_train,"val_ids":full_val,"test_ids":novel_test,
        "pilot_train_ids":pilot_train,"pilot_val_ids":pilot_val,
        "pilot_ids_render":sorted(pilot_train+pilot_val),
        "full_ids_render":sorted(full_train+full_val),
        "no_new_h3_video":True,
        "not_motion_sequence":True
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--creature",required=True,choices=("dino","dragon"))
    p.add_argument("--source",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    dest=Path(a.out)
    if dest.exists():raise FileExistsError("P4 split plan already exists")
    result=select(a.source,a.creature)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps({k:result[k] if k not in ("train_ids","val_ids","pilot_ids_render","full_ids_render") else len(result[k]) for k in result},indent=2))


if __name__=="__main__":main()
