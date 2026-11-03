"""Independent release audit/assembly for the complete HR V2 315°/45° dataset.

Does not import Blender, train a Reader, or generate images. Fails closed unless
all 12 motion-family batches, 192 unique clips, 2,688 stereo frames, 5,376
672x672 PNGs, exact 67-marker GT, visibility proxy, and clip-disjoint
Train/Validation/Test partitions are present and internally consistent.
"""
import argparse, hashlib, json, math, pathlib, struct
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from mocap.camera_geometry import project_orthographic
from tools import human_v3_fullbody_motion_v2 as motion

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
SOURCE=pathlib.Path("runs/human/human_skinmax_multicolor_v3_nose.blend")
SOURCE_SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"
FAMILIES=motion.FAMILIES
IMAGE_SIZE=(672,672)
EXPECTED_CLIPS=192
EXPECTED_PAIRS=2688
EXPECTED_PNG=5376
PER_FAMILY_CLIPS=16
PER_CLIP_FRAMES=14
EXPECTED_SPLITS={"train":144,"validation":24,"test":24}
SPLIT_PNG={"train":4032,"validation":672,"test":672}
SLOT_COUNTS={"front":2688,"side":2688}
ANGLE_MAP={"front":315.0,"side":45.0}
M_PER_PX=2.432802200317383/672
Z_CENTER=.8688581585884094

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):
            h.update(buf)
    return h.hexdigest()

def fail(msg):
    raise RuntimeError("HRV2_RELEASE_AUDIT_FAILED: "+str(msg))

def atomic_json(path,payload):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,separators=(",",":"),ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)

def calibrated_cameras():
    """Explicit level-orthographic camera-to-world and world-to-pixel contract."""
    records={}
    for slot,angle in ANGLE_MAP.items():
        radians=math.radians(angle)
        c,s=math.cos(radians),math.sin(radians)
        records[slot]={
            "azimuth_deg":angle,
            "camera_world_position_m":[-20*s,-20*c,Z_CENTER],
            "rotation_euler_xyz_rad":[math.pi/2,0.0,-radians],
            "orthographic_vertical_span_m":2.432802200317383,
            "world_right_unit":[c,-s,0.0],
            "world_up_unit":[0.0,0.0,1.0],
            "world_xyz1_to_pixel_uv_2x4":[
                [c/M_PER_PX,-s/M_PER_PX,0.0,336.0],
                [0.0,0.0,-1.0/M_PER_PX,336.0+Z_CENTER/M_PER_PX]
            ],
        }
    return records

def oracle_training_crop_hints(row, names):
    """TRAIN-only GT boxes for future *supervised* hand/foot crop experiments.

    These boxes must never be supplied as inference-time detections and are
    intentionally absent from Validation/Test crop metadata.
    """
    index={n:i for i,n in enumerate(names)}
    uv=np.asarray(row["uv"],dtype=np.float64)
    boxes={}
    for side in ("L","R"):
        hand=[n for n in names if n.startswith("CC_Base_"+side+"_") and
              (n.endswith("_Hand") or any(digit in n for digit in ("Pinky","Ring","Mid","Index","Thumb")))]
        foot=["CC_Base_"+side+"_"+part for part in ("Foot","ToeBase","ToeTip")]
        for part,group in (("hand",hand),("foot",foot)):
            if not group or any(n not in index for n in group):
                fail("Crop hint references missing body markers")
            points=uv[[index[n] for n in group]]
            lo=points.min(axis=0)
            hi=points.max(axis=0)
            extent=float(max(hi-lo))
            width=int(np.ceil(max(112.0,min(256.0,extent*1.45+20))))
            center=(lo+hi)/2
            x0=int(round(max(0,min(672-width,center[0]-width/2))))
            y0=int(round(max(0,min(672-width,center[1]-width/2))))
            if x0<0 or y0<0 or x0+width>672 or y0+width>672:
                fail("Crop hint outside native 672 image")
            boxes[f"{side}_{part}"]={"xyxy_px":[x0,y0,x0+width,y0+width],"source":"TRAIN_GT_ONLY"}
    return boxes

def audit(smoke=False,hash_images=True):
    dataset=ROOT/("smoke" if smoke else "dataset")
    if not SOURCE.is_file() or sha256(SOURCE)!=SOURCE_SHA:
        fail("Original V3 Nose source Blend SHA mismatch")
    batch_paths=([dataset/"smoke_manifest.json"] if smoke
                 else [dataset/f"family_{i:02d}_manifest.json" for i in range(len(FAMILIES))])
    batches=[]
    for i,path in enumerate(batch_paths):
        if not path.is_file():
            fail("Missing completed family batch "+str(path))
        data=json.loads(path.read_text(encoding="utf-8"))
        if (data.get("state")!="COMPLETE"
           or data.get("schema")!="anymove.human_v3_nose_hr_v2_family_batch/1"
           or data.get("source_blend_sha256")!=SOURCE_SHA
           or data.get("camera_angles_deg")!=[315,45]
           or data.get("render_size")!=[672,672]):
            fail("Batch metadata inconsistent: "+str(path))
        if not smoke and (data["total_clips"]!=PER_FAMILY_CLIPS
           or {c["family"] for c in data["clip_audits"]}!={FAMILIES[i]}):
            fail("Family batch clip count/type mismatch "+str(path))
        batches.append(data)
    names=batches[0]["joint_names"]
    if len(names)!=67 or len(set(names))!=67:
        fail("Expected 67 distinct keypoint names")
    if any(b["joint_names"]!=names for b in batches):
        fail("Joint names/order differ between batches")
    missing=batches[0]["visibility_missing_labels"]
    if len(missing)!=3 or not all(x in names for x in missing):
        fail("Unexpected missing visibility regions")
    joined=[]
    audits=[]
    rows={}
    counts=Counter()
    family_clip=Counter()
    family_split=defaultdict(Counter)
    image_digest={}
    vis_classes=Counter()
    max_uv_deviation=0.0
    gt_cache={}
    split_clips=defaultdict(set)
    clip_frames=defaultdict(set)
    for batch_i,batch in enumerate(batches):
        for ca in batch["clip_audits"]:
            key=ca["clip"]
            if key in rows:fail("Duplicate clip audit "+key)
            rows[key]=ca
            audits.append(ca)
            split_clips[ca["split"]].add(key)
            family_clip[ca["family"]]+=1
            family_split[ca["family"]][ca["split"]]+=1
            if ca["leg_max_travel_m"]<.025 or ca["hand_max_travel_m"]<.03 or ca["smallest_finger_radius_change_m"]<.001:
                fail("Clip not fullbody/finger animated "+key)
        for split,parts in batch["rows"].items():
            for row in parts:
                if row["split"]!=split or split not in EXPECTED_SPLITS:
                    fail("Invalid split name")
                clip=row["clip"]
                if clip not in rows or rows[clip]["split"]!=split:
                    fail("Row references missing or different split clip "+clip)
                if row["slot"] not in ANGLE_MAP or row["camera_azimuth_deg"]!=ANGLE_MAP[row["slot"]]:
                    fail("View/azimuth mismatch")
                name=row["file"]
                if not name.startswith(split+"/"+row["slot"]+"/") or not name.endswith(".png"):
                    fail("PNG directory path mismatch")
                path=dataset/name
                if not path.is_file() or path.stat().st_size<200:
                    fail("Missing or empty PNG "+str(path))
                try:
                    with Image.open(path) as img:
                        if img.size!=IMAGE_SIZE or img.format!="PNG" or img.mode!="RGBA":
                            fail("PNG does not have 672² RGBA pixels "+str(path))
                        if hash_images:
                            img.verify()  # verifies PNG data CRCs
                except Exception as exc:
                    fail("Corrupt or invalid PNG "+str(path)+" "+str(exc))
                if hash_images:
                    image_digest[name]=sha256(path)
                frame=int(row["frame"])
                if not smoke and (frame<=0 or frame>=106):
                    fail("Dataset includes a neutral rest endpoint")
                clip_frames[clip].add(frame)
                if clip not in gt_cache:
                    p=dataset/"motion_gt"/(clip+".json")
                    if not p.exists():fail("Motion GT missing "+str(p))
                    gt=json.loads(p.read_text(encoding="utf-8"))
                    if (gt.get("source_blend_sha256")!=SOURCE_SHA
                        or gt["joint_names"]!=names
                        or gt["clip"]["clip"]!=clip):
                        fail("Motion GT contract mismatch "+clip)
                    xyz=np.asarray(gt["joint_position_m"],dtype=np.float64)
                    if xyz.shape!=(107,67,3) or not np.isfinite(xyz).all():
                        fail("Non-finite/incorrect GT shape "+clip)
                    gt_cache[clip]=xyz
                gt=gt_cache[clip]
                uv=np.asarray(row["uv"],dtype=np.float64)
                if uv.shape!=(67,2) or not np.isfinite(uv).all():
                    fail("Invalid 67×2 pixel GT")
                projected=project_orthographic(gt[frame],row["camera_azimuth_deg"],672,672,M_PER_PX,Z_CENTER)
                deviation=float(np.max(np.abs(uv-projected)))
                max_uv_deviation=max(max_uv_deviation,deviation)
                if deviation>.0025:
                    fail("Camera/GT 2D projection mismatch "+clip)
                if np.min(uv)<0 or np.max(uv)>672:
                    fail("Outside cropped image "+clip)
                vis=row.get("visibility_proxy")
                if not isinstance(vis,list) or len(vis)!=67:
                    fail("Missing per-marker visibility fields")
                for marker,v in zip(names,vis):
                    label=v.get("label")
                    frac=v.get("fraction")
                    if marker in missing:
                        if label!="unknown" or frac is not None or v.get("sample_count")!=0:
                            fail("Invalid unknown visibility for unskinned control "+marker)
                    elif (label not in ("surface_visible","surface_occluded","uncertain")
                          or not isinstance(frac,(float,int)) or not 0<=frac<=1
                          or v.get("sample_count",0)<3):
                        fail("Bad surface visibility proxy "+marker)
                    vis_classes[label]+=1
                key=(clip,frame,row["slot"])
                if key in counts:
                    fail("Duplicate view/clip/frame key "+str(key))
                counts[key]=1
                joined.append(row)
    expected_clip=3 if smoke else EXPECTED_CLIPS
    expected_frames=9 if smoke else EXPECTED_PAIRS
    expected_images=18 if smoke else EXPECTED_PNG
    if len(rows)!=expected_clip or len(joined)!=expected_images:
        fail(f"Incorrect count: clips={len(rows)} images={len(joined)}")
    for clip,frames in clip_frames.items():
        if len(frames)!=(3 if smoke else PER_CLIP_FRAMES):
            fail("Clip has wrong frame count "+clip)
        for fr in frames:
            if (clip,fr,"front") not in counts or (clip,fr,"side") not in counts:
                fail("Missing synchronized stereo pair "+clip+" "+str(fr))
    if not smoke:
        if any(family_clip[f]!=16 or family_split[f]!={"train":12,"validation":2,"test":2} for f in FAMILIES):
            fail("Unbalanced family and split distribution")
        if {split:len(clips) for split,clips in split_clips.items()}!=EXPECTED_SPLITS:
            fail("Train/validation/test clip count mismatch")
        if {k:sum(row["split"]==k for row in joined) for k in SPLIT_PNG}!=SPLIT_PNG:
            fail("Train/validation/test image count mismatch")
        if {slot:sum(row["slot"]==slot for row in joined) for slot in SLOT_COUNTS}!=SLOT_COUNTS:
            fail("Per-camera image count below 2,200")
        if not all(SLOT_COUNTS[slot]>=2200 for slot in SLOT_COUNTS):
            fail("Per-view minimum 2,200 unmet")
        if len(set.union(*split_clips.values()))!=len(rows):
            fail("Clip split data leakage")
    out={
        "schema":"anymove.human_v3_nose_hr_dataset_v2/1",
        "status":"PASS",
        "source_model_sha256":SOURCE_SHA,
        "source_blend":str(SOURCE),
        "camera_slot_azimuth_deg":ANGLE_MAP,
        "camera_calibration":calibrated_cameras(),
        "stereo_pose_pairs":expected_frames,
        "image_count":expected_images,
        "per_view_count":{slot:sum(r["slot"]==slot for r in joined) for slot in ANGLE_MAP},
        "clip_count":len(rows),
        "motion_families":dict(family_clip),
        "motion_family_per_split":{f:dict(family_split[f]) for f in family_split},
        "clip_counts_by_split":{k:len(v) for k,v in split_clips.items()},
        "images_by_split":{k:sum(r["split"]==k for r in joined) for k in ("train","validation","test")},
        "markers":len(names),"marker_names":names,
        "source_native_resolution_px":[WIDTH for WIDTH in IMAGE_SIZE],
        "no_neutral_endpoints_in_training":True,
        "all_stereo_pairs_present":True,
        "max_projected_uv_gt_diff_px":round(max_uv_deviation,7),
        "reprojection_acceptance_threshold_px":0.0025,
        "visibility_missing_markers":missing,
        "visibility_label_distribution":dict(vis_classes),
        "visibility_warning":"Skinned surface raycast fraction is a proxy, NOT calibrated joint visibility/uncertainty or true independent motion GT.",
        "validation_status":"No Reader training performed",
        "files_sha256":str(dataset/"image_sha256.json") if hash_images else None,
    }
    if not smoke:
        train_crop_hints=[{"clip":r["clip"],"frame":r["frame"],"slot":r["slot"],
                           "file":r["file"],"boxes":oracle_training_crop_hints(r,names)}
                          for r in joined if r["split"]=="train"]
        if len(train_crop_hints)!=SPLIT_PNG["train"]:
            fail("TRAIN crop hint count mismatch")
        atomic_json(dataset/"train_only_gt_crop_hints.json",{
            "schema":"anymove.hr_v2_train_only_gt_crop_hints/1",
            "source":"GT-derived for supervised Train ONLY; forbidden as inference-time crop predictions",
            "rows":train_crop_hints})
        atomic_json(dataset/"manifest.json",{
            "schema":"anymove.human_v3_nose_hr_dataset_v2/1",
            "status":"complete","source_blend_sha256":SOURCE_SHA,
            "camera_angles_deg":[315,45],"camera_slot_azimuth_deg":ANGLE_MAP,
            "camera_calibration":calibrated_cameras(),
            "width":672,"height":672,"metres_per_pixel":M_PER_PX,"z_center":Z_CENTER,
            "joint_names":names,"visibility_missing_markers":missing,
            "frame_count":107,"sampled_frames":sorted(next(iter(clip_frames.values()))),
            "train_clips":sorted(split_clips["train"]),
            "validation_clips":sorted(split_clips["validation"]),
            "test_clips":sorted(split_clips["test"]),
            "samples":joined,
            "motion_quality":audits,
            "train_only_gt_crop_hints":"train_only_gt_crop_hints.json",
            "notes":"Blender procedural human V3 Nose motion, native 672 pixels per camera, joints and surface visibility GT",
        })
        out["train_gt_crop_hint_images"]=len(train_crop_hints)
        if hash_images:atomic_json(dataset/"image_sha256.json",image_digest)
        atomic_json(dataset/"audit.json",out)
    else:
        atomic_json(dataset/"smoke_audit.json",out)
    print("HRV2_RELEASE_AUDIT_PASS",json.dumps({k:v for k,v in out.items() if k not in ("marker_names","motion_families","motion_family_per_split")},
                                            ensure_ascii=False),flush=True)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--smoke",action="store_true")
    args.add_argument("--skip-image-crc",action="store_true")
    x=args.parse_args()
    audit(smoke=x.smoke,hash_images=not x.skip_image_crc)
