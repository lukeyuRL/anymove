"""Train-derived temporal joint association and skeleton-length safety gate.

Pure NumPy; no GT or Blender needed at inference. First fit a hierarchical
calibration using TRAIN-only frames and a rig hierarchy extracted from Blender.
Then track each clip in strict frame order, rejecting 2D outliers that violate
projected bone lengths or train-calibrated relative motion continuity. A bad
measurement is inferred from its last reliable relative pose or Train mean
(never from test labels) and assigned low reliability. Independently cap
catastrophic 3D bone-length violations and preserve explicit provenance.

This P1 is a safety gate, not a motion prior or complete anatomically
constrained IK solver. It is opt-in, never applied silently to any rig.
"""
import hashlib
import json
import math
import pathlib
from collections import defaultdict

import numpy as np

from mocap.camera_geometry import triangulate_orthographic, project_orthographic

SCHEMA="anymove.human_v3_hr_v2_temporal_skeleton_calibration/1"
SOURCE_SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"

def _check_tree(names, parent_names):
    names=list(names)
    if len(names)!=67 or len(set(names))!=67:
        raise ValueError("Expected exact 67-marker human V3 hierarchy")
    if set(names)!=set(parent_names):
        raise ValueError("Rig parent tree must include exactly all 67 markers")
    index={n:i for i,n in enumerate(names)}
    parent=[]
    for j,name in enumerate(names):
        p=parent_names[name]
        if p is None:parent.append(-1)
        elif p not in index or index[p]>=j:
            raise ValueError("Parent must precede child: "+name)
        else:parent.append(index[p])
    if sum(p==-1 for p in parent)!=1:
        raise ValueError("Expected a single human root")
    return parent

def _row_manifest(manifest):
    if (manifest.get("schema")!="anymove.human_v3_nose_hr_dataset_v2/1"
        or manifest.get("status")!="complete"
        or manifest.get("source_blend_sha256")!=SOURCE_SHA
        or manifest.get("camera_angles_deg")!=[315,45]
        or manifest.get("width")!=672 or manifest.get("height")!=672):
        raise ValueError("Unapproved native HR V2 dataset")
    split={s:set(manifest[s+"_clips"]) for s in ("train","validation","test")}
    if [len(split[s]) for s in ("train","validation","test")]!=[144,24,24]:
        raise ValueError("Train/Validation/Test split counts invalid")
    if any(split[a]&split[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Clip split leakage")
    return split

def calibration_from_train(manifest,parent_names,train_gt_root):
    """Only use TRAIN sample rows and TRAIN 3D GT; reject unknown sources."""
    splits=_row_manifest(manifest)
    names=manifest["joint_names"]
    parents=_check_tree(names,parent_names)
    clips=defaultdict(lambda:{"front":[],"side":[]})
    for row in manifest["samples"]:
        if row["split"]!="train":
            continue
        if row["clip"] not in splits["train"]:
            raise ValueError("TRAIN sample references non-TRAIN clip")
        slot=row["slot"]
        if slot not in ("front","side") or row["camera_azimuth_deg"]!=(315 if slot=="front" else 45):
            raise ValueError("TRAIN camera slot mismatch")
        uv=np.asarray(row["uv"],dtype=np.float64)
        if uv.shape!=(67,2) or not np.isfinite(uv).all():
            raise ValueError("Invalid TRAIN UV")
        clips[row["clip"]][slot].append((int(row["frame"]),uv))
    if len(clips)!=144:
        raise ValueError("Expected exactly 144 TRAIN motion clips")
    gt_train=[]
    for clip in sorted(clips):
        path=pathlib.Path(train_gt_root)/(clip+".json")
        if not path.exists():
            raise ValueError("TRAIN GT not found: "+clip)
        raw=json.loads(path.read_text(encoding="utf-8"))
        if raw.get("source_blend_sha256")!=SOURCE_SHA or raw["joint_names"]!=names or raw["clip"]["split"]!="train":
            raise ValueError("TRAIN GT mismatch")
        xyz=np.asarray(raw["joint_position_m"],dtype=np.float64)
        if xyz.shape!=(107,67,3) or not np.isfinite(xyz).all():
            raise ValueError("Bad TRAIN GT shape")
        gt_train.append(xyz)
    all_gt=np.concatenate(gt_train)
    bone_length=[]
    mean_3d=[]
    for j,p in enumerate(parents):
        if p<0:
            bone_length.append(0.0)
            mean_3d.append([0.0,0.0,0.0])
        else:
            delta=all_gt[:,j]-all_gt[:,p]
            bone_length.append(float(np.median(np.linalg.norm(delta,axis=-1))))
            mean_3d.append(np.mean(delta,axis=0).tolist())
    cameras={}
    for slot in ("front","side"):
        relative=[[] for _ in names]
        velocities=[[] for _ in names]
        for clip in sorted(clips):
            rows=sorted(clips[clip][slot],key=lambda r:r[0])
            if len(rows)!=14 or len({r[0] for r in rows})!=14:
                raise ValueError("TRAIN clip missing unique 14-frame sequence")
            frames=np.array([f for f,_ in rows],dtype=int)
            xy=np.stack([uv for _,uv in rows])
            for j,p in enumerate(parents):
                vec=xy[:,j] if p<0 else xy[:,j]-xy[:,p]
                relative[j].append(vec)
                dt=np.diff(frames)
                vel=np.linalg.norm(np.diff(vec,axis=0),axis=-1)/dt
                velocities[j].append(vel)
        local=[]
        for j in range(67):
            points=np.concatenate(relative[j],axis=0)
            speed=np.concatenate(velocities[j],axis=0)
            maxlen=float(np.max(np.linalg.norm(points,axis=-1)))
            maxspeed=float(np.percentile(speed,99.9))
            local.append({
                "mean_delta_xy":np.mean(points,axis=0).tolist(),
                "length_gate_px":float(max(24.,maxlen*2.6+20.)),
                "velocity_gate_px_per_frame":float(max(1.6,maxspeed*1.8+0.9)),
                "train_max_segment_px":maxlen,
                "train_p99.9_speed_px_per_frame":maxspeed,
            })
        cameras[slot]=local
    return {
        "schema":SCHEMA,
        "source_blend_sha256":SOURCE_SHA,
        "camera_azimuth_deg":[315,45],
        "image_width":672,"image_height":672,
        "metres_per_pixel":float(manifest["metres_per_pixel"]),
        "z_center":float(manifest["z_center"]),
        "names":names,
        "parent_names":parent_names,
        "parent_indices":parents,
        "rest_lengths_m":bone_length,
        "mean_relative_xyz_m":mean_3d,
        "calibration_clip_count":len(clips),
        "calibration_joint_view_count":144*14*2*67,
        "cameras":cameras,
        "limits_source":"Train 3D bone GT + Train 2D joint-relative peak speeds; NO Validation or Test",
        "fallback_confidence_heuristic":0.2,
        "untrusted_confidence":0.0,
    }

def _validated_calibration(calib,names):
    if calib.get("schema")!=SCHEMA or calib.get("source_blend_sha256")!=SOURCE_SHA:
        raise ValueError("Unknown skeleton calibration contract")
    if calib.get("camera_azimuth_deg")!=[315,45] or calib.get("image_width")!=672:
        raise ValueError("Wrong camera calibration or resolution")
    if calib.get("names")!=list(names):
        raise ValueError("Wrong Reader joint names/order")
    parents=_check_tree(names,calib["parent_names"])
    if parents!=calib["parent_indices"]:
        raise ValueError("Bad skeleton parent indices")
    if len(calib["rest_lengths_m"])!=67:
        raise ValueError("Missing bone lengths")
    for slot in ("front","side"):
        if len(calib["cameras"][slot])!=67:
            raise ValueError("Incomplete train-only camera gates")
    return parents

def _isolated_spike_mask(raw,frames,parents,gates):
    """Reject only a brief one-frame excursion contradicted by both neighbours.

    Compared with a single forward speed limit, a symmetric three-frame
    observation avoids rejecting legitimate high-speed but continuous motion.
    P1 runs offline on a complete clip; time windows do not cross clips.
    """
    T,J=raw.shape[:2]
    result=np.zeros((T,J),dtype=bool)
    if T<3:
        return result
    for j,p in enumerate(parents):
        local=raw[:,j] if p<0 else raw[:,j]-raw[:,p]
        for t in range(1,T-1):
            dt0=int(frames[t]-frames[t-1])
            dt1=int(frames[t+1]-frames[t])
            prev_jump=np.linalg.norm(local[t]-local[t-1])
            next_jump=np.linalg.norm(local[t]-local[t+1])
            neighbours=np.linalg.norm(local[t-1]-local[t+1])
            threshold=max(40.0,2.5*gates[j]["velocity_gate_px_per_frame"]*min(dt0,dt1)+5.)
            if (prev_jump>threshold and next_jump>threshold
                and neighbours<.30*min(prev_jump,next_jump)):
                result[t,j]=True
    return result


def _guard_one_view(raw,frames,parents,gates):
    T,J=raw.shape[:2]
    corrected=raw.copy()
    weights=np.ones((T,J),dtype=np.float64)
    rejected=np.zeros((T,J),dtype=bool)
    reasons=defaultdict(int)
    spikes=_isolated_spike_mask(raw,frames,parents,gates)
    last_valid_delta=[None]*J
    last_valid_frame=[-1]*J
    inferred_run=[0]*J
    for ti,frame in enumerate(frames):
        for j,parent in enumerate(parents):
            # Compare against the RAW parent, so correcting a single parent
            # cannot cause cascading false rejections of its good children.
            observed=raw[ti,j] if parent<0 else raw[ti,j]-raw[ti,parent]
            reference=np.zeros(2) if parent<0 else corrected[ti,parent]
            cfg=gates[j]
            spatial=bool(np.linalg.norm(observed)>cfg["length_gate_px"])
            temporal=bool(spikes[ti,j])
            invalid=spatial or temporal
            prev=last_valid_delta[j]
            dt=int(frame-last_valid_frame[j])
            if invalid:
                rejected[ti,j]=True
                reasons["spatial" if spatial else "isolated_temporal_spike"]+=1
                inferred_run[j]+=1
                if 0<ti<T-1 and not spikes[ti-1,j] and not spikes[ti+1,j]:
                    # Interpolate only when BOTH neighbouring detections are
                    # plausible in the same TRAIN-derived skeleton bounds.
                    left=raw[ti-1,j] if parent<0 else raw[ti-1,j]-raw[ti-1,parent]
                    right=raw[ti+1,j] if parent<0 else raw[ti+1,j]-raw[ti+1,parent]
                    if (np.linalg.norm(left)<=cfg["length_gate_px"]
                        and np.linalg.norm(right)<=cfg["length_gate_px"]):
                        fraction=(frame-frames[ti-1])/(frames[ti+1]-frames[ti-1])
                        delta=(1-fraction)*left+fraction*right
                    else:
                        delta=None
                else:
                    delta=None
                if delta is None:
                    if prev is not None and dt<=28 and inferred_run[j]<=3:
                        delta=np.asarray(prev,dtype=np.float64).copy()
                    else:
                        delta=np.asarray(cfg["mean_delta_xy"],dtype=np.float64)
                norm=np.linalg.norm(delta)
                if norm>cfg["length_gate_px"]:
                    delta*=cfg["length_gate_px"]/max(norm,1e-8)
                corrected[ti,j]=(reference+delta) if parent>=0 else delta
                weights[ti,j]=0.2
                if parent>=0 and weights[ti,parent]<1:
                    weights[ti,j]=min(weights[ti,j],weights[ti,parent])
            else:
                corrected[ti,j]=raw[ti,j]
                if parent>=0 and weights[ti,parent]<1:
                    weights[ti,j]=min(weights[ti,j],0.5)
                if parent<0 or weights[ti,parent]>=1:
                    last_valid_delta[j]=observed.copy()
                    last_valid_frame[j]=int(frame)
                    inferred_run[j]=0
    return corrected,weights,rejected,dict(reasons)

def track_two_view(front_uv,side_uv,frame_indices,calibration,names, *, apply_3d_repair=False):
    """Joint-temporal + skeleton constraints on one synchronized camera clip.

    All values are inferred from Reader positions and Train-only calibration.
    Provides corrected 2D observations, explicit validity weights and derived
    3D coordinates. Never consumes ground-truth labels at inference.
    """
    p=_validated_calibration(calibration,names)
    f=np.asarray(front_uv,dtype=np.float64)
    s=np.asarray(side_uv,dtype=np.float64)
    frame=np.asarray(frame_indices,dtype=int)
    if f.shape!=s.shape or f.ndim!=3 or f.shape[-2:]!=(67,2):
        raise ValueError("Expected paired (T,67,2) Reader UV arrays")
    if frame.shape!=(f.shape[0],) or not np.all(np.diff(frame)>0):
        raise ValueError("Frame indices must be strictly increasing")
    if not np.isfinite(f).all() or not np.isfinite(s).all():
        raise ValueError("Non-finite Reader output")
    f1,fw,fb,fr=_guard_one_view(f,frame,p,calibration["cameras"]["front"])
    s1,sw,sb,sr=_guard_one_view(s,frame,p,calibration["cameras"]["side"])
    one=np.ones(f.shape[:2],dtype=np.float64)
    xyz=triangulate_orthographic(
        f1,s1,one,one,
        front_width=672,side_width=672,height=672,
        metres_per_pixel=calibration["metres_per_pixel"],
        z_center=calibration["z_center"],
        front_angle_deg=315,side_angle_deg=45,
        front_height_weight=1.0)
    # A gross 3D bone-length violation still indicates unsafe multi-view
    # fusion. Repair from the last credible segment direction or TRAIN mean,
    # NEVER by treating the distorted measurement as ground truth.
    constrained=xyz.copy()
    bone_repaired=np.zeros(f.shape[:2],dtype=bool)
    bone_untrusted=np.zeros(f.shape[:2],dtype=bool)
    last_direction=[None]*67
    for t in range(len(frame)):
        for j,parent in enumerate(p):
            if parent<0:continue
            length=float(calibration["rest_lengths_m"][j])
            if length<1e-5:continue
            delta=constrained[t,j]-constrained[t,parent]
            dist=float(np.linalg.norm(delta))
            # Only gross >22 cm bone stretch qualifies for destructive
            # correction. Small 1-5 px residuals must remain observed.
            limit=max(length*1.65,length+0.22)
            extreme=dist>limit
            if extreme:
                bone_untrusted[t,j]=True
                fw[t,j]=min(fw[t,j],0.2)
                sw[t,j]=min(sw[t,j],0.2)
                if not apply_3d_repair:
                    continue
                direction=last_direction[j]
                if direction is None:
                    direction=np.asarray(calibration["mean_relative_xyz_m"][j],dtype=np.float64)
                    lengthd=np.linalg.norm(direction)
                    if lengthd<1e-8:
                        direction=delta
                lengthd=np.linalg.norm(direction)
                if lengthd<1e-8:continue
                constrained[t,j]=constrained[t,parent]+direction/lengthd*length
                bone_repaired[t,j]=True
                # Both projected observations become imputed rather than
                # direct evidence and must be downweighted for downstream FK.
                fw[t,j]=min(fw[t,j],0.2)
                sw[t,j]=min(sw[t,j],0.2)
            elif min(fw[t,j],sw[t,j])>=0.9 and dist>1e-8:
                last_direction[j]=delta/dist
    # If the constrained skeleton differs, export the matched projection for
    # every inferred 3D joint; keep true detections unchanged otherwise.
    if bone_repaired.any():
        for slot,arr in (("front",f1),("side",s1)):
            cam=315 if slot=="front" else 45
            projection=project_orthographic(constrained,cam,672,672,
                calibration["metres_per_pixel"],calibration["z_center"])
            arr[bone_repaired]=projection[bone_repaired]
    return {
        "front_uv":f1,"side_uv":s1,
        "front_reliability":fw,"side_reliability":sw,
        "front_inferred":fb,"side_inferred":sb,
        "skeleton_repaired":bone_repaired,
        "skeleton_untrusted":bone_untrusted,
        "joints_world_m":constrained,
        "rejected_2d_by_reason":{"front":fr,"side":sr},
        "diagnostics":{
            "rejected_front":int(fb.sum()),
            "rejected_side":int(sb.sum()),
            "skeleton_repaired":int(bone_repaired.sum()),
            "skeleton_untrusted":int(bone_untrusted.sum()),
            "max_inferred_fraction":float(np.mean((fw<1)|(sw<1))),
        }
    }
