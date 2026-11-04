"""Export V3 Nose 67-point rig contract for optional AnyMove FK Solver.

Read-only Blender source; exports strict layout, bone matrices, parent graph,
unrestricted provisional Euler bounds, 67 keypoint head/tip mappings and a
three-pose numerical FK alignment probe. NOT production joint limits.
"""
import bpy,json,math,pathlib,hashlib,sys
import numpy as np
from mathutils import Matrix,Vector

ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools import render_human_v3_hr_v2_315_45 as runner
from tools import render_human_v3_fullbody_finger_ab as prior
from tools import human_v3_fullbody_motion_v2 as movement

OUT=ROOT/"runs/human/fullbody_finger_v2_hr_315_45/solver"
SRC=ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend"
SHA=runner.SHA256

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()

def main():
    if pathlib.Path(bpy.data.filepath).resolve()!=SRC.resolve() or sha(SRC)!=SHA:
        raise RuntimeError("V3 Nose model source changed")
    arm=bpy.data.objects["Armature"]
    rest_world=arm.matrix_world.copy()
    scales=rest_world.to_scale()
    print("ARMATURE_WORLD_SCALE",list(scales),"MATRIX",[list(row) for row in rest_world],flush=True)
    if max(scales)-min(scales)>1e-6:
        raise RuntimeError("Nonuniform Armature transform requires explicit FK basis handling")
    order=[b.name for b in arm.pose.bones]
    if len(order)!=55:raise RuntimeError("55-bone humanoid unexpected")
    names=order+movement.finger_tip_names()+movement.toe_tip_names()
    ids={n:i for i,n in enumerate(order)}
    if len(names)!=67 or len(set(names))!=67:raise RuntimeError("67 marker contract violated")
    # Exact global rest transform for each bone, consistent with Blender FK
    rest_byname={}
    for bone in arm.data.bones:
        rest_byname[bone.name]=rest_world @ bone.matrix_local
    kp=[]
    for name in names:
        if name in ids:
            src=name;end="head"
        elif name.endswith("ToeTip"):
            src=name.replace("ToeTip","ToeBase");end="tail"
        else:
            src=name.replace("Tip","3");end="tail"
        if src not in ids:raise RuntimeError("No Blender bone for synthetic tip: "+name)
        bone=arm.data.bones[src]
        restpoint=rest_world @ (bone.head_local if end=="head" else bone.tail_local)
        kp.append({"name":name,"bone":ids[src],"end":end,
                   "rest":[float(x) for x in restpoint]})
    kp_rest={r["name"]:np.asarray(r["rest"]) for r in kp}
    bones=[]
    for pbone in arm.pose.bones:
        bone=arm.data.bones[pbone.name]
        end=rest_world @ bone.tail_local
        candidate=[name for name,pos in kp_rest.items()
                   if name!=pbone.name and np.linalg.norm(pos-np.asarray(end))<.0002]
        # Prefer immediate child-bone head, then own distal fingertip endpoint.
        children=[b.name for b in pbone.children]
        candidate.sort(key=lambda n:(0 if n in children else 1,0 if n.startswith(pbone.name) else 1,n))
        # For an offset/disconnected CC child, its tracked head still gives
        # the parent rotation a usable rest-space vector for IK initialization.
        child_head=[n for n in children if n in kp_rest]
        tail=child_head[0] if child_head else (candidate[0] if candidate else pbone.name)
        record={
            "name":pbone.name,
            "parent":ids[pbone.parent.name] if pbone.parent else -1,
            "matrix_local":[list(map(float,row)) for row in rest_byname[pbone.name]],
            "length":float(bone.length),
            "head_joint":pbone.name,"tail_joint":tail,
            "limits_deg":[[-179.0,179.0],[-179.0,179.0],[-179.0,179.0]],
        }
        bones.append(record)
    layout={
        "H":672,"W_FRONT":672,"W_SIDE":672,
        "M_PER_PX":runner.METRES_PER_PIXEL,
        "GROUND_V":576,
        "FRONT_AZIMUTH_DEG":315.0,
        "SIDE_AZIMUTH_DEG":45.0,
        "Z_CENTER":runner.Z_CENTER,
    }
    info={
       "schema":"anymove.human_v3_nose_hr_v2_rig_solver_contract/1",
       "source_blend_sha256":SHA,"bones":bones,"keypoints":kp,
       "layout":layout,
       "limits_status":"PROVISIONAL_BROAD_EULER_BOUNDS_NOT_VALIDATED_FOR_HUMAN_ANATOMY",
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"rigged").mkdir(parents=True,exist_ok=True)
    rigpath=OUT/"rigged/rig_info.json"
    rigpath.write_text(json.dumps(info,indent=2),encoding="utf-8")
    creature={
       "name":"human_v3_nose_hr_v2",
       "feet":["CC_Base_L_Foot","CC_Base_R_Foot","CC_Base_L_ToeTip","CC_Base_R_ToeTip"],
       "layout":layout,
    }
    (OUT/"creature.json").write_text(json.dumps(creature,indent=2),encoding="utf-8")
    # Exact Blender deformed bone-head/endpoint FK numerical reference poses.
    probes=[]
    base_eul=[tuple(float(v) for v in bpy.data.objects["Armature"].pose.bones[n].rotation_euler) for n in order]
    for frame in (8,43,77):
        prev=prior.pose_changes
        try:
            prior.pose_changes=movement.pose_changes
            prior.apply_pose(arm,rest_world,names,"finger_independent_roll",91954,frame)
        finally:
            prior.pose_changes=prev
        bpy.context.view_layer.update()
        poses=np.asarray([tuple(pb.rotation_euler) for pb in arm.pose.bones],dtype=float)
        target=prior.joints_world(arm,names)
        delta_world=np.asarray(arm.matrix_world.translation-rest_world.translation,dtype=float)
        probes.append({
            "frame":frame,
            "euler_xyz_rad":np.round(poses,12).tolist(),
            "root_translation_m":np.round(delta_world,12).tolist(),
            "joints_world_m":np.round(target,12).tolist(),
        })
    (OUT/"fk_probe.json").write_text(json.dumps({
        "schema":"anymove.human_v3_nose_fk_blender_reference/1",
        "source_blend_sha256":SHA,
        "bone_names":order,"joint_names":names,
        "probes":probes,
    },indent=2),encoding="utf-8")
    print("HRV2_HUMAN_SOLVER_RIG_CONTRACT",json.dumps({
        "bones":len(bones),"markers":len(kp),
        "rig":str(rigpath),
        "probe_frames":[r["frame"] for r in probes],
        "provisional_joint_limits":True,
    }),flush=True)
if __name__=="__main__":
    main()
