"""Independent reopening audit: 107-frame substantial turning with RL_BoneRoot unkeyed."""
import bpy,json,pathlib,sys,numpy as np
ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools.verify_human_v3_root_static_blender import (
    action_paths,world_markers,sha,EXPECTED_SHA,SOURCE
)
BASE=ROOT/"runs/human/fullbody_finger_v2_hr_315_45"
path=BASE/"solver/spine_so3_research/root_static_blender_qa/pivot_turn_left_rootstatic_solver_107frames.blend"
motion_file=BASE/"solver/spine_so3_research/controlled_validation/spine_validation_pivot_turn_left_s405752.root_evidence_so3_600.motion.json"
if pathlib.Path(bpy.data.filepath).resolve()!=path.resolve():
    raise RuntimeError("Opened different Blender turn clip")
data=json.loads(motion_file.read_text(encoding="utf8"))
arm=bpy.data.objects["Armature"]
root=arm.pose.bones["RL_BoneRoot"]
curves=action_paths(arm)
if any("RL_BoneRoot" in s for s in curves):
    raise RuntimeError("RL_BoneRoot has animated FCurves")
if not all(x in curves for x in ("location","rotation_quaternion")):
    raise RuntimeError("No Armature Object global motion channels")
names=data["joint_names"]
expected=np.asarray(data["joints"],dtype=float)
errors=[]
basis_delta=0.
world_q=[]
for i,frame in enumerate(data["source_frame_indices"]):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    basis_delta=max(basis_delta,float(np.max(np.abs(np.asarray(root.matrix_basis)-np.eye(4)))))
    if np.max(np.abs(np.asarray(root.location)))>1e-8:
        raise RuntimeError("RL_BoneRoot local location has changed")
    if np.max(np.abs(np.asarray(root.scale)-1))>1e-8:
        raise RuntimeError("RL_BoneRoot local scale has changed")
    val=world_markers(arm,names)
    errors.append(np.linalg.norm(val-expected[i],axis=-1)*100)
    world_q.append(arm.matrix_world.to_quaternion())
e=np.asarray(errors)
arc=max(q.rotation_difference(world_q[0]).angle for q in world_q)
out={"status":"ROOT_STATIC_TURN_REOPEN_PASS" if e.max()<.03 and basis_delta<1e-6 and arc>.1 else "FAIL",
     "frames":len(e),"fps":bpy.context.scene.render.fps,
     "root_pose_no_keyframes":True,
     "root_pose_matrix_identity_max_deviation":basis_delta,
     "armature_object_motion_rotation_deg":float(np.rad2deg(arc)),
     "joint_replay_max_cm":float(e.max()),
     "joint_replay_mean_cm":float(e.mean()),
     "source_model_unchanged":sha(SOURCE)==EXPECTED_SHA,
     "root_motion_owner":"ArmatureObject","file":str(path)}
path.with_suffix(".audit.json").write_text(json.dumps(out,indent=2),encoding="utf8")
print("TURNING_STATIC_BONE_REOPEN_QA",json.dumps(out),flush=True)
if out["status"]!="ROOT_STATIC_TURN_REOPEN_PASS":
    raise RuntimeError("Static RL_BoneRoot failed nontrivial turn reimport QA")
