"""Read-only in-app validation of Dragon FK preview against actual bone poses.

No new keys or Motion Clips are written. The iClone playhead is temporarily
moved through sampled frames, then restored.

For each non-root bone we compare:
    inverse(R_rest_local) * R_evaluated_local
versus Rz(source_Z) * Ry(source_Y) * Rx(source_X).
The original Blender solve uses exactly Rz @ Ry @ Rx. This test specifically
checks the same-rig identity-basis hypothesis, not a general retarget.
"""
import math

from .anymove_bridge import from_euler_xyz
from .bone_registry import build_bone_registry
from .motion_contract import validate_motion
from .rlpy_reader import (
    _get_skeleton, _native_matrix, probe_matrix_layout, selected_object,
    snapshot,
)
from .schema import ContractError, skeleton_hash, validate_skeleton


def _transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _normalize(a):
    length = math.sqrt(_dot(a, a))
    if length < 1e-8:
        raise ContractError("bone transform has a degenerate rotation axis")
    return [x / length for x in a]


def _cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]


def _rotation(flat):
    """Orthonormalize 3x3 to remove uniform FBX scale and float noise."""
    if not isinstance(flat, (tuple, list)) or len(flat) != 16:
        raise ContractError("expected canonical 4x4 bone matrix")
    if not all(math.isfinite(float(v)) for v in flat):
        raise ContractError("bone pose matrix contains nonfinite values")
    x = _normalize([float(flat[i*4]) for i in range(3)])
    y0 = [float(flat[i*4+1]) for i in range(3)]
    y = _normalize([a - _dot(y0, x)*b for a,b in zip(y0,x)])
    z = _normalize(_cross(x,y))
    old_z = [float(flat[i*4+2]) for i in range(3)]
    if _dot(z, old_z) < 1e-4:
        raise ContractError("bone 3x3 matrix has reflection or invalid scale")
    return [[x[i],y[i],z[i]] for i in range(3)]


def _from_quat(q):
    w,x,y,z = q
    return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]


def _angle_deg(a,b):
    m = _matmul(_transpose(a),b)
    v = max(-1.,min(1.,(m[0][0]+m[1][1]+m[2][2]-1.)/2.))
    return math.degrees(math.acos(v))


def compare_fk_local_pose(rest_matrix, current_matrix, source_xyz):
    """One bone's orientation error in degrees (post/pre rest multiplication).

    Expected Blender local pose = local_rest @ RzRyRx. A small POST error
    indicates iClone FK is using the same local basis.
    """
    rest = _rotation(rest_matrix)
    current = _rotation(current_matrix)
    desired = _from_quat(from_euler_xyz(source_xyz))
    post = _matmul(_transpose(rest), current)
    pre = _matmul(current, _transpose(rest))
    return {"post_rest_error_deg": _angle_deg(post,desired),
            "pre_rest_error_deg": _angle_deg(pre,desired),
            "actual_delta_deg": _angle_deg(rest,current),
            "requested_delta_deg": _angle_deg([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], desired)}


def _ticks(t):
    return int(t) if isinstance(t,int) else int(t.ToInt())


def _sample_frames(motion, max_peak_bones=4):
    n = motion["frame_count"]
    frames = {0,n//4,n//2,3*n//4,n-1}
    energy = []
    for bone_id, track in motion["tracks"].items():
        amp = max(math.sqrt(sum(x*x for x in pose)) for pose in track)
        peak = max(range(n),key=lambda idx:sum(x*x for x in track[idx]))
        energy.append((amp,peak,bone_id))
    for _,peak,_ in sorted(energy,reverse=True)[:max_peak_bones]:
        frames.add(peak)
    return sorted(frames)


def validate_live_fk_pose(reference, motion, clip_index=None, rlpy=None):
    """Return measurable pose errors for an existing, already-keyed test clip.

    Do not call snapshot() as a rest baseline: it would capture the evaluated
    current pose. Use the separately exported, unposed skeleton.json.
    """
    validate_skeleton(reference)
    validate_motion(motion, reference)
    if motion.get("calibration_status") != "unverified_identity_same_rig_preview":
        raise ContractError("select a generated experimental Dragon FK preview JSON")
    if rlpy is None:
        import RLPy as rlpy
    obj = selected_object(rlpy)
    live = snapshot(obj, rlpy=rlpy)
    if skeleton_hash(reference) != skeleton_hash(live):
        raise ContractError("selected iClone Dragon hierarchy differs from reference")
    if int(rlpy.GetFrameIndex(rlpy.RGlobal.GetTime(),rlpy.RGlobal.GetFps())) < 0:
        raise ContractError("invalid scene time")
    fps = rlpy.RGlobal.GetFps()
    if abs(float(fps.ToDouble()) - float(motion["fps"])) > 1e-4:
        raise ContractError("Project FPS does not match Dragon preview")
    skel = _get_skeleton(obj)
    count = int(skel.GetClipCount())
    if count < 1:
        raise ContractError("no Motion Clip exists on selected Dragon")
    clip_index = count-1 if clip_index is None else int(clip_index)
    if not (0<=clip_index<count):
        raise ContractError("Clip index out of range")
    clip=skel.GetClip(clip_index)
    convert = getattr(clip,"ClipTimeToSceneTime",None)
    if not callable(convert):
        raise ContractError("RIClip.ClipTimeToSceneTime missing")
    entries,_=build_bone_registry(skel.GetSkinBones() or [])
    nodes={entry["id"]:entry["node"] for entry in entries}
    names={entry["id"]:entry["name"] for entry in entries}
    ref={b["id"]:b for b in reference["bones"]}
    for key in motion["tracks"]:
        if key not in nodes or key not in ref:
            raise ContractError("missing live test Bone "+key)
    layout = probe_matrix_layout(rlpy)
    frames = _sample_frames(motion)
    original_time=rlpy.RGlobal.GetTime()
    sample_data={}
    bone_errors={key:[] for key in motion["tracks"]}
    bone_issues={}
    try:
        for frame in frames:
            local=rlpy.IndexedFrameTime(frame,fps)
            scene=convert(local)
            state=rlpy.RGlobal.SetTime(scene)
            if getattr(state,"IsError",lambda:False)():
                raise ContractError("RGlobal.SetTime failed at frame "+str(frame))
            update=getattr(obj,"Update",None)
            if callable(update):
                update()
            item={"frame":frame,"scene_ticks":_ticks(scene),"bones":{}}
            for key,track in motion["tracks"].items():
                node=nodes[key]
                update=getattr(node,"Update",None)
                if callable(update):
                    update()
                actual=_native_matrix(node.LocalTransform(),layout)
                err=compare_fk_local_pose(ref[key]["snapshot_local_matrix"],
                                          actual,track[frame])
                err["name"]=names[key]
                item["bones"][key]=err
                bone_errors[key].append(err)
            sample_data[frame]=item
    finally:
        restored=rlpy.RGlobal.SetTime(original_time)
        if getattr(restored,"IsError",lambda:False)():
            bone_issues["restore_error"]="RGlobal.SetTime failed restoring original time"
        update=getattr(obj,"Update",None)
        if callable(update):
            update()

    results=[]
    for key, errors in bone_errors.items():
        post=[e["post_rest_error_deg"] for e in errors]
        pre=[e["pre_rest_error_deg"] for e in errors]
        requested=[e["requested_delta_deg"] for e in errors]
        observed=[e["actual_delta_deg"] for e in errors]
        # Less than 1 degree of intended movement does not validate axes.
        meaningful=max(requested)>=1.0
        results.append({
            "name":names[key],
            "id":key,
            "meaningful_motion":meaningful,
            "max_requested_angle_deg":max(requested),
            "max_observed_angle_deg":max(observed),
            "mean_post_rest_error_deg":sum(post)/len(post),
            "max_post_rest_error_deg":max(post),
            "mean_pre_rest_error_deg":sum(pre)/len(pre),
            "max_pre_rest_error_deg":max(pre),
        })
    usable=[r for r in results if r["meaningful_motion"]]
    unmatched=[r for r in usable if r["max_post_rest_error_deg"]>10.]
    coherent=[r for r in usable if r["max_post_rest_error_deg"]<=5.]
    verdict=("insufficient_moving_bones" if not usable else
             "consistent_in_sampled_fk_axes" if len(coherent)==len(usable) else
             "fk_axis_or_evaluation_mismatch")
    return {
        "result":verdict,
        "clip_index":clip_index,
        "clip_count":count,
        "sampled_frames":frames,
        "sampled_bone_count":len(results),
        "meaningful_motion_bone_count":len(usable),
        "consistent_bone_count":len(coherent),
        "over_10deg_bones":[r["name"] for r in unmatched],
        "by_bone":sorted(results,key=lambda r:r["max_post_rest_error_deg"],reverse=True),
        "sample_details":list(sample_data.values()),
        "issues":bone_issues,
        "threshold_note":"<=5 deg max is a strict engineering diagnostic threshold, NOT a certified motion-quality standard.",
        "limitations":[
            "Only non-root FK bones present in the selected preview are measured",
            "Pelvis root quaternion/translation from AnyMove are intentionally omitted",
            "A changing bone local matrix does not prove correct skinned mesh deformation",
            "Requires an original UNPOSED skeleton.json snapshot as reference",
            "Does not verify motion interpolation between sampled frames",
        ],
    }
