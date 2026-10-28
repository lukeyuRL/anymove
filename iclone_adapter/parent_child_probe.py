"""Generate a deterministic Blender/iClone parent-child FK convention A/B test.

A: source Blender XYZ Euler angles directly copied to iClone FK Layer.
B: conjugate source rotation by each bone's *relative-to-parent* rest basis:
       R_B = O_i @ R_source @ inverse(O_i)

Neither candidate is assumed to be correct. Only use on disposable Dragon
scenes. Does not edit Blender, iClone, or the production calibration mapping.

Python 3.8+, standard library only:
    python -m iclone_adapter.parent_child_probe F:/anymove_test/skeleton.json \
        F:/anymove_test/wing_probe --parent wing_arm_L --child wing_fore_L
"""
import argparse
import copy
import json
import math
import pathlib

from .anymove_bridge import (from_euler_xyz, qmul, qinverse_unit,
                             qnormalize, to_euler_xyz)
from .dragon_preview import make_preview
from .motion_contract import validate_motion
from .schema import ContractError, matrix, relative
from .verify_import import DEFAULT_SOURCE

# Explicit peaks, separated by keyframed 0-rotation poses. All intermediate
# frames are linearly interpolated, to match Blender and iClone's FK input.
ANCHORS_DEG = [
    (0,   (0,0,0),  (0,0,0),  "rest"),
    (12,  (25,0,0), (0,0,0),  "parent_only_x"),
    (24,  (0,0,0),  (0,0,0),  "rest"),
    (36,  (0,25,0), (0,0,0),  "parent_only_y"),
    (48,  (0,0,0),  (0,0,0),  "rest"),
    (60,  (0,0,25), (0,0,0),  "parent_only_z"),
    (72,  (0,0,0),  (0,0,0),  "rest"),
    (84,  (0,0,0),  (25,0,0), "child_only_x"),
    (96,  (0,0,0),  (0,0,0),  "rest"),
    (108, (0,0,0),  (0,25,0), "child_only_y"),
    (120, (0,0,0),  (0,0,0),  "rest"),
    (132, (0,0,0),  (0,0,25), "child_only_z"),
    (144, (0,0,0),  (0,0,0),  "rest"),
    (156, (25,0,0), (0,25,0), "parent_x_plus_child_y"),
    (168, (0,0,0),  (0,0,0),  "rest"),
]


def _quat_from_rest_matrix(rest):
    """Quaternion for positive-handed, near-orthogonal 3x3 rest rotation."""
    r = [[float(rest[i*4+j]) for j in range(3)] for i in range(3)]
    # The input is from Blender Bone.matrix_local: SHOULD be pure rotation.
    # Reject shear/mirror/scale rather than constructing an invalid basis.
    for c in range(3):
        length=sum(r[i][c]**2 for i in range(3))
        if abs(length-1.)>0.01:
            raise ContractError("relative Blender Bone rest matrix is scaled")
        for c2 in range(c):
            dot=sum(r[i][c]*r[i][c2] for i in range(3))
            if abs(dot)>0.01:
                raise ContractError("relative Blender Bone rest matrix is sheared")
    det=(r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
         -r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
         +r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if abs(det-1.)>0.01:
        raise ContractError("relative Blender Bone rest matrix is mirrored")
    trace=r[0][0]+r[1][1]+r[2][2]
    if trace>0:
        s=2.*math.sqrt(max(1e-12,trace+1.))
        q=[.25*s,(r[2][1]-r[1][2])/s,(r[0][2]-r[2][0])/s,
           (r[1][0]-r[0][1])/s]
    else:
        i=max(range(3),key=lambda c:r[c][c])
        j=(i+1)%3
        k=(i+2)%3
        s=2.*math.sqrt(max(1e-12,1.+r[i][i]-r[j][j]-r[k][k]))
        v=[0.,0.,0.,0.]
        v[i+1]=.25*s
        v[0]=(r[k][j]-r[j][k])/s
        v[j+1]=(r[j][i]+r[i][j])/s
        v[k+1]=(r[k][i]+r[i][k])/s
        q=v
    return qnormalize(q)


def relative_rest_basis(info, bone_name):
    """Return parent-local Bone rest quaternion, not armature-world basis."""
    bones=info.get("bones")
    if not isinstance(bones,list):
        raise ContractError("invalid source rig_info bones")
    names=[b.get("name") for b in bones]
    if names.count(bone_name)!=1:
        raise ContractError("source rig Bone missing or duplicated: "+bone_name)
    b=bones[names.index(bone_name)]
    p=b.get("parent")
    if type(p) is not int or p<0 or p>=len(bones):
        raise ContractError("probe requires a non-root bone with parent")
    current=matrix([x for row in b["matrix_local"] for x in row])
    parent=matrix([x for row in bones[p]["matrix_local"] for x in row])
    return _quat_from_rest_matrix(relative(parent,current))


def _interpolate(a,b,t):
    return [math.radians(float(x)+(float(y)-float(x))*t)
            for x,y in zip(a,b)]


def control_source_motion(info,parent_name,child_name,fps=24):
    """Create unambiguous, zero-root-motion Euler keyframes for Blender."""
    bones=info.get("bones")
    if not isinstance(bones,list) or len(bones)<3:
        raise ContractError("invalid source rig")
    names=[x["name"] for x in bones]
    if len(set(names))!=len(names):
        raise ContractError("source Bone names duplicated")
    if parent_name==child_name or parent_name not in names or child_name not in names:
        raise ContractError("unknown or identical parent/child bones")
    ip,ic=names.index(parent_name),names.index(child_name)
    if ip==0 or ic==0:
        raise ContractError("do not use pelvis root in non-root FK probe")
    if bones[ic]["parent"]!=ip:
        raise ContractError("%s is not the direct child of %s"%(child_name,parent_name))
    if type(fps) is not int or not 1<=fps<=240:
        raise ContractError("fps must be integer 1..240")
    num=ANCHORS_DEG[-1][0]+1
    euler=[[[0.,0.,0.] for _ in names] for _ in range(num)]
    for idx in range(len(ANCHORS_DEG)-1):
        f0,p0,c0,_=ANCHORS_DEG[idx]
        f1,p1,c1,_=ANCHORS_DEG[idx+1]
        for f in range(f0,f1+1):
            w=(f-f0)/float(f1-f0)
            euler[f][ip]=_interpolate(p0,p1,w)
            euler[f][ic]=_interpolate(c0,c1,w)
    return {
        "fps":float(fps),
        "bones":names,
        "euler":euler,
        "root_quat":[[1.,0.,0.,0.] for _ in range(num)],
        "trans":[[0.,0.,0.] for _ in range(num)],
    }


def make_probe(info,target,parent_name="wing_arm_L",child_name="wing_fore_L",
               fps=24):
    """A/B FK files and independent Blender motion from one controlled sequence."""
    source=control_source_motion(info,parent_name,child_name,fps=fps)
    raw=make_preview(source,target,info,bones=[parent_name,child_name])
    raw["experiment"]="controlled_parent_child"
    raw["candidate"]="A_raw_euler"
    raw["warning"]=(
        "UNVERIFIED FK BASIS TEST A: direct Blender Euler XYZ to iClone FK "
        "Layer channels. Use disposable scene; compare with Blender reference."
    )
    corrected=copy.deepcopy(raw)
    corrected["candidate"]="B_relative_rest_conjugation"
    corrected["calibration_status"]="unverified_rest_basis_conjugation_preview"
    corrected["warning"]=(
        "UNVERIFIED FK BASIS TEST B: source Quaternion conjugated by "
        "each Bone's PARENT-RELATIVE rest basis; not production calibrated."
    )
    byname={b["name"]:b for b in target["bones"]}
    for name in (parent_name,child_name):
        basis=relative_rest_basis(info,name)
        bone_id=byname[name]["id"]
        out=[]
        for angles in raw["tracks"][bone_id]:
            q=from_euler_xyz(angles)
            calibrated=qmul(qmul(basis,q),qinverse_unit(basis))
            result=to_euler_xyz(calibrated)
            # Exact rest stays rest (avoid tiny floating point rounding keys).
            if max(abs(v) for v in angles)<1e-10:
                result=[0.,0.,0.]
            out.append([float(x) for x in result])
        corrected["tracks"][bone_id]=out
    validate_motion(raw,target)
    validate_motion(corrected,target)
    probes=[{
        "frame":frame,"label":label,
        "parent_euler_deg":list(p),
        "child_euler_deg":list(c),
    } for frame,p,c,label in ANCHORS_DEG]
    manifest={
        "experiment":"same_rig_parent_child_FK_order_probe",
        "parent":parent_name,"child":child_name,
        "fps":fps,"frames":len(source["euler"]),
        "candidate_A":"probe_fk_raw.json",
        "candidate_B":"probe_fk_rest_conjugate.json",
        "blender_reference_input":"probe_source_motion.json",
        "reference_root_motion":"zero_for_all_frames",
        "relative_rest_basis_quaternion_wxyz":{
            n:relative_rest_basis(info,n) for n in (parent_name,child_name)},
        "sample_frames":probes,
        "known_limitations":[
            "Only two non-root FK bones are written; all other bones rest",
            "Neither A nor B is certified as iClone FK convention",
            "Parent and Child must be a direct hierarchy pair",
            "Run each candidate in a NEW disposable scene to avoid Motion Layer blending",
            "Blender must load the freshly rendered animation.blend, not rigged.blend"
        ]
    }
    return source,raw,corrected,manifest


def main():
    p=argparse.ArgumentParser(
        description="Generate two Dragon FK conventions plus matching Blender source motion")
    p.add_argument("skeleton", help="Original unposed iClone skeleton.json")
    p.add_argument("output_dir",help="Directory for the three generated JSON files")
    p.add_argument("--source-rig",default=str(DEFAULT_SOURCE))
    p.add_argument("--parent",default="wing_arm_L")
    p.add_argument("--child",default="wing_fore_L")
    p.add_argument("--fps",type=int,default=24)
    args=p.parse_args()
    with open(args.skeleton,encoding="utf-8") as f:
        target=json.load(f)
    with open(args.source_rig,encoding="utf-8") as f:
        info=json.load(f)
    source,raw,corrected,manifest=make_probe(
        info,target,args.parent,args.child,args.fps)
    out=pathlib.Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    for name,data in (
        ("probe_source_motion.json",source),
        ("probe_fk_raw.json",raw),
        ("probe_fk_rest_conjugate.json",corrected),
        ("probe_manifest.json",manifest),
    ):
        with open(out/name,"w",encoding="utf-8") as f:
            json.dump(data,f,indent=2,ensure_ascii=False)
    print("CONTROLLED PARENT/CHILD FK TEST; neither candidate is calibrated")
    print("Parent:",args.parent,"Child:",args.child,
          "Frames:",len(source["euler"]),"FPS:",args.fps)
    for x in manifest["sample_frames"]:
        if x["label"]!="rest":
            print("Frame %d: %s"%(x["frame"],x["label"]))
    print("Blender source:",out/"probe_source_motion.json")
    print("iClone A (raw):",out/"probe_fk_raw.json")
    print("iClone B (rest conjugate):",out/"probe_fk_rest_conjugate.json")
    print("Open each iClone candidate in a separate CLEAN scene.")


if __name__=="__main__":
    main()
