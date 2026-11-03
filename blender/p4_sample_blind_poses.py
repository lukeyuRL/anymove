"""Generate fresh, deterministic 3D pose metadata for untouched P4 blind test.

Creates no images or video, uses existing creature Rig and synthetic pose
sampling distribution with a NEW seed/IDs outside the 2,200 original poses.
The resulting independent GT is used only for evaluation, never training,
checkpoint selection, or solver input.
"""
import argparse,json,math,pathlib,sys
import bpy
import numpy as np
from mathutils import Euler,Vector

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'blender'))
import common,dataset_framing


def min_deformed_z(mesh):
    dg=bpy.context.evaluated_depsgraph_get()
    obj=mesh.evaluated_get(dg)
    evaluated=obj.to_mesh()
    try:
        verts=np.empty(len(evaluated.vertices)*3,dtype=np.float64)
        evaluated.vertices.foreach_get('co',verts)
        verts=verts.reshape(-1,3)
        matrix=np.asarray(mesh.matrix_world,dtype=float)
        world=verts@matrix[:3,:3].T+matrix[:3,3]
        return float(world[:,2].min())
    finally:
        obj.to_mesh_clear()


def run(args):
    path=pathlib.Path(args.rig).resolve()
    blend=pathlib.Path(args.blend).resolve()
    out=pathlib.Path(args.out).resolve()
    if out.exists() and list(out.iterdir()):
        raise FileExistsError('P4 new heldout test must use a fresh empty GT output')
    out.mkdir(parents=True,exist_ok=True)
    info=json.loads(path.read_text(encoding='utf8'))
    creature=json.loads((pathlib.Path(common.CREATURE)/'creature.json').read_text())
    if creature['name'] not in ('dino','dragon'):
        raise ValueError('test creature must match known current Rig')
    for key in ('W_FRONT','W_SIDE','M_PER_PX','GROUND_V'):
        if abs(float(info['layout'][key])-float(common.LAYOUT[key]))>1e-6:
            raise ValueError('P4 test creature layout mismatch '+key)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    rig,body=common.skinned_character_objects()
    names=[x['name'] for x in info['bones']]
    if set(names)!=set(pb.name for pb in rig.pose.bones):
        raise ValueError('armature/bone naming mismatch')
    for pb in rig.pose.bones:pb.rotation_mode='XYZ'
    lim={b['name']:np.deg2rad(np.asarray(b['limits_deg'],float)) for b in info['bones']}
    yaw_max=dataset_framing.validate_yaw_limit(creature.get('dataset',{}).get('yaw_max_deg'))
    framing=dataset_framing.framing_signature(info,common.LAYOUT,yaw_max)
    for i in range(args.start,args.start+args.count):
        rng=np.random.default_rng(args.seed+i)
        rig.animation_data_clear()
        for pb in rig.pose.bones:pb.location=(0,0,0)
        wide=bool(rng.random()<.5)
        pose={}
        for name in names:
            lo,hi=lim[name][:,0],lim[name][:,1]
            if wide:
                value=rng.uniform(lo,hi)
            else:
                value=np.clip(rng.normal(0,.3*(hi-lo)/2),lo,hi)
            pose[name]=np.asarray(value,float)
            rig.pose.bones[name].rotation_euler=Euler(value.tolist(),'XYZ')
        yaw=math.radians(dataset_framing.choose_yaw_degrees(rng,yaw_max))
        root=Vector((float(rng.uniform(-.08,.08)),float(rng.uniform(-.25,.25)),0))
        rig.rotation_mode='XYZ'
        rig.rotation_euler=(0,0,yaw)
        rig.location=root
        bpy.context.view_layer.update()
        root.z=-min_deformed_z(body)+(float(rng.uniform(.05,.45)) if rng.random()<.2 else 0)
        rig.location=root
        bpy.context.view_layer.update()
        mat=rig.matrix_world
        jw=np.asarray([list(mat@(rig.pose.bones[names[k['bone']]].head if
                        k['end']=='head' else rig.pose.bones[names[k['bone']]].tail))
                        for k in info['keypoints']])
        m=common.M_PER_PX
        front=np.stack((common.W_FRONT/2+jw[:,0]/m,
                        common.H/2-(jw[:,2]-common.Z_CENTER)/m),axis=1)
        side=np.stack((common.W_SIDE/2-jw[:,1]/m,
                       common.H/2-(jw[:,2]-common.Z_CENTER)/m),axis=1)
        meta={
            'schema':'anymove.p4_independent_3d_blender_pose/1',
            'source_kind':'new_synthetic_pose_not_previous_2200',
            'joints_world':jw.round(5).tolist(),
            'front_uv':front.round(3).tolist(),
            'side_uv':side.round(3).tolist(),
            'scale':1.0,
            'blur':False,
            'blur_shutter':0.,
            'framing_signature':framing,
            'sample_seed':args.seed+i,
            'yaw':yaw,'root':list(root),
            'pose':{n:np.round(v,5).tolist() for n,v in pose.items()},
            'not_natural_motion':True,'not_model_training':True,
        }
        (out/f'{i:05d}_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
        print('NEW_BLIND_GT_POSE',i,'joints',len(jw),
              'root',np.round(jw[0],3).tolist(),flush=True)
    print('P4_NEW_BLIND_POSE_METADATA_COMPLETE',args.count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--rig',required=True)
    p.add_argument('--blend',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--start',type=int,default=9001)
    p.add_argument('--count',type=int,default=12)
    p.add_argument('--seed',type=int,default=820240)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if a.start<8000 or a.count<1 or a.count>100 or a.seed<2700:
        raise ValueError('P4 novel test ID/seed must be outside original 2200')
    run(a)
