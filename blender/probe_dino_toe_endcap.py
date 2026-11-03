import bpy,numpy as np,json
from pathlib import Path
bpy.ops.wm.open_mainfile(filepath=r"F:\anymove\assets\dino\rigged\rigged.blend")
arm=next(o for o in bpy.data.objects if o.type=="ARMATURE")
obj=next(o for o in bpy.data.objects if o.type=="MESH")
verts=np.array([list(obj.matrix_world@v.co) for v in obj.data.vertices],float)
info=json.loads(Path(r"F:\anymove\assets\dino\rigged\rig_info.json").read_text())
vg={g.name:g.index for g in obj.vertex_groups}
lookup={g.index:np.zeros(len(verts),float) for g in obj.vertex_groups}
for v in obj.data.vertices:
 for w in v.groups:lookup[w.group][v.index]=w.weight
for side in ("L","R"):
 print("SIDE",side)
 for stem in ("BackToe","LeftToe","MidToe","RightToe"):
  k=f"CC_Dinosaur_{side}_{stem}"
  b1=arm.data.bones[k+"01"];b2=arm.data.bones.get(k+"02");b3=arm.data.bones.get(k+"03")
  ref=b2 or b1
  q0=np.array(list(arm.matrix_world@ref.head_local))
  q1=np.array(list(arm.matrix_world@ref.tail_local))
  ab=q1-q0;ll=np.dot(ab,ab)
  xyz=verts-q0
  t=np.einsum("ni,i->n",xyz,ab)/ll
  d=np.linalg.norm(xyz-np.clip(t,0,1)[:,None]*ab,axis=1)
  group_list=[k+"01",k+"02"]
  candidate=np.array([np.max([lookup[vg[x]][i] if x in vg else 0 for x in group_list]) for i in range(len(verts))])
  idx=np.where(candidate>.2)[0]
  ts=t[idx]; ds=d[idx]
  counts=[]
  for rad in (.04,.07,.10,.15,.20,.25):
   counts.append(int(((t>.60)&(t<2.1)&(d<rad)).sum()))
  print(stem,'groupverts',len(idx),'relative_t_quantiles',np.round(np.quantile(ts,[0,.05,.25,.5,.75,.95,1]),2),
   'distance_quantiles',np.round(np.quantile(ds,[.05,.25,.5,.75,.95]),2),
   'max_toe_surface_near_end',counts,
   'tipnear1',int(((t>.7)&(t<1.8)&(d<.12)&(candidate>.2)).sum()),
   'tipnear2',int(((t>.7)&(t<1.8)&(d<.2)&(candidate>.2)).sum()))
