import bpy,json,numpy as np
from collections import Counter
bpy.ops.wm.open_mainfile(filepath=r"F:\anymove\assets\dino\rigged\rigged.blend")
obj=next(x for x in bpy.data.objects if x.type=='MESH')
counts=Counter()
wg={g.index:g.name for g in obj.vertex_groups}
for v in obj.data.vertices:
 groups=sorted(v.groups,key=lambda w:w.weight,reverse=True)
 if groups:counts[wg[groups[0].group]]+=1
for s in ('L','R'):
 print('SIDE',s)
 for stem in ('Calf','Foot','Toe','BackToe01','LeftToe01','LeftToe02','MidToe01','MidToe02','RightToe01','RightToe02'):
  n='CC_Dinosaur_'+s+'_'+stem
  print(n,counts[n])
print('TOTAL',len(obj.data.vertices),'skinnedbonegroups',len(counts),'unweighted',len(obj.data.vertices)-sum(counts.values()))
