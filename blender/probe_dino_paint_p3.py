import bpy,numpy as np,json
from pathlib import Path
p=Path(r'F:\anymove\assets\dino\rigged\rigged.blend')
bpy.ops.wm.open_mainfile(filepath=str(p))
body=next(o for o in bpy.data.objects if o.type=='MESH')
print('BODY',body.name,'attrs',[(a.name,a.domain,a.data_type) for a in body.data.color_attributes])
for a in body.data.color_attributes:
 try:
  n=len(a.data)
  z=np.empty(n*4,dtype=np.float32);a.data.foreach_get('color',z);z=z.reshape(-1,4)
  mask=z[:,3]>.2
  print('ATTR',a.name,'size',n,'colored Alpha>0.2',int(mask.sum()),
  'minmax',np.round(z.min(axis=0),3),np.round(z.max(axis=0),3))
 except Exception as e: print('ATTR_ERROR',a.name,str(e))
for i,m in enumerate(body.data.materials):
 if m is None:continue
 print('MAT',i,m.name,'nodes',m.use_nodes)
 if not m.use_nodes:continue
 for node in m.node_tree.nodes:
  if node.type in ('TEX_IMAGE','VERTEX_COLOR','ATTRIBUTE','MIX','BSDF_PRINCIPLED'):
   print('NODE',node.name,node.type,'layer',getattr(node,'layer_name',''),
         'attribute_name',getattr(node,'attribute_name',''),
         'image',getattr(getattr(node,'image',None),'name',None))
   if node.type=='BSDF_PRINCIPLED':
    col=node.inputs.get('Base Color')
    if col:print('BASE_COLOR_LINK',[(x.from_node.name,x.from_socket.name) for x in col.links])
