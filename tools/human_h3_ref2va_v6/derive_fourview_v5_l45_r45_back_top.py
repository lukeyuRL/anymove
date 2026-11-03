"""Derive NEW four-view scripts from successful V5 Front/±45/Back scripts.

No source file overwrite. New view contract: LEFT(front-left -45deg),
RIGHT(front-right +45deg), BACK(180), TOP(overhead).
"""
import pathlib
root=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\conditioned_backend\fal_h3_ref2va")
def unique(s,a,b):
 if s.count(a)!=1:raise RuntimeError(f"expected one match {s.count(a)}: {a[:160]}")
 return s.replace(a,b)
src=(root/"prepare_fourview_v5_cameras.py").read_text(encoding="utf8")
src=unique(src,'HERE=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"/"fourview_v5"',
           'HERE=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"/"fourview_v5_l45_r45_back_top"')
src=unique(src,'VIEWS=("front","front_right_45","back","front_left_45")',
           'VIEWS=("left_45","right_45","back","top")')
src=unique(src,'ANGLES=(0,45,180,-45) # camera positive angle defined clockwise towards actor-right (-X)',
           'ANGLES=(-45,45,180,"OVERHEAD") # -45 = camera world +X (actor-left), +45 = world -X (actor-right)')
src=unique(src,'    "front":[0,0],\n    "front_right_45":[1,0],\n    "back":[0,1],\n    "front_left_45":[1,1]',
           '    "left_45":[0,0],\n    "right_45":[1,0],\n    "back":[0,1],\n    "top":[1,1]')
src=unique(src,'def project(xyz,cam):',
'''def camera_overhead(original):
 # TOP: camera looking straight down on world XY plane.
 # screen-right=+world X (actor-left), screen-up=+world Y.
 # This camera is right-handed: right X, up Y, camera-back Z.
 front=original["views"]["front"]
 cx=np.asarray(original["rest_bounds"]["center_xyz"],dtype=np.float64)
 fM=np.asarray(front["matrix_world"],dtype=np.float64)
 radius=float(np.linalg.norm(fM[:3,3]-cx))
 M=np.eye(4,dtype=float)
 M[:3,3]=cx+radius*np.array([0.,0.,1.],dtype=float)
 assert abs(np.linalg.det(M[:3,:3])-1.0)<1e-10
 return {**front,"matrix_world":M.tolist(),
         "camera_axis_order":"pixels x right, y down; no rescale",
         "angle_from_front_deg":"TOP_ORTHOGRAPHIC",
         "camera_direction_world":[0.,0.,1.],
         "top_screen_right_world":[1.,0.,0.],
         "top_screen_up_world":[0.,1.,0.]}
def project(xyz,cam):''')
src=unique(src,''' m={
     "front":cameras["views"]["front"],
     "front_right_45":camera_for_front_oblique(cameras,45),
     "back":cameras["views"]["back"],
     "front_left_45":camera_for_front_oblique(cameras,-45)}''',
''' m={
     "left_45":camera_for_front_oblique(cameras,-45),
     "right_45":camera_for_front_oblique(cameras,45),
     "back":cameras["views"]["back"],
     "top":camera_overhead(cameras)}''')
src=unique(src,'"source_body_mesh_unchanged":True,',
           '"source_body_mesh_unchanged":True,\n  "no_front_panel":True,\n  "top_camera_is_true_overhead":True,')
src=unique(src,'"schema":"anymove.fal_h3_fourview_prep/1",',
           '"schema":"anymove.fal_h3_fourview_l45_r45_back_top/1",')
compile(src,str(root/"prepare_fourview_v5_l45_r45_back_top.py"),"exec")
(root/"prepare_fourview_v5_l45_r45_back_top.py").write_text(src,encoding="utf8")

bl=(root/"render_fourview_v5_oblique_blender.py").read_text(encoding="utf8")
bl=unique(bl,'OUT=HERE/"conditioned_backend"/"fal_h3_ref2va"/"fourview_v5"',
          'OUT=HERE/"conditioned_backend"/"fal_h3_ref2va"/"fourview_v5_l45_r45_back_top"')
bl=unique(bl,'for view in ("front_right_45","front_left_45"):',
          'for view in ("left_45","right_45","top"):')
bl=unique(bl,'"views":["front_right_45","front_left_45"],',
          '"views":["left_45","right_45","top"],')
bl=unique(bl,'  "source_rgb_front_back_reused":True,',
          '  "source_rgb_front_back_reused":False,\n  "source_rgb_back_reused":True,\n  "top_overhead_rendered":True,\n  "front_panel_used":False,')
compile(bl,str(root/"render_fourview_v5_l45_r45_back_top_blender.py"),"exec")
(root/"render_fourview_v5_l45_r45_back_top_blender.py").write_text(bl,encoding="utf8")

bat=(root/"render_fourview_v5_batch.py").read_text(encoding="utf8")
bat=unique(bat,'OUT=HERE/"fourview_v5"',
           'OUT=HERE/"fourview_v5_l45_r45_back_top"')
bat=unique(bat,'SCRIPT=HERE/"render_fourview_v5_oblique_blender.py"',
           'SCRIPT=HERE/"render_fourview_v5_l45_r45_back_top_blender.py"')
bat=unique(bat,'for v in ("front_right_45","front_left_45")',
           'for v in ("left_45","right_45","top")')
compile(bat,str(root/"render_fourview_v5_l45_r45_back_top_batch.py"),"exec")
(root/"render_fourview_v5_l45_r45_back_top_batch.py").write_text(bat,encoding="utf8")
print("NEW_SCRIPTS_STAGED_NO_PAIRED_DATA_REWRITTEN")
for f in ("prepare_fourview_v5_l45_r45_back_top.py",
          "render_fourview_v5_l45_r45_back_top_blender.py",
          "render_fourview_v5_l45_r45_back_top_batch.py"):
 print(root/f)
