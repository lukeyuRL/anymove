"""Render close-up Human V3 nose BEFORE/AFTER from calibrated camera azimuths.

Invoked with Blender -b -P script; opens source and target separately.
Read-only relative to both .blend files. Writes side-by-side QA PNGs locally.
"""
import bpy, math, pathlib, json
from mathutils import Vector

ROOT=pathlib.Path("F:/anymove/runs/human")
SCENES={
    "before": ROOT/"human_skinmax_multicolor_v3.blend",
    "nose": ROOT/"human_skinmax_multicolor_v3_nose.blend",
}
OUT=ROOT/"camera_angle_p0_v3/nose_preview"
OUT.mkdir(parents=True, exist_ok=True)
FRONT_TARGET=Vector((0.0,0.0,1.628))
VIEWS=(0,45,90,315)
records=[]
for scene_label, scene_path in SCENES.items():
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    scene=bpy.context.scene
    head=bpy.data.objects.get("Dummy")
    arm=bpy.data.objects.get("Armature")
    if not head or not arm or len(head.data.vertices)!=30188:
        raise RuntimeError("Source Human V3 scene topology changed")
    scene.render.engine="BLENDER_EEVEE"
    scene.eevee.taa_render_samples=48
    scene.render.film_transparent=True
    scene.render.resolution_percentage=100
    scene.render.resolution_x=768
    scene.render.resolution_y=768
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGBA"
    scene.view_settings.view_transform="Standard"
    try:scene.view_settings.look="Medium High Contrast"
    except (TypeError,ValueError):pass
    world=scene.world or bpy.data.worlds.new("HumanNoseQAWorld")
    scene.world=world
    world.use_nodes=True
    background=world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value=(.55,.55,.55,1)
        background.inputs["Strength"].default_value=.85
    for name, rot, power in (
        ("NoseQA_Key",(math.radians(45),0,math.radians(-35)),3.0),
        ("NoseQA_Fill",(math.radians(60),0,math.radians(115)),1.3),
        ("NoseQA_Rim",(math.radians(35),0,math.radians(200)),1.4)):
        sun=bpy.data.objects.new(name,bpy.data.lights.new(name,"SUN"))
        scene.collection.objects.link(sun)
        sun.rotation_euler=rot
        sun.data.energy=power
        sun.data.angle=math.radians(9)
    cam=bpy.data.objects.new("HumanNoseCloseupCamera",bpy.data.cameras.new("HumanNoseCloseupCamera"))
    scene.collection.objects.link(cam)
    cam.data.type="ORTHO"
    cam.data.ortho_scale=.38
    cam.data.clip_end=100
    scene.camera=cam
    for azimuth in VIEWS:
        angle=math.radians(azimuth)
        cam.location=FRONT_TARGET + Vector((-2.5*math.sin(angle),-2.5*math.cos(angle),0.0))
        cam.rotation_euler=(FRONT_TARGET-cam.location).to_track_quat("-Z","Y").to_euler()
        filename=OUT/f"v3_{scene_label}_face_{azimuth:03d}.png"
        scene.render.filepath=str(filename)
        bpy.ops.render.render(write_still=True)
        if not filename.is_file() or filename.stat().st_size<3000:
            raise RuntimeError("Render failed: "+str(filename))
        records.append({"scene":scene_label,"angle":azimuth,"path":str(filename),
                        "bytes":filename.stat().st_size})
        print("NOSE_QA_RENDER",scene_label,azimuth,filename,flush=True)
summary={"status":"PASS","v3_source_untouched":True,"records":records}
(OUT/"render_manifest.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
print("NOSE_RENDER_DONE",json.dumps({"status":"PASS","count":len(records)},indent=2),flush=True)
