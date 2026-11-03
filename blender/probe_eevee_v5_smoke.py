import bpy
scene = bpy.context.scene
print("BLENDER", bpy.app.version_string)
print("CURRENT_ENGINE", scene.render.engine)
print("EEVEE_PROPS", [x.identifier for x in scene.eevee.bl_rna.properties])
try:
    scene.render.engine="BLENDER_EEVEE"
    print("EEVEE_ENGINE_ASSIGN_OK",scene.render.engine)
except Exception as exc:
    print("EEVEE_ENGINE_ASSIGN_FAILED",repr(exc))
print("HAS_TAA_RENDER_SAMPLES",hasattr(scene.eevee,"taa_render_samples"))
print("HAS_RENDER_MOTION_BLUR",hasattr(scene.render,"use_motion_blur"))
print("HAS_RENDER_IMAGE_MEDIA_TYPE",hasattr(scene.render.image_settings,"media_type"))
