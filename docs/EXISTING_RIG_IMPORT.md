# Existing Rig Import for the iClone_High_Res branch

Use this tool when the Dinosaur already has an Armature, Mesh and Skin Weight
in a saved Blender file. It is read-only with respect to the original .blend.
Do NOT use tools/rig.sh: it creates a replacement rig and automatic weights.

## Windows PowerShell (from repository root)

    cd F:\anymove
    git switch iClone_High_Res
    git pull --ff-only origin iClone_High_Res
    & "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tools/import_existing_rig.py -- assets/dinosaur/rigged/rigged.blend

For source files outside assets/<name>/rigged/, supply the output directory:

    & "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tools/import_existing_rig.py -- "D:\models\dinosaur.blend" --asset-dir assets/dinosaur

Outputs:
- assets/dinosaur/rigged/rig_info.json
- assets/dinosaur/creature.json
- assets/dinosaur/rigged/rig_import_report.json

Existing JSONs are not overwritten unless --overwrite is passed. Back up
manual edits before overwriting. The original Blender file is never saved.

## Important compatibility requirements

- Exactly one Armature, and at least one Mesh with its Armature modifier.
- Armature Object matrix must be identity; Blender scene unit_scale must be 1.0.
- Current FK, Solver and camera math assume model coordinates are metres,
  forward is -Y, up is +Z, with ground at Z=0. Normalize a COPY if needed.
- Current render_dataset.py expects object names rig (Armature) and body
  (a single skinned Mesh). The exporter does not rename these or merge Meshes.
  If this contract is not met, report says dataset_loader_compatible=false
  until a prepared copy or extended multi-mesh renderer is available.
- Only local active Limit Rotation constraints are interpreted. Remaining
  bone limits are temporary +/-20-degree drafts. Do not train until reviewed.
- Foot semantics, limb paint, creature description and colour-code prompts
  need review; these cannot be inferred reliably from generic bone names.
- The exporter does not guarantee correct Solver root orientation or final
  iClone FK conversion for new rigs. Compare a few samples and their joints.

The High Res branch uses higher-resolution training images and reader inputs;
the existing front/side two-view camera contract is unchanged.

## Tests, then smoke dataset

    py -3.14 -m unittest discover -s tests -p "test_iclone_import_existing_rig.py" -v

After verifying the JSONs, rig orientation, body/rig names, and limits,
from Git Bash run:

    bash tools/dataset.sh dinosaur 4 1 500 1.0 runs/dinosaur/dataset_smoke


## FBX Armature Object transform is not identity

This is common when importing FBX (Object rotation, scale or location).
It does not imply bad Skin Weights. Do not Apply Transform directly on the
original character.

Use the new opt-in normalization mode:

    & "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" -b -P tools/import_existing_rig.py -- assets/dino/rigged/rigged.blend --prepare-copy

It saves a completely separate asset:

    assets/dino_prepared/
      creature.json
      rigged/
        rigged.blend
        rig_info.json
        rig_import_report.json

The original assets/dino/rigged/rigged.blend is never saved or overwritten.
The prepared Blender file bakes the *uniform positive* Armature Object
rotation, scale and translation into the rest bones; world transforms and
scene units are baked into skinned Mesh vertices. Existing Vertex Groups
and Skin Weights remain intact, and sample Rest vertices plus all Bone
head/tail positions are checked against the original world coordinates.

The tool refuses to bake a nonuniform, sheared, mirrored or zero-scale
Armature, and refuses an already posed skeleton. Such cases need manual
diagnosis on a COPY before trying again. A multi-Mesh character produces
JSON drafts but the current one-Mesh dataset renderer still needs changes.

This validates REST coordinates, not the full animation skin deformation.
Compare the prepared mesh appearance to the original, and check the
first four training samples plus labels before full training.

    bash tools/dataset.sh dino_prepared 4 1 500 1.0 runs/dino_prepared/dataset_smoke


## Imported FBX Object names and near-coincident Bone endpoints

The runtime Blender renderers now discover exactly one skinned Armature
and one Mesh from the Armature modifier, regardless of their Object
names (for example Armature / Dinosaur_Geo). This is supported in
render_dataset.py, render_ref.py and apply_motion.py. The animation
writer also accepts a character without the old legpaint colour attribute.
Multi-Mesh rendering is still unsupported.

The importer merges near-identical Bone endpoints using a 0.00001-metre
Euclidean Rest Position threshold. This avoids duplicate reader heatmaps
caused by float32 FBX rest coordinates. Re-export rig_info.json before
training. Mixing pre-fix joint metadata with a reader.pt trained on the
new metadata is invalid.

The report's current_dataset_loader_compatible value now tests whether
there is exactly one skinned Mesh, and does not require literal names
or a perfect floor position. Those remain separate diagnostics.

To re-export an identity-transform original .blend:

    blender -b -P tools/import_existing_rig.py -- assets/dino/rigged/rigged.blend --overwrite

If Armature Object has non-identity transform, use the separate
--prepare-copy workflow above. Do not modify the original asset.
