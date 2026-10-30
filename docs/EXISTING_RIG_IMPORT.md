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
