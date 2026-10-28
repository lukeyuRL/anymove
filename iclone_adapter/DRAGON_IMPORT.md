# AnyMove Dragon → iClone: import audit and mapping

Once a Dragon FBX generated from `assets/dragon/rig.json` is in iClone:

1. Select Dragon in iClone; **Export current-pose skeleton snapshot** (do not
   pose or animate it before taking the snapshot).
2. With *external* Python, compare it to the original AnyMove source Rig
   and produce a conservative direct-name mapping:

   ```powershell
   cd F:\anymove
   .\.venv\Scripts\python.exe -m iclone_adapter.verify_import F:\anymove_test\skeleton.json --report-out F:\anymove_test\dragon_import_report.json --map-out F:\anymove_test\dragon_mapping.json
   ```

   A full PyTorch installation is not needed for this comparison; Python 3.8+
   standard library is sufficient.

3. Check `report.result` is `compatible_static_rest_snapshot`,
   `matched_unique_names=42`, `matched_parent_relations=42` and the
   inferred scale (the observed FBX imported to iClone at `~100x`).
   Only then use `dragon_mapping.json`.

4. The generated mapping covers **41 non-root** source bones and stores
   `calibrated: false`. That is intentional. Static Blender/iClone bone
   matrices may match but the runtime FK Layer XYZ channels still need a
   calibration test before `calibrated: true` can be approved.
   `pelvis` root motion is excluded until a separate writer is implemented.

The comparison uses Blender Bone `matrix_local` in `rig_info.json` (rest
matrix in armature space) against the iClone snapshot's Bone World Matrix
after a single global scale. It does **not** infer Skin Bind Pose from an
arbitrary current snapshot.

On the uploaded `rigged` scene, the iClone JSON had 57 nodes: 42 actual
AnyMove bones, 12 FBX `_end` helpers, plus `rigged`, `body`, `rig`.
All 42 original names/parents matched, and the imported `rig` has a 100x
world scale. Source matrices were compared at 1e-4 quantization when
checking the uploaded file; numerical source-vs-target differences are
negligible at the scale of the model.
