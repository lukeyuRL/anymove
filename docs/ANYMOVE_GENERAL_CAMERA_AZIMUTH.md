# AnyMove iClone — Generalized Two-Camera Azimuths (P0)

**Branch:** `feature/iclone-general-camera-angles`
**Base:** `iClone` at `250285e`
**Scope:** calibrated, level orthographic two-view cameras at any well-conditioned horizontal azimuth pair.

## Geometry

The original Solver assumes Front image u = world X and Side image u = -world Y; these equations only work at 0°/90°. The new projector, Blender render positions, 2D training labels, triangulation and Torch FK reprojection loss share a calibrated angle contract.

Azimuth is measured around Blender world Z. At 0°, the camera is at (0,-20,z_center), looking toward +Y. At 90°, the camera is at (-20,0,z_center), looking toward +X. The familiar labels `front` and `side` are two **data slots**, not fixed physical views.

For an azimuth `a` in radians:

~~~text
u = width / 2 + (cos(a) * X - sin(a) * Y) / M_PER_PX
v = height / 2 - (Z - Z_CENTER) / M_PER_PX
~~~

World X/Y are recovered by inverting the 2×2 matrix with rows `[cos(a1), -sin(a1)]` and `[cos(a2), -sin(a2)]`. The matrix condition number must be at most 10; parallel and opposing cameras, including 45°/225°, are rejected. 315°/45° is a well-conditioned pair.

**P0 limits:** level orthographic cameras at common scale only. Perspective, elevation angles, moving cameras and 3+ camera configurations require a separate generalized-extrinsics solution.

## Set the camera angles in PowerShell

From `F:\anymove`, with no editing of rig/model assets:

~~~powershell
$env:MOCAP_CREATURE = 'assets/dragon'
$env:MOCAP_CAMERA_AZIMUTHS = '315,45'
python -c "from mocap import layout; print(layout.FRONT_AZIMUTH_DEG, layout.SIDE_AZIMUTH_DEG)"
~~~

The override applies to Blender, dataset Reader training, detection and Solver under the same environment. To return to creature defaults:

~~~powershell
Remove-Item Env:MOCAP_CAMERA_AZIMUTHS
~~~

The alternative is adding `FRONT_AZIMUTH_DEG` and `SIDE_AZIMUTH_DEG` to the creature.json `layout` (leave its original widths, metres_per_pixel and ground row unchanged).

## Reference and dataset provenance

- Blender reference rendering emits `render_camera_angles.json`.
- Composition verifies that manifest, writes `ref.png` and a `ref.camera.json` sidecar including the image SHA-256.
- Training samples contain `camera_azimuth_deg`; reusing a mismatched dataset directory fails.
- Reader checkpoints store `camera_azimuth_deg`. Detection refuses a checkpoint for the wrong configured camera pair, and writes angles into `det.npz`.
- Solver verifies angles, creature layout and provenance before optimization. Historical untagged 0°/90° data is still supported. Nondefault camera angles require explicit provenance.

Changing Solver angles *alone* is invalid: the Reader must be trained on images of the appropriate views, and the generated video must show those same angles.

## Example new-angle experiment

Use fresh folders and the corresponding rigged Blender file; this is an example workflow, not a request to generate a video:

~~~powershell
$env:MOCAP_CREATURE = 'assets/dragon'
$env:MOCAP_CAMERA_AZIMUTHS = '315,45'
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b -P blender/render_ref.py -- assets/dragon/rigged/rigged.blend runs/dragon/315_45/ref 32
python tools/compose_ref.py runs/dragon/315_45/ref
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b -P blender/render_dataset.py -- assets/dragon/rigged/rigged.blend assets/dragon/rigged/rig_info.json runs/dragon/315_45/dataset 0 100 1 0.6
python -m mocap.train runs/dragon/315_45/dataset assets/dragon/rigged/rig_info.json runs/dragon/315_45/reader.pt 1 2
~~~

The dataset and epoch counts above are illustrative small smoke parameters, not a quality target. No training or AI video generation has been performed.

## Verification

~~~powershell
python -m pytest -q tests/test_iclone_camera_geometry.py tests/test_iclone_solver_camera_layout_guard.py
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b -P tools/qa_camera_angles_blender.py
~~~

The Blender QA compares actual Blender camera matrices with shared 3D-to-2D projection on 0/90, 315/45, 45/135 and 20/100. The next acceptance gate is GT-based A/B/C measurement of 3D MPJPE, both-view joint visibility, joint swaps and temporal jitter.
