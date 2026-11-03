# Dino SkinMax41 �X Four-View Eevee Dataset Renderer

Branch: `feature/dino-motion-quality-p123`

## Scope

This is an **independent GitHub renderer** for replaying the existing 2,200
Dino source poses with the **PRE-PAINTED SkinMax41 material** and four
orthographic cameras in Eevee. The original Cycles two-view renderer stays
unchanged. The earlier local P3/P4 four-view renderer, its extra visibility
labels and the four-view Reader training loader have **not yet been pushed**.

**The 41-color palette is not recalculated by this script.** It renders
the baked materials/vertex colors already in the following Blender file:

- `runs/dino/skinmax_camera_aware_v1/dino_viewaware_skinmax_v4.blend`
- `assets/dino/rigged/rig_info.json`
- `runs/dino/dataset_hr_v1/*_meta.json`

The script hashes the colorized .blend and verifies it has vertex color
attributes and the correct bone names. It cannot independently confirm
that all 41 palette colors were assigned to the intended body parts.
Visually inspect the smoke images.

## Blender 5.0 API

Blender 5.0 Eevee engine ID: `BLENDER_EEVEE` (not `BLENDER_EEVEE_NEXT`).
Blender 4.2�V4.x uses `BLENDER_EEVEE_NEXT`. Use
`scene.eevee.taa_render_samples`; do NOT set removed GTAO/bloom properties.

Reference: https://developer.blender.org/docs/release_notes/5.0/python_api/

## Smoke test: five poses

Run in PowerShell from `F:\anymove` after pulling the branch:

```powershell
git switch feature/dino-motion-quality-p123
git pull --ff-only

$env:MOCAP_CREATURE = "assets/dino"
$blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
$blend = "runs/dino/skinmax_camera_aware_v1/dino_viewaware_skinmax_v4.blend"
$rig = "assets/dino/rigged/rig_info.json"
$src = "runs/dino/dataset_hr_v1"
$out = "runs/dino/skinmax_camera_aware_v1/dataset_fourview_color41_full_v1_eevee"

& $blender -b -P blender/render_four_view_color41.py -- `
    --blend $blend `
    --rig $rig `
    --source-dataset $src `
    --out $out `
    --indices 1,2,3,25,50 `
    --render-engine eevee `
    --samples 16 `
    --scale 1
```

Before running all poses, inspect color appearance, front/back and
side/opposite mirrors, correct part/depth raster masks, and
`max_fk_vs_source_gt_error_m`. This static replay must agree with
source joints to within 1 mm.

## Complete 2,200 poses and resume

Use exactly the same output directory, Blender version, painted .blend,
source metadata, engine, sample count and scale:

```powershell
& $blender -b -P blender/render_four_view_color41.py -- `
    --blend $blend `
    --rig $rig `
    --source-dataset $src `
    --out $out `
    --indices all `
    --render-engine eevee `
    --samples 16 `
    --scale 1 `
    --resume
```

`render_contract.json` is written first. With `--resume`, an existing
pose is skipped **only if all 8 PNG/NPZ artifacts match SHA-256, image
dimensions, source metadata hash and render contract hash**. An
incomplete pose renders again as a whole. A changed scene/material/
Blender version requires a NEW output directory. Never start two workers
against the same output directory.

## Files per pose

For pose `00001`:

| File | Description |
|---|---|
| `00001_front.png` | RGBA Front (400 �� 720 at Dino scale 1) |
| `00001_side.png` | RGBA Side (880 �� 720) |
| `00001_back.png` | RGBA Back |
| `00001_opposite_side.png` | RGBA Opposite Side |
| `00001_front_labels.npz` | Geometry labels for Front |
| `00001_side_labels.npz` | Geometry labels for Side |
| `00001_back_labels.npz` | Geometry labels for Back |
| `00001_opposite_side_labels.npz` | Geometry labels for Opposite Side |
| `00001_visibility_meta.json` | Source pose SHA, FK check, camera projections and artifact SHA |

Each `*_labels.npz` contains:

- `part_id`: (H,W) int16; triangle's largest summed bone skin weight.
  -1 means background or unknown region.
- `depth_m`: (H,W) float32; camera-axis distance to visible
  mesh surface, infinity on background.
- `joints_uv`: (J,2) float32; projected Blender FK joint positions.
- `joints_depth_m`: (J,) float32; each joint's camera-space depth.

Note: `visibility_meta.json` does **not** represent learned or
validated joint visibility probabilities. Geometric surface Part-ID
does not establish whether an internal anatomical joint is observable.

Total at 2,200 poses: **8,800 PNG + 8,800 NPZ + 2,200 JSON =
19,800 pose files**, plus `render_contract.json`.

## Compatibility and known limitations

- Replays the source `pose` / `root` / `yaw` / `joints_world`.
  No new random pose is generated.
- Original source-motion-blur (approximately 20% of poses) is **not
  reproduced**: every output is a static Eevee still.
- Four cameras are orthographic and fixed. Front/Back and
  Side/Opposite Side are left-right reversed.
- Labels are computed from evaluated mesh geometry and are independent
  of Eevee/Cycles RGB shading, but the Part-ID semantic is
  per-triangle dominant skin bone.
- Renderer uses a Numpy CPU z-buffer. Eevee does not imply fast labels;
  end-to-end performance is **not benchmarked**.
- Does not change existing Cycles datasets or checkpoints.
- The local P4 four-view training pipeline is NOT on this GitHub branch.
  Before using this dataset for Reader training, check that the actual
  loader's expected NPZ keys and surface labeling conventions match.
- These changes were made on GitHub, and the pure contract tests are
  included. **Blender 5.0 smoke rendering has NOT been executed** due
  the user's desktop connection being offline.

## Verification

```powershell
python -m unittest discover -s tests -p test_dino_fourview_eevee_contract.py
(Get-ChildItem $out -Filter "*_visibility_meta.json").Count
(Get-ChildItem $out -Filter "*_front.png").Count
```
