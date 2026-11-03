# AnyMove P1 �X Four-View Blender Surface Visibility Ground Truth

**Implementation date:** 2026-11-01  
**Status:** Four-view static-pose Ground Truth Generator **PASS** on Dino and Dragon; **NO** N-view Video Reader / Motion Solver yet.  
**Branch:** `feature/dino-motion-quality-p123` on `F:\\anymove`. Local files only; not committed or pushed.

## 1. Delivered scope

The original 2-view Reader dataset is untouched. The new Blender script replays **existing, independently stored joint poses** from `*_meta.json` and outputs four synchronized views **of the identical static 3D pose**:

| View ID | Camera position relative to Creature | Relationship |
|---|---|---|
| `front` | Along -Y, looking +Y | Existing `cam_front`; matches old Front UVs |
| `side` | Along -X, looking +X | Existing `cam_side`; matches old Side UVs |
| `back` | Along +Y, looking -Y | Geometric 180�X opposite of Front |
| `opposite_side` | Along +X, looking -X | Geometric 180�X opposite of Side |

All cameras are orthographic, share the same world-to-pixel scale and vertical orientation, and use a saved world-to-camera matrix. The scripts validate exact mirror geometry rather than assuming camera labels imply symmetry.

For each pose and each view:

- One optional neutral Blender **still PNG**, no video.
- One `*_labels.npz`: camera depth in metres (`depth_m` float32, NaN outside mesh); `part_id` (int16, -1 background, -2 unlabeled but opaque surface, 0..B-1 bone-owned surface part), per-bone amodal and visible area counts, per-bone surface visible fraction, per-keypoint image UV and surface-part visibility label.
- One pose `*_visibility_meta.json` with per-view camera extrinsics, exact root/pose source SHA256, Blender source asset SHA256, validation statistics and notes.
- `four_view_audit_v2.json` and static `*_four_view_contact.png` preview created by independent Python audit.

### Correct visibility semantics

**This is geometrically derived, known-rendered Mesh SURFACE PART visibility, not direct visibility of a buried bone/joint center.**

After Skinning, each triangle is assigned the bone with the largest *average vertex skinning weight*. A triangle with weak or ambiguous ownership receives ID -2. Such triangles **still occlude** farther parts in the z-buffer.

For each camera and part:

```text
amodal_pixel_area = number of screen pixels covered by part triangles,
                    ignoring whether another part is in front
visible_pixel_area = number of pixels where this part wins nearest-surface Z
surface_visibility_ratio = visible_pixel_area / amodal_pixel_area
```

For each Keypoint whose owner bone is mapped to a reliable surface part:

```text
1  = some significant area visible (>=2.5% visible/amodal)
0  = part projects with >=12 pixels, but visible ratio <2.5%
-1 = insufficient projected part area, no assigned surface,
     or otherwise UNKNOWN
```

Thresholds are experimental label conventions; 0/1 is **not yet a calibrated probability** and must NOT be advertised as individual bone-center ray visibility. This avoids the fundamental error of casting a ray to an internal skeleton joint and declaring everything hidden.

The image pixel z-buffer uses the evaluated deformed skinned mesh (including Armature deformation), and the independent Blender `scene.ray_cast` tests check nearest-surface depth on 80 interior pixels per view. Non-part-assigned triangles still win the visibility depth test when physically in front.

## 2. Source poses and smoke outputs

The new renderer did **not** generate 2,200 new poses. It replays stored pose `00000` and `00001` for each Creature, starting from:

- Dino: `assets/dino/rigged/rigged.blend`, `runs/dino/dataset_hr_v1/` (67 bones, 92 keypoints).
- Dragon: `assets/dragon/rigged/rigged.blend`, `runs/dragon/dataset_hr/` (42 bones, 50 keypoints).

Validation outputs:

```text
F:\\anymove\\runs\\dino\\four_view_visibility_p1_validation\\
F:\\anymove\\runs\\dragon\\four_view_visibility_p1_validation\\
```

Each contains **2 poses �� 4 views = 8 still renders + 8 Depth/Part-ID NPZ**, 2 per-pose JSON manifests, 2 static composite previews, and the independent audit JSON.

No new H3 video, no motion sequence and no Reader model checkpoint were generated. Ground Truth silhouettes are sharp static meshes even if the original Reader-training image was motion-blurred. The pose is replayed from the *stored* `pose`, `root`, and `yaw`, with FK compared to the original `joints_world` metadata.

### Independent verification

| Check | Dino | Dragon |
|---|---|---|
| Poses / four-view renders | 2 / 8 | 2 / 8 |
| Exact original Front/Side camera contract | PASS | PASS |
| Opposite camera geometry mirror | PASS | PASS |
| Original 3D pose replay | error below 0.001 m | error below 0.001 m |
| 3D source GT �� stored 2D UV reproject | max below 0.1 px | max below 0.1 px |
| Workbench silhouette vs depth mask IoU | about 0.94�V0.96 | about 0.94�V0.96 |
| Sampled Blender independent Raycast depth P95 | 0.000002�V0.000004 m | 0.000002�V0.000004 m |
| Surface-part label/count integrity and source hashes | PASS | PASS |

### What the extra two views actually add (in these two poses)

| Pose | Creature | Keypoints visibly represented in Front/Side | Any of four views | Extra |
|---|---|---:|---:|---:|
| 00000 | Dino | 75 | 75 | 0 |
| 00001 | Dino | 75 | 78 | **3** |
| 00000 | Dragon | 50 | 50 | 0 |
| 00001 | Dragon | 49 | 50 | **1** |

**Interpretation:** On these static poses, extra views improve *surface-part observation coverage* for a few landmarks, not an established improvement in 3D Motion Capture. Some Dino joints map to bone regions without enough labeled skin surface: keep them UNKNOWN rather than invent visibility.

For Dino sample 00000: `23980` evaluated triangles, `22890` confidently bone-assigned, `1090` ambiguous, 54 surface-labeled bones, and 14 unknown keypoints in each view. Unlabeled faces still participate in depth occlusion. Relative proportions depend on pose, angle and resolution.

## 3. How to reproduce (PowerShell, from F:\\anymove)

Run a **new empty** output directory. The renderer refuses nonempty output to protect earlier datasets.

```powershell
cd F:\\anymove
$BLENDER = 'C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe'

$env:MOCAP_CREATURE = 'assets/dino'
& $BLENDER -b -P blender/render_four_view_visibility.py -- `
  --blend assets/dino/rigged/rigged.blend `
  --rig assets/dino/rigged/rig_info.json `
  --source-dataset runs/dino/dataset_hr_v1 `
  --indices 0,1 --scale 0.5 --render-stills `
  --out runs/dino/four_view_visibility_new_run

python tools/audit_four_view_visibility.py `
  --out runs/dino/four_view_visibility_new_run `
  --rig assets/dino/rigged/rig_info.json `
  --source-dataset runs/dino/dataset_hr_v1 `
  --indices 0,1 --create-contact-sheets

$env:MOCAP_CREATURE = 'assets/dragon'
& $BLENDER -b -P blender/render_four_view_visibility.py -- `
  --blend assets/dragon/rigged/rigged.blend `
  --rig assets/dragon/rigged/rig_info.json `
  --source-dataset runs/dragon/dataset_hr `
  --indices 0,1 --scale 0.5 --render-stills `
  --out runs/dragon/four_view_visibility_new_run

python tools/audit_four_view_visibility.py `
  --out runs/dragon/four_view_visibility_new_run `
  --rig assets/dragon/rigged/rig_info.json `
  --source-dataset runs/dragon/dataset_hr `
  --indices 0,1 --create-contact-sheets
```

For a larger training-resolution dataset, use `--scale 1` and a **new dataset folder**. The smoke resolution `0.5` is for validation and NOT a replacement for the existing high-res Reader training data.

Tests:

```powershell
python -m unittest discover -s tests -p "test_iclone_surface_visibility.py" -v
python -m unittest discover -s tests -p "test_iclone_four_view_visibility_artifacts.py" -v
```

Both groups PASS (7 + 3 tests). The artifact tests verify the local 4-view samples and reject tampered Part-ID or mismatched original source SHA256.

## 4. Implementation files

- `mocap/surface_visibility.py`: pure NumPy orthographic z-buffer and surface-part visibility labeling, no species names, reusable tests.
- `blender/render_four_view_visibility.py`: standalone Blender 5.0 script, replay existing true 3D pose, evaluated mesh skinning, four cameras, stills, depth/part labels, independent Raycast QA.
- `tools/audit_four_view_visibility.py`: independent artifact audit, reprojection consistency, source hashes, label/area invariants, static four-view contact-sheet preview.
- `tests/test_iclone_surface_visibility.py`: synthetic near/far raster, unknown part, triangle order, rest feasibility.
- `tests/test_iclone_four_view_visibility_artifacts.py`: real Dino/Dragon smoke, fail-closed tampering and provenance checks.

One helper `blender/probe_four_view_source.py` was used to inspect source pose replay and the skinned mesh; it is not required for production.

## 5. Limits and next step

- **No synchronized 4-view video** and **no 4-view Reader/Solver** exists yet. This P1 is ground-truth preparation. Original `mocap.detect`, `mocap.solve`, `motion.json`, the iClone adapter, and source 2-view dataset all remain unchanged.
- **No optical flow / motion prior** is trained or validated by these independent still poses.
- This version assumes one Armature, one skinned Mesh, four orthographic cameras sharing pixel scale, and each surface triangle mapped to the strongest Skinning bone. Multi-mesh rigs and perspective cameras require an explicit extension.
- Some internal or extremely thin bone regions lack an identifiable surface: UNKNOWN is the correct label; do not falsely present them as visible/occluded.
- The 2-view/4-view coverage comparison must be repeated on a statistically independent set of poses and ultimately synchronized continuous animation clips before claiming four views improve motion capture.
- Shadowing, blur and AI video inconsistency are distinct from geometry-based surface occlusion.

**Next engineering milestone P2:** a view-indexed observation contract with independent `CameraWorldToView`, `UV`, `Confidence`, `PartVisibility` per view, and an N-Camera FK Solver. Validate 2-view preservation and 4-view accuracy **using the existing Blender GT**, before requesting new AI-generated videos.

## P1/P3 appearance correction (2026-11-01)

The first four-view P1 visualisations deliberately used `BLENDER_WORKBENCH` with a SINGLE gray colour for mesh/depth auditing. This accidentally became the source of P3 Reader-training images. Dino's blue/red leg and toe identification originates in source material textures, not the generic Part-ID raster or a `legpaint` vertex attribute (which is absent on the imported Dino).

**Fixed:** `blender/render_four_view_visibility.py` now defaults to `--appearance source-material`, rendering original materials via Cycles. `--appearance neutral-geometry` remains explicitly available for gray geometric debugging. Newly rendered material-coloured 0.5 and 1.0-scale samples under `runs/dino/four_view_material_fix_*` pass the independent Depth/Mask/Camera audit. Render appearance is recorded in pose metadata and P3 training now requires `source-material` unless a legacy override is given.

The 13 Dino bones without Skinning Vertex Groups (including 8 terminal toe bones) still have no independent Surface Part-ID. Surface visibility labels for these parts must remain unknown rather than inventing mesh ownership.

**Do not reuse gray P3 smoke results as an evaluation of textured four-view Reader training.** A fair rerun must use a fresh, source-material dataset and pose-disjoint validation.
