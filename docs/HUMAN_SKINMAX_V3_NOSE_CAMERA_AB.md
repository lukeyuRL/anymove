# AnyMove Human SkinMax53 V3 — Simple Nose + Camera Azimuth A/B

**Date:** 2026-11-03
**Branch:** `feature/iclone-general-camera-angles`
**Status:** nose sculpt and controlled Blender QA PASS; actual Reader/H3 Motion quality not yet measured.

## Inputs and outputs

- Audited source model: `F:\anymove\runs\human\human_skinmax_multicolor_v3.blend`
- Source SHA-256: `93ced4096791fbc84712527a3c023e7c3bee381106e15b898cd4295f24c8008b`
- Mesh-only sculpt output: `F:\anymove\runs\human\human_skinmax_multicolor_v3_nose.blend`
- Output SHA-256: `677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc`
- Independent asset QA: `runs/human/human_skinmax_multicolor_v3_nose_validation.json`
- Nose sculpt audit: `runs/human/human_skinmax_multicolor_v3_nose_audit.json`
- Source V3 is **untouched** and available for a controlled comparison.
- The new binary asset is local-only; it was not placed in the public GitHub repository.

## Model change

A **small stylized nose** was formed by localized Gaussian deformations of the *existing head mesh*, in world -Y (the model's original forward direction). It has a narrow bridge, rounded tip and subtle lower alae. No extra mesh object, separate nose material, new vertices, UV seams or extra rig bones were added.

Independent Blender 5 reopen comparison confirmed:

| Check | Result |
| --- | --- |
| Vertices changed, all belonging to head skin group | 364 |
| Faces affected (indices/topology unchanged) | 413 |
| Maximum world-space displacement | 3.681 cm |
| Total mesh vertices / faces | 30,188 / 30,176 unchanged |
| Bones | 55 unchanged |
| Skin weights / skeleton rest matrices | Identical |
| SkinMax53 vertex-color attribute | Identical |
| Material nodes, UV layers, packed images | Identical |
| Face-normal flips | 0 |
| Saved original SHA-256 | Unchanged |
| Blender 5 replay of V6 human motion GT | PASS, max 0.00028 cm joint error |

## Before / after renders

Using the same Blender Eevee lighting, orthographic camera, color material and head framing, the neutral V3 head was rendered before and after sculpt from **0°, 45°, 90°, and 315°**.

- Close-up contact: `runs/human/camera_angle_p0_v3/nose_preview/human_v3_before_after_nose_contact.png`
- Six-view Reach motion frame 53, nose model: `runs/human/camera_angle_p0_v3/human_v3_nose_camera_angles_contact.png`

Those files are local render outputs; they are **not AI-generated videos**.

## Continue camera-angle comparison with the nose model

Reused the same four selected V6 Reach motion clips and 16 frames as the original V3 mesh experiment. For each color region, up to 80 dominant-skin-group vertices were sampled with the same random seed (20261104). A region counted as exposed if 8% of its sampled visible surface was not occluded by its own mesh.

| Camera azimuth pair | Original V3 two-view color region exposure | Nose V3 exposure | Nose finger-region exposure |
| --- | ---: | ---: | ---: |
| 0°/90° | 35.73% | 35.50% | 31.88% |
| **315°/45°** | **67.57%** | **67.57%** | **69.58%** |
| 45°/135° | 64.15% | 64.15% | 60.83% |

The unchanged hand/finger results demonstrate that the nose deformation did not disturb limb visibility in the sampled poses. The small baseline color-region difference is confined to the head surface sampling/raycast results. **These are mesh-surface visibility proxies, not detected-joint visibility confidence or Reader accuracy.**

The corresponding human GT triangulation benchmark remains unchanged by a mesh-only nose sculpt: with identical 2-pixel Gaussian UV perturbations, the three 90° camera pairs yield **1.577 cm 3D MPJPE** each. The nose may help with head front/back identity in a generated video, but that must be verified by the actual Reader/video workflow.

Nose variant visibility report:
`runs/human/camera_angle_p0_v3/human_v3_nose_surface_visibility_seed20261104_samples80.json`

## Reproduction

With the original audited V3 `.blend` and the target `nose.blend` **absent**, create the sculpt once:

~~~powershell
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b runs/human/human_skinmax_multicolor_v3.blend -P tools/add_human_v3_simple_nose.py
~~~

The sculpt tool intentionally refuses to overwrite an existing output. Verify the existing new model:

~~~powershell
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b runs/human/human_skinmax_multicolor_v3.blend -P tools/verify_human_v3_nose_integrity.py
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b runs/human/human_skinmax_multicolor_v3_nose.blend -P tools/validate_human_motion_replay.py
~~~

To rerun all angle visibility measurements on the nose model:

~~~powershell
$env:ANYMOVE_HUMAN_VARIANT = 'nose'
$env:ANYMOVE_VIS_SEED = '20261104'
$env:ANYMOVE_VIS_SAMPLES_PER_REGION = '80'
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b runs/human/human_skinmax_multicolor_v3_nose.blend -P tools/eval_human_v3_camera_visibility_blender.py
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' -b runs/human/human_skinmax_multicolor_v3_nose.blend -P tools/render_human_v3_camera_comparison.py
python tools/make_human_v3_angle_contact_sheet.py
Remove-Item Env:ANYMOVE_HUMAN_VARIANT,Env:ANYMOVE_VIS_SEED,Env:ANYMOVE_VIS_SAMPLES_PER_REGION
~~~

## Remaining gate

The camera-angle solver work is ready for 315°/45° **controlled image-Reader tests**, but this human project currently uses its V6 human generator/training layout, which is not the same as the original wolf/dragon `mocap.train` data contract. The appropriate next work is to adapt the human V6 image dataset/Reader pipeline to the chosen nose model and camera extrinsics, retrain or condition the Reader, and then evaluate actual held-out 2D joint errors, swap rates, joint visibility and reconstructed 3D motion. Do not compare angles with a Reader trained only for the old camera positions.

No new H3 videos or paid inference jobs were started.
