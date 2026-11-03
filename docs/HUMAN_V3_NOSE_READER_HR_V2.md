# Human SkinMax V3 Nose — Reader HR V2 (672 / 192)

**Branch:** `feature/iclone-general-camera-angles`
**Dataset:** `runs/human/fullbody_finger_v2_hr_315_45/dataset`, audited `PASS`: 5,376 native RGBA images, 192 unique fullbody-finger procedural motion clips, 67 joints.
**Camera slots:** front=315°, side=45° (these are data-slot names, not literal physical front/side views).
**Backbone:** cached local `facebook/dinov2-with-registers-base`, fixed weights; native 672 px images create a 48×48 patch grid.
**Head:** 192×192 heatmaps for all 67 joints, trained from the dataset's native UV labels. This implementation does **not** change the legacy `mocap.model.JointReader` (448 / 128).
**No cloud cost, no H3 generation.**

## Scope and data integrity

- Independent Train/Validation/Test **clip** partitions: 144 / 24 / 24 (4,032 / 672 / 672 camera images).
- Images in Test are never read during training or model selection; lowest Validation 2D joint error selects the checkpoint.
- Backbone uses a frozen DINOv2 model with interpolated positional embeddings at 672×672. Only heatmap neck/refinement/output layers are trained.
- Outputs both native-pixel 2D error and reconstructed 3D MPJPE by calibrated 315°/45° orthographic triangulation.
- The 10 fingertip and 2 toe-tip markers are measured separately from the remaining bone heads.
- Visibility is **not** independently learned here. The V2 dataset's raycast-derived surface visibility annotations are proxy data, not calibrated probabilities. Train-only GT crop hints do not enter Test inference.
- Direct numeric comparison of 672 px image errors to 448 px Reader errors is invalid without scale normalization or comparing 3D metre errors on the **same** motion GT.
- The generated motions are procedural; results do not establish performance on natural human MoCap, H3 synchronized video or final iClone FK motion.

## Training and acceptance

Source scripts:
- `mocap/reader_hr_v2.py`: isolated high-resolution Reader implementation and 192² heatmap decoder.
- `tools/train_human_v3_hr_v2_reader.py`: clip-disjoint training and Validation.
- `tools/eval_human_v3_hr_v2_reader.py`: heldout 3D/2D/finger/toe evaluation (run **after** training).
- `tests/test_human_reader_hr_v2.py`: unit tests with no pretrained network download.

From `F:\anymove` PowerShell:

~~~powershell
python -m pytest -q tests/test_human_reader_hr_v2.py

# Local GPU smoke, separate directory and checkpoint:
python -m tools.train_human_v3_hr_v2_reader --smoke --smoke-steps 4 --epochs 1 --batch 16

# Full native 672/192 Reader training:
python -m tools.train_human_v3_hr_v2_reader --epochs 6 --batch 16

# Only after training and Validation checkpoint selection:
python -m tools.eval_human_v3_hr_v2_reader
~~~

Training outputs:
`runs/human/fullbody_finger_v2_hr_315_45/reader_v2/full/best.pt`, `training_report.json`, and log `../train_full_672_192.log`.

The `best.pt` checkpoint is large (includes local pretrained backbone state) and remains on F:; it is not appropriate for public Git without an explicit asset publishing policy.

## Quality risks

- Some fingertip separations are smaller than the full-body heatmap effective stride (672/192 = 3.5 native pixels). High-resolution alone does not guarantee solved finger identity; evaluate hand crop detectors as a separate ablation.
- Rare ToeBase/ToeTip swaps require the train-calibrated Foot Chain Guard, but only after revalidating its geometry in native 672 px coordinates. Do **not** reuse fixed 448 px pixel thresholds without recalibration.
- Time tracking, temporal jitter guards, confidence calibration, actual 6DoF cameras and iClone FK Solver integration are separate later stages.
- One training seed is a smoke/initial benchmark, not a statistical claim of superiority over Front/Side or other camera angles.
