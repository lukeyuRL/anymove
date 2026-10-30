---
name: anymove-reader
description: Train the joint reader for one anymove creature - render about 2,200 random poses from both layout cameras with every joint's true position, fine-tune DINOv2 (ViT-B/14 with registers) plus a heatmap head on them, check the held-out error, and optionally run the exam (a made-up motion with known joints, read back and scored in centimetres). Use after anymove-rig and before anymove-capture, when a reader misses or swaps joints on real clips, or when planning GPU time for dataset rendering and training.
license: Apache-2.0
compatibility: NVIDIA GPU with CUDA (672-input memory not yet measured; start batch 8 and lower if OOM), Blender 5.x, Python 3.11 venv from requirements.txt. Downloads facebook/dinov2-with-registers-base on first use.
---

# anymove-reader: teach DINOv2 one creature

The reader is trained on the creature itself, so it knows every joint name and every limb colour. Data is free:
Blender poses the rig at random inside its joint limits and records where each joint lands in both views.

## 1. Render the high-res dataset

```bash
tools/dataset.sh <c> 2200 3 500 1.0    # full layout renders -> runs/<c>/dataset_hr
```

This uses an independent dataset directory so the legacy 0.6x renders are not mixed
with the high-resolution 1.0x renders. On Dragon, each view is 640x720, saved as
RGBA. The 1280x720 two-view video layout is **not** changed.
For more detail and full regression steps, see `docs/HIGH_RES.md`.

Half the poses spread uniformly over the limits, half stay near rest; the pelvis sometimes rears; half carry motion
blur from a random velocity (video models smear fast limbs); lights, turn (±30°, sometimes ±70°), position and
airtime are random. Open `runs/<c>/dataset_samples.jpg`: dots must sit on the joints. Interrupted renders resume:
samples that exist are skipped.

GPU budget: prior 12 GB observation was for 448 input and batch 20, NOT 672 input.
Render first and train separately to avoid CUDA OOM; measure the 672/batch 8 peak on target hardware.
Blender uses CUDA (OptiX has been unreliable under WSL).

## 2. Train at 672x672 input / 192x192 heatmap

```bash
tools/train.sh <c> 20 8 672 192   # runs/<c>/reader_hr.pt; held-out error in layout pixels
```

The high-res checkpoint is new and does **not** overwrite the original `reader.pt`.
The training batch defaults to 8 for GPUs with more memory. For 8GB GPUs,
set `MOCAP_LOW_VRAM=1` and pass batch `1` / workers `0` to `tools/train.sh`.
This activates gradient accumulation of 8 and gradient checkpointing; see
`docs/HIGH_RES.md` for the exact Dragon smoke and full-run commands.
The old `reader.pt` is still supported by `mocap.detect` at 448/128.
Timing/memory from old 448 training must not be assumed to hold for 672.

Every render is laid on a fresh random backdrop and degraded like video (framing jitter, colour, blur, downscale,
noise, JPEG) so the reader works on AI clips, not just clean renders. 4% of the samples are held out. The checkpoint
is rewritten after each epoch.

What good looks like (front / side, layout pixels, after 20 epochs): spider 2.9 / 3.6, dragon 6.2 / 5.5, wolf
6.9 / 3.8, demon 8.4 / 9.4. Judge it against the creature's size in the frame: the demon stands 412 px tall, so
8.4 px is about 2% of its height, like the dragon. If the error stalls well above that, check the dataset sheet (dots
off the joints means the layout in creature.json changed after rendering) and the joint limits.

## 3. The exam (optional, about 10 min): known motion in, error in centimetres out

A random smooth motion inside the limits is rendered in the same studio, read back and scored per joint.

```bash
export MOCAP_CREATURE=assets/<c>; R=assets/<c>/rigged; E=runs/<c>/exam
.venv/bin/python -m mocap.exam make $R/rig_info.json $E/truth.json 0 120
blender -b -P blender/apply_motion.py -- $R/rigged.blend $R/rig_info.json $E/truth.json $E/truth --views front,side --paint --blur
.venv/bin/python -m mocap.exam video $E/truth $E/exam.mp4 24
.venv/bin/python -m mocap.detect $E/exam.mp4 runs/<c>/reader_hr.pt runs/<c>/ref/ref.png $E/det.npz --no-calib
.venv/bin/python -m mocap.solve $E/det.npz $R/rig_info.json $E/solved.json
blender -b -P blender/apply_motion.py -- $R/rigged.blend $R/rig_info.json $E/solved.json $E/solved
.venv/bin/python -m mocap.exam score $E/truth/joints.json $E/solved/joints.json
```

The wolf scored 0.94 cm mean 3D joint error on a 1.6 m body (median 0.63 cm, paws 1.5 cm, worst joint 3.4 cm).

## When the reader struggles on real clips

- Joints jump between limbs: the limb colours in the clip drifted from the paint (anymove-film), or two limbs share
  similar colours. Repaint far-apart colours and retrain.
- A pose the clip shows never appeared in training (lying on its side, a full curl): widen that bone's limits in
  rig.json, rebuild the rig, render and train again. Random poses don't cover lying-down; captured motions would.
- Fur, cloth or ears flapping: the wolf's tracked-fur reader added a tip joint per fur bone
  (`blender/add_fur_bones.py`, `blender/export_fur_rig_info.py`) and read swings of 7-28° from the video.
