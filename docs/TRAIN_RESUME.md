# Reader training: persistent epoch checkpoints and resume

This workflow is available on `iClone_High_Res` and is **optional**. It supports
Dino, Dragon, and any creature using `mocap.train`. `reader_hr_v1.pt` remains a
normal inference-only Reader checkpoint, not a complete training state.

## Runpod A40 + Dino example

Keep the repository and unpacked dataset on fast local storage. Put completed
checkpoints on storage that survives Pod termination. In particular, Runpod
**Global Volumes are object-backed and do not guarantee atomic rename**: they
are not suitable for running a Git repository or thousands of dataset files.
They can hold completed large training checkpoints. `/workspace-global` is the
mount point for the Global Volume in this example.

```bash
cd /workspace/anymove
export MOCAP_CREATURE=assets/dino
export MOCAP_RESUME_DIR=/workspace-global/anymove/dino_v1_resume
export MOCAP_KEEP_CHECKPOINTS=3
bash tools/train.sh dino 20 4 672 192 \
    runs/dino/dataset_hr_v1 runs/dino/reader_hr_v1.pt 0
```

With `MOCAP_RESUME_DIR` set, the trainer:

1. On startup scans `epoch_###_full.pt` from newest to oldest and restores the
   first intact and compatible model, optimizer, AMP scaler, optimizer step,
   and epoch. It keeps the original warmup/cosine schedule position.
2. Finishes each Epoch's normal validation and writes `reader_hr_v1.pt` to local
   disk, then writes a **full** training state to a new numbered persistent
   file. A separate `*.complete.json` marker is written **after** the large
   file is copied and its size checked. On resume, the SHA-256 is checked for
   checkpoints that have completion markers. No remote rename is required.
3. Maintains a separate lowest-mean-Front/Side-error model-only checkpoint:
   `reader_hr_v1_best_epoch_###.pt` (plus its completion marker).
4. Keeps the newest `MOCAP_KEEP_CHECKPOINTS` full-state files, default **3**,
   and the latest best-model file. The dataset and `.pt` files are excluded
   from Git by `.gitignore`.

A successful backup prints `PERSISTENT CHECKPOINT SAVED: epoch N`. A future
run with the **same total Epoch count** and training configuration prints
`RESUMED: completed N epochs` and continues at Epoch N (zero-based index).
Never start two writers against the same checkpoint directory.

## Runpod temporary-patch compatibility

Checkpoints produced by the earlier `RUNPOD_DINO_RESUME_V1` hotfix are named
`epoch_001_full.pt`, etc. They have the same model/optimizer/scaler/step/epoch
fields but no completion marker or annotation fingerprint. The new code can
load them, after checking they are readable and the listed training settings
match. It allows relocating a dataset when the final path components remain
`runs/<creature>/<dataset-name>`; newer checkpoints also compare a SHA-256 of
all JSON metadata labels. **Image bytes are not hashed.**

Avoid `git pull`, `git restore`, or replacing `mocap/train.py` while an existing
training process is running. Python typically imports this module when its
process starts, but changing the working tree mid-run is still unnecessary
risk. Back up the current working copy and wait until the running job exits
before reconciling its local hotfix with the Git version.

## Constraints / recovery

- The resume point is the **end** of an Epoch. Work in a partly completed Epoch
  must be repeated after interruption.
- A truncated/corrupt newest checkpoint is skipped and an older valid one
  tried. A valid checkpoint from the wrong creature, dataset, precision,
  architecture, batch, or total Epoch schedule **fails** instead of silently
  restarting. Changing the total Epoch count changes the LR schedule.
- Random background augmentation uses fresh entropy per sample; even after
  restoring optimizer state, resumed training is **not bitwise identical** to
  uninterrupted training.
- The best-model file is evaluated by mean Front/Side 2D validation error;
  this does not guarantee best 3D motion or iClone quality.
- Full-state files can be much larger than the model-only Reader file. Verify
  the first full checkpoint on the actual persistent mount before depending
  on it. A valid backup marker and a successful read on a new Pod are the
  strongest practical checks; global storage may have eventual consistency.
- If no completed epoch remains (for example after manual cleanup), training
  starts from scratch. Do not delete the dataset or original checkpoints until
  model quality has been verified.

## CPU-only unit tests

```bash
python -m unittest discover -s tests \
    -p 'test_iclone_training_checkpoint.py' -v
```
