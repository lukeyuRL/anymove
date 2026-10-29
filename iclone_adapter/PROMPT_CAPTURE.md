# Dragon Prompt -> fal video -> AnyMove Motion -> iClone

This optional external Python 3.11 or 3.14 command adds the final upstream step:
turn an action text into a two-camera Dragon video with fal.ai using
the original tools/make_prompt.py and mocap.generate utilities.

IMPORTANT: this sends a billable request to an EXTERNAL service.
No requests are made without explicit opt-in from the user.

The module writes:
- prompt.txt (exact Dragon two-camera prompt)
- generated/h3max.mp4 (raw AI video)
- generated/h3max.json (fal request metadata)
- capture/det.npz, capture/overlay.mp4, capture/motion.json
- capture/capture_receipt.json and root capture_receipt.json

The existing video_capture worker validates the generated motion's
source FPS is precisely 24 and checks 2-view sync and reprojection
quality, so a malformed or out-of-sync AI clip is not automatically
imported into iClone. The actual GPU inference requires the trained
runs/dragon/reader.pt and the original ref.png.

To invoke explicitly outside iClone:

    .venv\Scripts\python.exe -m iclone_adapter.prompt_capture ^
        --action "the dragon raises its wings and roars, then settles." ^
        --reader runs\dragon\reader.pt ^
        --reference runs\dragon\ref\ref.png ^
        --output F:\anymove_test\prompt_capture_001 ^
        --model h3max --seconds 5 --seed 1

FAL_KEY must be configured in the environment or repository .env.
API prices depend on vendor/model and may change; review them before
generating. This user-confirmed operation is different from using a
previously downloaded, no-extra-charge MP4.

For new characters, model creation, rigging and training a DINOv2
reader are still per-character prerequisites. A known Dragon pretrained
Reader is downloadable from the upstream examples-v1 release (use
tools/fetch_examples.sh dragon --readers --clips on Git Bash), or
can be trained using the upstream dataset/train steps.


## v0.7.0 — Optional explicit user-approved AI prompt to native iProject

Added a SEPARATE opt-in button in the iClone panel:

GENERATE Dragon video from PROMPT (fal.ai PAID) -> iClone

The user enters an English motion description and explicitly confirms
a billable fal.ai MiniMax H3 Max 5-second video generation. The UI
checks FAL_KEY in the configured environment/.env, selects the existing
Dragon-trained reader.pt and reference ref.png, and launches an EXTERNAL
Python 3.11 or 3.14 QProcess (never heavy dependencies inside iClone).

The original AnyMove scripts build the correct two-camera prompt,
submit the user-approved fal video request, download the MP4, run
mocap.detect for both views, run mocap.solve, validate 24 FPS and
the Dragon motion-quality gates. It then uses the same clean-scene
reverification, explicit scene-write approval, full 42-Bone per-frame
FK + Pelvis Position, all-frame SDK readback, and native iProject save.

No video API charge is made unless the user clicked this specific
paid option and approved the additional confirmation dialog.
The existing MP4/NPZ button never contacts fal.ai.

One-time setup for this known Dragon remains:
- 42-Bone original Dragon imported in iClone at 100x FBX scene scale;
- Dragon-trained reader.pt (pretrained release asset or self-trained);
- dragon ref.png; NVIDIA CUDA and external Python 3.11 or 3.14 dependencies;
- FAL_KEY only when using the PAID Prompt button.

Costs and remote model availability are NOT guaranteed. This and
the MP4-to-iProject UI need actual on-device smoke tests; all
repository CI tests are software/mock tests.

## v0.7.3 — Defer iClone target validation until actual import

Generate Prompt and Capture MP4/NPZ no longer inspect the selected iClone
actor or require clean 42-Bone Rest geometry up front. Video capture and
motion solving are actor-independent: they can run with an already-animated
Dragon, an incompatible actor, or no actor selected.

The fal.ai charge still requires an available FAL_KEY and an explicit paid
request confirmation. The trained Dragon Reader, reference image, external
Python 3.11 or 3.14, output folder and capture-quality gates remain mandatory.

When motion.json is ready, the UI runs the unchanged strict direct-import
preflight on the **then-selected** actor. If import preparation fails (wrong
rest geometry, existing clips, nonzero playhead, wrong FPS, etc.), it displays
"Dragon motion ready - import deferred", preserves the solved motion.json and
does NOT write animation. To complete import, open a clean original Dragon
scene and run "IMPORT AnyMove motion.json -> iClone + iProject".

Previously keyed scenes must not be reused for a second import. GitHub CI
covers mock UI/contract tests; real iClone and paid fal.ai runs have not
been performed as part of this patch.

## v0.7.4 — Diagnose early external Python failures before a paid request

The iClone UI performs a lightweight, non-billable external Python check
before the fal.ai paid confirmation dialog or any local capture starts:
- verifies the selected interpreter actually reports Python 3.11 or 3.14;
- checks that the required top-level Python packages are installed;
- identifies the exact executable and missing packages in the error.

This catches a stale or incompatible `.venv` before a charge. The preflight
does NOT confirm a functioning CUDA driver, import binary extensions,
download model weights, or contact fal.ai. It is not a full runtime test.

The UI now also retains a bounded tail of the worker's combined stdout/stderr.
If the process exits before `capture_receipt.json` exists, the popup
includes the exit code, Python interpreter path and recent worker output,
rather than only "no receipt / external worker exited".

**Avoid duplicate charges:** inspect an existing capture directory for
`generated/h3max.mp4` and fal.ai request records before launching a new
paid Prompt job. Never overwrite a previous capture folder.

## v0.7.5 — Python 3.14 external worker support

iClone's embedded Python remains unchanged. The external Prompt, MP4,
det.npz and Solver workers now allow CPython **3.11 or 3.14**.
The UI version preflight and both standalone worker entrypoints share
the same allowlist. Interpreter checks happen before billable fal.ai
generation, and offline tests now include Python 3.14.

On Windows, use an existing Python 3.14 virtual environment at
`F:\anymove\.venv\Scripts\python.exe` (or your clone root). Verify it:

```powershell
cd F:\anymove
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch, cv2, numpy, transformers, requests, PIL; print('Torch:',torch.__version__,'CUDA:',torch.cuda.is_available())"
```

If the virtual environment is absent, create it using
`py -3.14 -m venv .venv` first. Existing virtual environments use
the Python version with which they were created; changing the Windows
default Python does not migrate them.

The lightweight preflight verifies Python and installed module names;
it does NOT import CUDA, prove a working GPU driver, or guarantee
successful DINOv2 inference. No remote fal.ai request was made to test
this change. Review previous capture folders before repeating a paid job.
