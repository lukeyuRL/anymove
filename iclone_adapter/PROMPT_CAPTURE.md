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

## v0.7.6 — Raw DINOv2 heatmap peaks are not probability confidences

The trained joint reader uses a linear head with MSE heatmap targets.
Its `mocap.model.decode` function returns raw peak amplitudes; unlike
calibrated probabilities, valid finite values can fall below 0 or above 1.

The previous iClone adapter incorrectly rejected any `det.npz`
containing such overshoots with "joint confidence outside [0,1]".
The updated adapter accepts finite heatmap peak arrays with correct
Dragon joint names/shapes/24-FPS; it records original minimum/maximum,
mean and clipped-sample counts in `detection_summary`.

The solver now converts raw peaks to bounded [0,1] **weights in
memory only**, without editing the original `det.npz`. The clipping
is not an assertion that the joint coordinates are correct. Nonfinite
values still fail closed, and final 3D reprojection/sync quality gates
are unchanged. Existing `det.npz` files from v0.7.5 can be reused.

**Resume a previous PAID Prompt run without another fal.ai charge**:
use the existing-MP4/NPZ **CAPTURE** button and choose
`runs/capture_dragon_ai_prompt_001/capture/det.npz` if it exists;
otherwise choose `runs/capture_dragon_ai_prompt_001/generated/h3max.mp4`
to repeat only local GPU joint detection and the solver. Choose a NEW
capture output folder, leave the original run unchanged. After solving,
iClone import still checks the original clean Dragon target separately.

## v0.7.7 — Actionable quality review without changing acceptance thresholds

A completed solve can be blocked solely by the post-solve reference
quality gates (for example, `side reprojection 15.2 px > 15 px`).
That is `quality_gate_blocked`, **not a failed solver**. The original
`motion.json`, `det.npz`, and receipts are preserved.

The iClone UI now distinguishes that status from actual capture failures
and displays the saved motion path, measured reprojection and blocking
reasons. The 15 px side / 10 px front X / 0.5 paw agreement thresholds
are **unchanged**. No automatic retarget/write/charge occurs.

Users can deliberately inspect the source, joint overlay (when available)
and the animation in an expendable clean Dragon scene. After review,
`IMPORT AnyMove motion.json` can load the saved file without rerunning
fal.ai, detection or solving, but this manual path does **not** claim
that a quality-failing motion has passed the solver's gates.

## v0.7.8 — Safe existing-Motion replacement and experimental root-flight repair

### Existing Motion no longer needs manual deletion

The direct import panel checks for a compatible Dragon hierarchy if an
existing Motion Clip prevents clean Rest verification. If there is one,
AnyMove offers a **BACKUP AND REPLACE** confirmation (default No).
No old clip is deleted without this explicit approval.

With approval, AnyMove:
1. Saves a full backup of the CURRENT scene to a unique .iProject.
2. Deletes ONLY clips on the selected Dragon Skeleton, checking each count.
3. Evaluates Frame 0 and runs the UNCHANGED strict Rest Pose / clean layer
   tests, on the actor AFTER removing the old clips.
4. Imports the requested motion, performs full per-frame controller readback,
   and saves a separate new .iProject.
5. On any failure after the backup, attempts to reload the full original
   scene from the backup and reports the backup path.

If the original Dragon bone names/parents are incompatible, the operation
is refused before any backup or deletion. The backup remains after success
and can be manually re-opened if needed. This path is available after
source MP4/NPZ Capture and in manual "IMPORT AnyMove motion.json".

### ROOT FLIGHT: inspect / rescale existing Pelvis motion (BACKUP)

Users who have a valid-looking 42-Bone animation but no visible global
lift can use the new ROOT FLIGHT button. Choose the **original**
motion.json that created the CURRENT iClone clip. A read-only preflight
requires exactly one frame-zero clip and checks that every stored Pelvis
Position XYZ controller key still matches the existing source-scale-1
AnyMove trajectory, within 0.002 native values.

The UI reports the source Z displacement (for the sample Dragon,
approximately +1.22 m at Frame 57) and asks the user to enter a multiplier.
A value of 100.0 is the suggested **experimental** unit-scale hypothesis
for the 100x imported FBX, NOT a verified iClone layer convention.
The FBX parent may already apply that scale: blindly multiplying 100x
can overshoot. Use a disposable scene and compare the result visually.
The multiplier is NEVER applied without user input and a second approval.

After approval the repair:
- Saves a complete, new backup iProject first.
- Adjusts ONLY Pelvis Layer Position XYZ on every frame.
- Reads back every adjusted scalar, confirming tolerance <= 0.002.
- Saves a DIFFERENT new iProject and an audit receipt.

FK rotations, original motion.json, det.npz, source video, and older
iProjects are untouched. If writing fails, the backup is reloaded.
Successful controller readback does not prove the evaluated Pelvis world
translation (IK overrides, actor Transform or SDK evaluation can differ).
This feature is a measurable root-scale experiment, not a claim that
Root Motion is calibrated or quality-gated.

The source Dragon MP4 and solver do NOT need to be generated again, so
this process involves no additional fal.ai charges. GitHub CI validates
mock SDK safety and math only; actual iClone 8 playback still needs a
real on-device smoke test.

## v0.7.9 — Why deleting Motion is not a clean Rest Pose (supersedes v0.7.8)

**Critical iClone behavior:** removing an object's animation or deleting
its Motion Clips can leave the original *current pose* in the scene.
The Reallusion iClone 8 manual explicitly warns that removing object
animation does not restore the startup/default pose. iClone's UI
"Reset to Default" adds keys and is not necessarily identical to the
FBX skeletal Rest pose expected by this importer's same-rig profile.

Consequently the v0.7.8 approach "backup current scene, delete clips,
assume Rest is restored, import new Motion" was **not reliable** and is
no longer used by the UI. Merely having zero Clips is insufficient.
The 42-Bone hierarchy and current world matrices must still align
with the verified original FBX (100x) when evaluated.

### Recovery using a genuinely clean iProject

1. If you do not yet have a **CLEAN Dragon iProject** saved from an
   original unanimated 42-Bone Dragon FBX imported at 100x:
   - Open a **NEW iClone scene**, set 24 FPS and Frame 0.
   - Import the original Dragon FBX (100x), keeping it unposed and
     without any preexisting Motion Clips or animated Layer keys.
   - Select Dragon and use "IMPORT AnyMove motion.json" once.
   - The new import now saves, **before writing any animation keys**,
     `AnyMove_Dragon_CLEAN_before_motion.iProject` inside the unique
     output package, alongside the separate animated project
     `AnyMove_Dragon.iProject`.
2. On later attempts in a scene containing an old Motion or a pose
   retained *after* clip deletion, choose "IMPORT AnyMove motion.json".
   If strict Rest validation fails, the UI asks for this independently
   **clean** template .iProject (not an animated output project).
3. Before changing anything, the UI backs up the **WHOLE current scene**
   to `AnyMove_BeforeSceneReplace_XXX.iProject`.
4. With explicit user confirmation, it **REPLACES THE ENTIRE SCENE**
   with the chosen clean template. Any other current-scene objects
   remain only in the backup. No old clip is deleted in place.
5. It locates exactly one 100x rest-compatible Dragon, checks that no
   motion keys are active, re-selects it, runs the original exact
   42-Bone same-rig/clean-scene gates, imports the supplied Motion JSON,
   audits all integer-frame FK and Pelvis Position controls, then
   saves a separate new .iProject.
6. If the template is animated, incompatible, the scene cannot be
   loaded, or the import is incomplete, it reloads the original scene
   backup and reports the original path. If recovery fails, it
   reports the intact backup path for manual recovery.

**Do NOT mistake**
`AnyMove_BeforeMotionReplace_001.iProject` from a failed v0.7.8
attempt for an unposed baseline: it is a backup of the *posed* scene.

An updated Rest diagnostic reports measured bone deviations and
scale/hierarchy details rather than just `needs_mapping_review`.
The strict Rest geometry and safety thresholds were NOT relaxed.
This does not require rerunning fal.ai, DINOv2 or the pose solver.

Runtime caveat: GitHub CI tests the code with mocked RLPy interfaces,
not an actual running iClone installation. The project load/select
workflow needs an on-device smoke test. AnyMove does not promise that
loading a template will preserve the previous scene's props/camera;
the full previous scene is kept in the mandatory backup.

Official reference:
https://manual.reallusion.com/iclone-8/content/enu/8.0/50-animation/motion-layer/Setting_Default_Keys_and_Resetting_Motion_Layer_Keys.htm

## v0.7.10 — Fix iClone 8 RTime clip-origin false failure in ROOT FLIGHT

Root flight preflight previously called `RTime.GetValue()` before inspecting
Pelvis Position, which is the **legacy** iClone time API. iClone 8 instead
uses `RTime.ToInt()` and 6000 ticks per second. On the real iClone 8 Python
host a valid frame-zero Motion Clip was incorrectly rejected with:

```
ContractError: Cannot verify Motion Clip origin at Frame 0
```

The ROOT FLIGHT guard now uses the same `motion_writer._ticks()` converter
as the already working Motion Writer and Scene Guard, checking that
`RIClip.ClipTimeToSceneTime(IndexedFrameTime(0, fps))` maps to frame-zero
scene ticks. An origin shifted away from frame zero is still refused,
and unavailable conversion APIs or unknown time objects still fail safely.

The regression suite now drives the real `_controls()` logic using a
ToInt-only iClone 8 RTime mock (deliberately NO GetValue), checks an
offset clip remains rejected, and covers missing conversion/bad time types.

**This corrects only the preflight method call**. It does not retarget
the Dragon, apply the experimental 100x multiplier, or assert that visible
world-space lift will match source motion. Before modifying any Pelvis
Position keys, the tool still requires exact source-motion controller
readback and a full-scene backup plus user confirmation.
If preflight reports source-key mismatch, stop and inspect the stored keys.
