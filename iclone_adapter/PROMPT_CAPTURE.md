# Dragon Prompt -> fal video -> AnyMove Motion -> iClone

This optional external Python 3.11 command adds the final upstream step:
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
