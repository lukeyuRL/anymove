"""Fal MiniMax H3 Max single static-quad reference I2V test.

This file NEVER sends paid generation without explicit --approve-one-video flag.
External video submission requires a separate user approval in the chat.
The image is a single 2x2 PNG, NOT a pre-rendered motion clip.

For later approved execution only:
    python -m pip install fal-client
    python tools/submit_dino_h3max_quad_once.py --approve-one-video
Uses the local authenticated FAL_KEY; credentials are never printed/saved.
"""
import argparse,json,os,hashlib
from pathlib import Path

MODEL="minimax/h3-max/image-to-video"
ROOT=Path(r"F:\anymove\runs\dino\fal_h3max_multiview_v1")
IMAGE=ROOT/"dino_skinmax41_RESET_fourview_reference_2048x1152.png"
RESET_MANIFEST=ROOT/"reset_pose_reference_manifest.json"
PROMPT=(
"A single fixed 2-by-2 multi-camera video of ONE vividly color-coded dinosaur, "
"simultaneously seen from four orthographic viewpoints. The upper-left panel is FRONT, "
"upper-right is SIDE, lower-left is BACK, lower-right is OPPOSITE SIDE. "
"These are synchronized views of the SAME dinosaur at the SAME moment, not four different dinosaurs. "
"The first frame must show the exact neutral RESET POSE from the supplied image, with all four camera views precisely aligned. "
"Keep the four panel borders, panel sizes, framing, gray studio backgrounds and static camera positions perfectly fixed. "
"The dinosaur performs ONE slow, small stepping motion in place, with a gentle single tail sway "
"and subtle natural forearm adjustment. All four panels show the EXACT same motion synchronized frame by frame. "
"Every bright color patch stays rigidly assigned to its original dinosaur bone, toe, arm, "
"leg and tail segment; do not swap or blend color identities between segments or across views. "
"Do not morph anatomy, add or remove limbs, change the camera, pan, zoom, rotate the view, "
"cut shots, switch panels, show captions or introduce other objects."
)
SETTINGS={
    "prompt":PROMPT,
    "duration":5,
    "resolution":"768P",
    "seed":1,
    "prompt_expansion_mode":"disabled",
    "enable_safety_checker":True
}
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--approve-one-video",action="store_true",
                        help="Explicitly authorize exactly one paid H3 Max video submission. Never pass without chat confirmation.")
    args=parser.parse_args()
    if not IMAGE.exists() or IMAGE.stat().st_size<1000:
        raise FileNotFoundError("static 2x2 quad Reference not found")
    if not RESET_MANIFEST.is_file():
        raise ValueError('H3 input must be an audited exact Reset Pose, not a randomly posed image')
    reset=json.loads(RESET_MANIFEST.read_text(encoding='utf8'))
    if (reset.get('schema')!='anymove.fal_h3max_dino_RESET_quad_reference/1' or
            not reset.get('reset_pose_is_identity') or
            not reset.get('bone_euler_rotations_all_zero') or
            reset.get('bone_count')!=67 or
            reset.get('clean_sha256')!=sha(IMAGE)):
        raise ValueError('fal H3 reset-pose image manifest is stale or inconsistent')
    if not args.approve_one_video:
        print(json.dumps({"status":"PREFLIGHT_ONLY_NO_FAL_UPLOAD_OR_VIDEO_SUBMISSION",
                          "model":MODEL,"reference":str(IMAGE),"image_sha256":sha(IMAGE),
                          "duration":5,"resolution":"768P","seed":1,
                          "prompt_expansion_mode":"disabled",
                          "fal_key_configured":bool(os.getenv("FAL_KEY")),
                          "prompt":PROMPT},indent=2))
        return
    # This code branch MUST NOT be reached until the user explicitly approves
    # this exact one-video generation after reviewing the 2x2 reference.
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY not found in local environment; no video submitted")
    flag=ROOT/"h3max_reset_single_video_request.json"
    result_out=ROOT/"h3max_reset_single_video_result.json"
    video=ROOT/"h3max_reset_single_video_output.mp4"
    if any(p.exists() for p in (flag,result_out,video)):
        raise FileExistsError("Preventing accidental second fal H3 video submission")
    import fal_client
    from urllib.parse import urlparse
    import requests
    url=fal_client.upload_file(str(IMAGE))
    if not isinstance(url,str) or not url.startswith("https://"):
        raise ValueError("fal file upload URL missing")
    # Saved before job submission to reduce the chance of duplicate paid jobs
    # after an interruption. No API keys are stored.
    flag.write_text(json.dumps({"state":"ABOUT_TO_SUBMIT","model":MODEL,
                 "image_sha256":sha(IMAGE),"image_url":url,
                 "input":SETTINGS},indent=2),encoding="utf-8")
    reply=fal_client.subscribe(MODEL,arguments={"image_url":url,**SETTINGS})
    result_out.write_text(json.dumps(reply,indent=2),encoding="utf-8")
    file=reply.get("video",{}).get("url")
    if not isinstance(file,str) or not file.startswith("https://"):
        raise RuntimeError("fal returned no valid output video URL")
    response=requests.get(file,timeout=180)
    response.raise_for_status()
    video.write_bytes(response.content)
    print("H3_MAX_ONE_VIDEO_COMPLETED",str(video),"bytes",video.stat().st_size)
if __name__=="__main__":
    main()
