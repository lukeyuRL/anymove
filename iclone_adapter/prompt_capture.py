"""Opt-in paid prompt -> fal.ai Dragon video -> DINOv2 -> solved motion.json.

External Python 3.11 or 3.14 ONLY. This operation CAN INCUR remote fal.ai charges.
It is never called automatically by the iClone UI; the user must explicitly
confirm the billable AI video generation before QProcess starts.

Original project components:
- tools/make_prompt.py: template preserving two-camera, color-coded Dragon
- mocap.generate: fal queue client, writes generated/<model>.mp4
- video_capture.run_capture: DINOv2 detect, then mocap.solve

Video model outputs can violate the two-camera contract even when API calls
succeed. Capture still has numeric sync/reprojection quality gates.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

from .schema import ContractError
from .video_capture import (
    REPO, _required_file, _atomic_json, _run_step, run_capture,
    supports_external_python, SUPPORTED_EXTERNAL_PYTHON_TEXT
)

VERSION="anymove.iclone.dragon_prompt_capture/1"
RECEIPT="capture_receipt.json"
ALLOWED_MODELS=("h3max","h3maxturbo","h3","seedance","flux3max")


def _credential_available():
    if os.environ.get("FAL_KEY"):
        return True
    path=REPO/".env"
    if not path.is_file():
        return False
    try:
        with open(str(path),encoding="utf-8") as f:
            for line in f:
                if line.startswith("FAL_KEY=") and line.split("=",1)[1].strip():
                    return True
    except OSError:
        return False
    return False


def _prompt_plan(action,output_dir,reader,reference,model="h3max",
                 seconds=5,seed=1,require_credentials=True,
                 require_supported_python=True):
    if require_supported_python and not supports_external_python():
        raise ContractError(
            "AI prompt worker requires external Python %s; current is %d.%d"%
            (SUPPORTED_EXTERNAL_PYTHON_TEXT,
             sys.version_info[0],sys.version_info[1]))
    if not isinstance(action,str) or not 8<=len(action.strip())<=1000:
        raise ContractError("action text must contain 8 to 1000 characters")
    if "\x00" in action:
        raise ContractError("invalid action text")
    if model not in ALLOWED_MODELS:
        raise ContractError("unsupported fal video model")
    if type(seconds) is not int or not 3<=seconds<=10:
        raise ContractError("video duration out of supported range")
    if type(seed) is not int or not 0<=seed<=1000000:
        raise ContractError("invalid video generation seed")
    reader_path=_required_file(reader,"trained Dragon reader",(".pt",))
    ref_path=_required_file(reference,"Dragon two-camera reference",(".png",".jpg",".jpeg"))
    output=pathlib.Path(output_dir).resolve()
    if not output.parent.is_dir() or output.exists():
        raise ContractError("prompt capture output must be a NEW directory under an existing parent")
    if require_credentials and not _credential_available():
        raise ContractError(
            "FAL_KEY not configured. AI video generation uses billable fal.ai; "
            "put FAL_KEY in the user environment or repository .env")
    return {
        "schema_version":VERSION,"action":action.strip(),
        "model":model,"seconds":seconds,"seed":seed,
        "reader_path":str(reader_path),"reference_path":str(ref_path),
        "output_path":str(output),"billable_external_video_request":True,
        "status":"preflight_pass"
    }


def run_prompt_capture(action,output_dir,reader,reference,model="h3max",
                       seconds=5,seed=1,runner=None):
    """One explicit opt-in paid generation, followed by local capture."""
    plan=_prompt_plan(action,output_dir,reader,reference,
                      model=model,seconds=seconds,seed=seed)
    output=pathlib.Path(plan["output_path"])
    output.mkdir()
    report={
        "schema_version":VERSION,"status":"prompt_capture_started",
        "mode":"prompt_to_video_to_motion",
        "plan":plan,"generated_video":None,"motion_file":None,
        "local_capture_receipt":None,"error":None
    }
    _atomic_json(output/RECEIPT,report)
    run=runner or _run_step
    env=os.environ.copy()
    env["MOCAP_CREATURE"]="assets/dragon"
    try:
        prompt=output/"prompt.txt"
        generated=output/"generated"
        generated.mkdir()
        report["status"]="building_dragon_two_camera_prompt"
        _atomic_json(output/RECEIPT,report)
        run([sys.executable,"tools/make_prompt.py",
             str(REPO/"assets"/"dragon"),plan["action"],str(prompt)],env)
        if not prompt.is_file() or not prompt.read_text(encoding="utf-8").strip():
            raise ContractError("make_prompt.py did not create a valid prompt")
        report["status"]="generating_remote_video_billable"
        _atomic_json(output/RECEIPT,report)
        print("Launching user-approved fal.ai video request:",model,
              "seconds:",seconds,"seed:",seed,flush=True)
        run([sys.executable,"-m","mocap.generate",plan["reference_path"],
             str(prompt),str(generated),model,str(seconds),str(seed)],env)
        video=generated/(model+".mp4")
        if not video.is_file() or video.stat().st_size<=0:
            raise ContractError(
                "fal.ai video did not produce expected "+str(video))
        report["generated_video"]=str(video)
        report["status"]="local_dinov2_and_solver_running"
        _atomic_json(output/RECEIPT,report)
        local=run_capture(str(video),str(output/"capture"),
                          reader_path=plan["reader_path"],
                          ref_path=plan["reference_path"],runner=run)
        report["local_capture_receipt"]=str(
            output/"capture"/"capture_receipt.json")
        report["motion_file"]=local.get("motion_file")
        report["source_motion_sha256"]=local.get("source_motion_sha256")
        if local["status"]=="motion_json_ready":
            report["status"]="motion_json_ready"
        else:
            report["status"]=local["status"]
            report["error"]=local.get("error")
        _atomic_json(output/RECEIPT,report)
        print("PROMPT CAPTURE STATUS:",report["status"],flush=True)
        return report
    except Exception as exc:
        report["status"]="prompt_capture_failed"
        report["error"]=type(exc).__name__+": "+str(exc)
        _atomic_json(output/RECEIPT,report)
        print("PROMPT CAPTURE FAILED:",report["error"],flush=True)
        raise


def main():
    parser=argparse.ArgumentParser(
        description="Explicitly approved BILLABLE fal.ai video generation, then local Dragon mocap")
    parser.add_argument("--action",required=True,
                        help="English movement instruction for the Dragon")
    parser.add_argument("--output",required=True,
                        help="new output folder (must not already exist)")
    parser.add_argument("--reader",required=True)
    parser.add_argument("--reference",required=True)
    parser.add_argument("--model",choices=ALLOWED_MODELS,default="h3max")
    parser.add_argument("--seconds",type=int,default=5)
    parser.add_argument("--seed",type=int,default=1)
    args=parser.parse_args()
    status=run_prompt_capture(
        args.action,args.output,args.reader,args.reference,
        model=args.model,seconds=args.seconds,seed=args.seed)
    if status["status"]!="motion_json_ready":
        raise SystemExit(2)


if __name__=="__main__":
    main()
