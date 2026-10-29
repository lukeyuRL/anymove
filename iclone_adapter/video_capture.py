"""Capture a Dragon motion.json from an existing two-camera MP4 or det.npz.

This is an EXTERNAL Python 3.11 or 3.14 worker. Never import torch, OpenCV or DINOv2
inside iClone's Python 3.8 process. The worker invokes the original
mocap.detect and mocap.solve unchanged, in subprocesses so stdout/progress
can stream to the calling iClone QProcess.

The original Dragon rig and DINOv2 Reader (reader.pt) must already exist.
Video input additionally requires an unmodified split-screen reference PNG.
No output files are overwritten; every run gets its own directory.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys

from .schema import ContractError
from .verify_import import DEFAULT_SOURCE

REPO=pathlib.Path(__file__).resolve().parents[1]

# Only EXTERNAL workers use these interpreters; iClone stays on Python 3.8.
SUPPORTED_EXTERNAL_PYTHON_VERSIONS=((3,11),(3,14))
SUPPORTED_EXTERNAL_PYTHON_TEXT="3.11 or 3.14"


def supports_external_python(version=None):
    """Whether an external interpreter uses a supported AnyMove version."""
    version=sys.version_info if version is None else version
    try:
        return tuple(version[:2]) in SUPPORTED_EXTERNAL_PYTHON_VERSIONS
    except (IndexError,TypeError):
        return False


CAPTURE_VERSION="anymove.iclone.dragon_capture/1"
RECEIPT="capture_receipt.json"
OUTPUT_MOTION="motion.json"
OUTPUT_DETECTION="det.npz"
OUTPUT_OVERLAY="overlay.mp4"


def _sha(path):
    digest=hashlib.sha256()
    with open(str(path),"rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path,payload):
    target=pathlib.Path(path)
    temp=target.with_name(target.name+".partial")
    try:
        with open(str(temp),"w",encoding="utf-8") as f:
            json.dump(payload,f,ensure_ascii=False,indent=2,
                      allow_nan=False)
            f.write("\n")
        os.replace(str(temp),str(target))
    finally:
        if temp.exists():
            temp.unlink()


def _required_file(value,label,suffix):
    if not value:
        raise ContractError(label+" path required")
    path=pathlib.Path(value).resolve()
    if path.suffix.lower() not in suffix or not path.is_file():
        raise ContractError(label+" must be an existing "+"/".join(suffix)+" file")
    if path.stat().st_size<=0:
        raise ContractError(label+" file is empty")
    return path


def _input_plan(input_path,reader_path=None,ref_path=None,
                output_dir=None,require_supported_python=True):
    if require_supported_python and not supports_external_python():
        raise ContractError(
            "AnyMove detection/solver requires external Python %s; "
            "current worker is %d.%d"%
            (SUPPORTED_EXTERNAL_PYTHON_TEXT,
             sys.version_info[0],sys.version_info[1]))
    original=_required_file(input_path,"source clip/detections",
                            (".mp4",".npz"))
    mode="video" if original.suffix.lower()==".mp4" else "detections"
    reader=None
    ref=None
    if mode=="video":
        reader=_required_file(reader_path,"trained Dragon reader",(".pt",))
        ref=_required_file(ref_path,"two-camera reference",(".png",".jpg",".jpeg"))
    rig=_required_file(DEFAULT_SOURCE,"Dragon rig_info",(".json",))
    if not isinstance(output_dir,(str,pathlib.Path)):
        raise ContractError("new output directory path required")
    output=pathlib.Path(output_dir).resolve()
    if not output.parent.is_dir():
        raise ContractError(
            "capture output parent directory must exist: "+str(output.parent))
    if output.exists():
        raise ContractError("capture output directory already exists; cannot overwrite")
    return {
        "schema_version":CAPTURE_VERSION,
        "mode":mode,
        "input_path":str(original),
        "input_sha256":_sha(original),
        "reader_path":str(reader) if reader else None,
        "reference_path":str(ref) if ref else None,
        "rig_info_path":str(rig),
        "rig_info_sha256":_sha(rig),
        "output_path":str(output),
        "motion_path":str(output/OUTPUT_MOTION),
        "required_python":SUPPORTED_EXTERNAL_PYTHON_TEXT,
        "creature":"dragon",
        "note":"Reader requires trained DINOv2 + CUDA; default Wolf reader "
               "or arbitrary video is not supported. Use one fixed "
               "two-camera Dragon video or existing det.npz."
    }


def _read_detection_stats(path):
    """Requires NumPy in external Python, never iClone."""
    try:
        import numpy as np
    except ImportError:
        raise ContractError("numpy unavailable; install requirements in Python 3.11 or 3.14")
    with np.load(str(path),allow_pickle=False) as source:
        required=("front_uv","side_uv","front_conf","side_conf","fps","names")
        missing=[x for x in required if x not in source.files]
        if missing:
            raise ContractError("det.npz missing fields: "+", ".join(missing))
        u=source["front_uv"]
        v=source["side_uv"]
        fc=source["front_conf"]
        sc=source["side_conf"]
        fps=float(source["fps"])
        names=[str(x) for x in source["names"].tolist()]
        with open(str(DEFAULT_SOURCE),encoding="utf-8") as f:
            expected=[x["name"] for x in json.load(f)["keypoints"]]
        count=len(u)
        if (count<2 or count>50000 or u.shape!=(count,len(expected),2)
            or v.shape!=u.shape or fc.shape!=(count,len(expected))
            or sc.shape!=fc.shape):
            raise ContractError("det.npz frame/joint shapes invalid for Dragon")
        if names!=expected:
            raise ContractError(
                "DINOv2 reader joint names/order do NOT match Dragon rig. "
                "Use the Dragon reader.pt, not another creature's.")
        if not math.isfinite(fps) or abs(fps-24.0)>1e-5:
            raise ContractError(
                "source video/detections FPS is %.6f; exact 24 FPS required "
                "by verified iClone reference profile (do not silently resample)"%
                fps)
        for label,array in (("front_uv",u),("side_uv",v),
                            ("front_conf",fc),("side_conf",sc)):
            if not np.isfinite(array).all():
                raise ContractError("nonfinite detections in "+label)
        if (np.any(fc<0) or np.any(fc>1) or
            np.any(sc<0) or np.any(sc>1)):
            raise ContractError("joint confidence outside [0,1]")
        return {
            "frame_count":int(count),
            "joint_count":len(names),
            "fps":fps,
            "mean_front_confidence":float(fc.mean()),
            "mean_side_confidence":float(sc.mean())
        }


def _read_motion_stats(path,expected_frames,allow_low_quality=False):
    with open(str(path),encoding="utf-8") as f:
        m=json.load(f)
    with open(str(DEFAULT_SOURCE),encoding="utf-8") as f:
        rig=json.load(f)
    names=[b["name"] for b in rig["bones"]]
    frames=m.get("euler")
    n=len(frames) if isinstance(frames,list) else 0
    if (n!=expected_frames or not 2<=n<=50000 or
        m.get("bones")!=names or
        type(m.get("fps")) not in (int,float) or
        abs(float(m["fps"])-24)>1e-5 or
        len(m.get("root_quat",[]))!=n or
        len(m.get("trans",[]))!=n):
        raise ContractError("motion.json output is not a 24-FPS complete Dragon motion")
    def finite(row,size):
        return (isinstance(row,(list,tuple)) and len(row)==size
                and all(type(x) in (float,int) and math.isfinite(x)
                        for x in row))
    for i in range(n):
        if (not isinstance(frames[i],list) or len(frames[i])!=42
            or not all(finite(row,3) for row in frames[i])
            or not finite(m["root_quat"][i],4)
            or not finite(m["trans"][i],3)):
            raise ContractError("nonfinite/incomplete motion at frame %d"%i)
    reproj=m.get("reproj_px") or {}
    sync=m.get("sync") or {}
    paw=(sync.get("paw_agreement") or {}).get("score")
    if paw is not None and not(math.isfinite(float(paw)) and -1<=paw<=1):
        raise ContractError("invalid paw agreement")
    side=reproj.get("side")
    front=reproj.get("front_x_only")
    problems=[]
    if paw is not None and paw<0.5:
        problems.append("paw agreement %.3g below 0.5"%paw)
    if side is not None and (not math.isfinite(float(side)) or side>15):
        problems.append("side reprojection %.3g px > 15 px"%side)
    if front is not None and (not math.isfinite(float(front)) or front>10):
        problems.append("front X reprojection %.3g px > 10 px"%front)
    quality_status=("needs_review" if problems else "within_reference_quality_gates")
    return {
        "frame_count":n,
        "bones":len(names),
        "fps":float(m["fps"]),
        "paw_agreement":paw,
        "reprojection_px":reproj,
        "quality_status":quality_status,
        "quality_issues":problems,
        "accepted":bool(not problems or allow_low_quality)
    }


def _run_step(command,env):
    print("Running:", " ".join(str(v) for v in command),flush=True)
    subprocess.run(command,cwd=str(REPO),env=env,check=True)


def run_capture(input_path,output_dir,reader_path=None,ref_path=None,
                allow_low_quality=False,runner=None):
    """Create motion.json; failures retain partial artifact and receipt."""
    plan=_input_plan(input_path,reader_path,ref_path,output_dir)
    output=pathlib.Path(plan["output_path"])
    output.mkdir()
    report={
        "schema_version":CAPTURE_VERSION,
        "status":"capture_started",
        "source":plan,"det_file":None,"motion_file":None,
        "detection_summary":None,"motion_summary":None,
        "source_motion_sha256":None,"error":None
    }
    _atomic_json(output/RECEIPT,report)
    run=runner or _run_step
    env=os.environ.copy()
    env["MOCAP_CREATURE"]="assets/dragon"
    try:
        det=output/OUTPUT_DETECTION
        if plan["mode"]=="video":
            report["status"]="detecting_joint_reader"
            _atomic_json(output/RECEIPT,report)
            run([sys.executable,"-m","mocap.detect",
                 plan["input_path"],plan["reader_path"],
                 plan["reference_path"],str(det),
                 str(output/OUTPUT_OVERLAY)],env)
            if not det.is_file() or det.stat().st_size==0:
                raise ContractError("mocap.detect did not create det.npz")
        else:
            shutil.copyfile(plan["input_path"],str(det))
        summary=_read_detection_stats(det)
        report["detection_summary"]=summary
        report["det_file"]=str(det)
        report["status"]="solving_3d_pose"
        _atomic_json(output/RECEIPT,report)
        print("Detection OK:",summary["frame_count"],"frames,",
              summary["joint_count"],"joints at 24 FPS",flush=True)
        motion=output/OUTPUT_MOTION
        run([sys.executable,"-m","mocap.solve",
             str(det),plan["rig_info_path"],str(motion)],env)
        if not motion.is_file() or motion.stat().st_size==0:
            raise ContractError("mocap.solve did not create motion.json")
        result=_read_motion_stats(motion,summary["frame_count"],
                                  allow_low_quality=allow_low_quality)
        report["motion_file"]=str(motion)
        report["motion_summary"]=result
        report["source_motion_sha256"]=_sha(motion)
        if not result["accepted"]:
            report["status"]="quality_gate_blocked"
            report["error"]="; ".join(result["quality_issues"])
        else:
            report["status"]="motion_json_ready"
        _atomic_json(output/RECEIPT,report)
        print("CAPTURE STATUS:",report["status"],flush=True)
        if report["error"]:
            print("QUALITY REVIEW:",report["error"],flush=True)
        return report
    except Exception as exc:
        report["status"]="capture_failed"
        report["error"]=type(exc).__name__+": "+str(exc)
        _atomic_json(output/RECEIPT,report)
        print("CAPTURE FAILED:",report["error"],flush=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=
        "Capture original Dragon AnyMove motion.json in external Python 3.11 or 3.14")
    parser.add_argument("--input",required=True,
                        help="existing split-screen Dragon MP4 or detections NPZ")
    parser.add_argument("--output",required=True,
                        help="NEW output folder (must not exist)")
    parser.add_argument("--reader",default=None,
                        help="trained Dragon DINOv2 reader.pt (MP4 only)")
    parser.add_argument("--reference",default=None,
                        help="Dragon two-camera ref.png (MP4 only)")
    parser.add_argument("--allow-low-quality",action="store_true",
                        help="override the quality gate only after reviewing the video")
    parser.add_argument("--dry-run",action="store_true",
                        help="validate paths and Python version, no heavy compute")
    args=parser.parse_args()
    if args.dry_run:
        print(json.dumps(_input_plan(
            args.input,args.reader,args.reference,args.output),
            ensure_ascii=False,indent=2))
        return
    result=run_capture(args.input,args.output,args.reader,
                       args.reference,allow_low_quality=args.allow_low_quality)
    if result["status"]!="motion_json_ready":
        raise SystemExit(2)


if __name__=="__main__":
    main()
