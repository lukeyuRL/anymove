"""iClone UI coordinator: Dragon MP4 or det.npz -> solved Motion -> iProject.

The iClone 8 Python 3.8 process remains RLPy/PySide2 ONLY.
Expensive video decoding, DINOv2 CUDA inference and torch fitting execute
in a separate Python 3.11 or 3.14 QProcess, with live progress and receipt files.
Video generation and motion solving do not require a selected or clean
iClone actor. Once motion.json is ready, direct_import validates the
THEN-selected Dragon, with scene-write approval required.

User confirms the scene change explicitly after motion capture finishes.
The original existing MP4/NPZ path never calls fal.ai; a separate
start_prompt action is explicitly billable and requires user approval.
"""
import json
import pathlib
import subprocess

from .direct_import import _selected_and_fps, prepare_direct_import, execute_direct_import
from .rlpy_reader import selected_object, _get_skeleton, snapshot
from .scene_guard import check_clean_clip
from .schema import ContractError
from .verify_import import DEFAULT_SOURCE, audit_rig
from .video_capture import _input_plan, OUTPUT_MOTION, RECEIPT, supports_external_python
from .prompt_capture import _prompt_plan, _credential_available

REPO=pathlib.Path(__file__).resolve().parents[1]


def preflight_video_target(rlpy):
    """Optional clean-target guard; not a prerequisite for video capture."""
    obj,fps=_selected_and_fps(rlpy,24)
    with open(str(DEFAULT_SOURCE),encoding="utf-8") as f:
        source_info=json.load(f)
    live=snapshot(obj,rlpy=rlpy)
    report=audit_rig(source_info,live)
    scale=report.get("source_to_target_uniform_scale")
    if (report["result"]!="compatible_static_rest_snapshot" or
        type(scale) not in (int,float) or abs(scale-100.)>.25):
        raise ContractError(
            "Select the CLEAN original 42-Bone Dragon FBX imported at 100x; "
            "current actor does not match its Rest hierarchy/pose")
    clip=check_clean_clip(_get_skeleton(obj),obj,rlpy)
    return {
        "status":"clean_dragon_ready_for_external_motion_capture",
        "object_name":str(obj.GetName()),
        "matched_bones":report["matched_unique_names"],
        "fbx_import_scale":scale,
        "existing_clips":clip["existing_clip_count"],
        "project_fps":float(fps.ToDouble()),
        "note":"No scene mutation; actual clean reference is rechecked "
               "after the external worker finishes."
    }


def find_external_python(root):
    """Only reuse repo virtual environments; never use the iClone interpreter."""
    home=pathlib.Path(root)
    for path in (home/".venv"/"Scripts"/"python.exe",
                 home/".venv"/"bin"/"python",
                 home/"venv"/"Scripts"/"python.exe"):
        if path.is_file():
            return str(path)
    return None


def preflight_external_python(python,required_modules=()):
    """Verify an external interpreter before remote charges or GPU work.

    Avoid importing heavy CUDA libraries or contacting fal.ai here.
    This catches the common no-receipt failure: a wrong Python version
    or a required package absent from the selected .venv.
    """
    code=(
        "import importlib.util,json,sys;"
        "print(json.dumps({'version':list(sys.version_info[:2]),"
        "'executable':sys.executable,"
        "'missing':[m for m in sys.argv[1:] "
        "if importlib.util.find_spec(m) is None]}))"
    )
    try:
        check=subprocess.run(
            [str(python),"-c",code]+list(required_modules),
            cwd=str(REPO),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            timeout=20,encoding="utf-8",errors="replace",check=False)
    except (OSError,subprocess.TimeoutExpired) as exc:
        raise ContractError(
            "Cannot start/check external Python: %s (%s). "
            "Select a working Python 3.11 or 3.14 environment."%
            (str(python),str(exc)))
    output=(check.stdout or "").strip()
    if check.returncode!=0:
        raise ContractError(
            "External Python failed to start (%s, exit code %s): %s"%
            (str(python),check.returncode,output[-1200:] or "no output"))
    try:
        report=json.loads(output.splitlines()[-1])
    except (ValueError,IndexError,TypeError):
        raise ContractError(
            "External Python produced no valid environment report (%s): %s"%
            (str(python),output[-1200:]))
    version=report.get("version")
    if not supports_external_python(version):
        raise ContractError(
            "Wrong external Python version: %s uses %s; AnyMove requires "
            "Python 3.11 or 3.14. Recreate .venv using Python 3.11 or 3.14."%
            (report.get("executable") or python,
             ".".join(map(str,version)) if isinstance(version,list)
             else str(version)))
    missing=report.get("missing") or []
    if missing:
        raise ContractError(
            "External Python 3.11 or 3.14 is missing packages: %s. "
            "Install AnyMove requirements into %s before requesting "
            "a paid fal.ai video."%
            (", ".join(missing),report.get("executable") or python))
    return report


def unique_capture_folder(parent,source):
    folder=pathlib.Path(parent)
    if not folder.is_dir():
        raise ContractError("Output parent folder must exist")
    stem=pathlib.Path(source).stem
    allowed="".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    if not allowed:
        allowed="dragon"
    allowed=allowed[:40]
    for i in range(1,1000):
        candidate=folder/("capture_"+allowed+"_%03d"%i)
        if not candidate.exists():
            return candidate
    raise ContractError("too many capture runs in selected folder")


def prompt_worker_arguments(action,capture_path,reader,reference,
                            model="h3max",seed=1):
    if not isinstance(action,str) or not action.strip():
        raise ContractError("prompt action must be nonempty")
    return ["-m","iclone_adapter.prompt_capture",
            "--action",action.strip(),"--output",str(capture_path),
            "--reader",str(reader),"--reference",str(reference),
            "--model",model,"--seconds","5","--seed",str(seed)]


def worker_arguments(input_path,capture_path,reader=None,reference=None):
    argv=["-m","iclone_adapter.video_capture",
          "--input",str(input_path),"--output",str(capture_path)]
    if str(input_path).lower().endswith(".mp4"):
        if reader is None or reference is None:
            raise ContractError("Video capture requires Dragon reader.pt and ref.png")
        argv.extend(["--reader",str(reader),"--reference",str(reference)])
    elif not str(input_path).lower().endswith(".npz"):
        raise ContractError("only MP4 and NPZ are supported")
    return argv


class VideoCaptureCoordinator:
    def __init__(self,panel,qtwidgets,qtcore,rlpy):
        self.panel=panel
        self.widgets=qtwidgets
        self.core=qtcore
        self.rlpy=rlpy
        self.process=None
        self.job=None
        self._worker_log_tail=""

    def running(self):
        return self.process is not None and self.process.state()!=(
            self.core.QProcess.NotRunning)

    def _log(self,message):
        self.panel.out.appendPlainText(str(message))

    def _warning(self,title,text):
        self.widgets.QMessageBox.warning(self.panel,title,str(text))

    def start(self):
        if self.running():
            self._warning("Dragon capture","A Dragon capture is already running.")
            return
        path,_=self.widgets.QFileDialog.getOpenFileName(
            self.panel,"Select two-camera Dragon MP4 or existing det.npz",
            "","Dragon two-camera videos (*.mp4);;Reader detections (*.npz)")
        if not path:
            return
        try:
            # Actor-independent video generation/solving; validate on import.
            reader=None
            reference=None
            if path.lower().endswith(".mp4"):
                reader=REPO/"runs"/"dragon"/"reader.pt"
                reference=REPO/"runs"/"dragon"/"ref"/"ref.png"
                if not reader.is_file():
                    chosen,_=self.widgets.QFileDialog.getOpenFileName(
                        self.panel,"Select trained DRAGON DINOv2 reader.pt",
                        str(reader.parent),"PyTorch reader checkpoint (*.pt)")
                    if not chosen:
                        return
                    reader=pathlib.Path(chosen)
                if not reference.is_file():
                    chosen,_=self.widgets.QFileDialog.getOpenFileName(
                        self.panel,"Select DRAGON two-camera reference ref.png",
                        str(reference.parent),"Reference images (*.png *.jpg *.jpeg)")
                    if not chosen:
                        return
                    reference=pathlib.Path(chosen)
            python=find_external_python(REPO)
            if not python:
                python,_=self.widgets.QFileDialog.getOpenFileName(
                    self.panel,"Select EXTERNAL Python 3.11 or 3.14 executable "
                               "(NOT iClone 3.8)",
                    str(REPO),"Python interpreter (python.exe python);;All files (*)")
                if not python:
                    return
            modules=("numpy",) if path.lower().endswith(".npz") else (
                "numpy","cv2","torch","transformers","PIL")
            python_info=preflight_external_python(python,modules)
            self._log("External Python ready: "+python_info["executable"]+
                      " (Python "+ ".".join(str(v) for v in python_info.get("version", []))+
                      "; required package names found)")
            parent=self.widgets.QFileDialog.getExistingDirectory(
                self.panel,"Choose folder for new capture and native iProject",
                str(REPO/"runs"))
            if not parent:
                return
            capture=unique_capture_folder(parent,path)
            _input_plan(path,reader,reference,capture,require_supported_python=False)
            arguments=worker_arguments(path,capture,reader,reference)
            text=(
                "Dragon Motion Capture from an EXISTING two-camera source\n\n"
                "Source: %s\n"
                "Python: %s (must be 3.11 or 3.14)\n"
                "Capture folder: %s\n\n"
                "The worker runs original mocap.detect (for MP4), then "
                "mocap.solve in an external Python 3.11 or 3.14 process.\n"
                "It reads GPU-trained Dragon DINOv2 Reader for MP4; "
                "an existing det.npz skips Reader inference.\n"
                "No fal.ai or paid video requests are made.\n"
                "No Dragon selection is required for capture. "
                "After motion.json is produced, the THEN-selected Dragon "
                "is validated before iClone asks to write animation. "
                "Use a clean original 42-Bone Dragon at 100x, "
                "24 FPS and Frame 0 for import.\n\n"
                "Start capture?"%
                (path,python,str(capture))
            )
            answer=self.widgets.QMessageBox.question(
                self.panel,"Start Dragon Motion Capture",text,
                self.widgets.QMessageBox.Yes |
                self.widgets.QMessageBox.No,
                self.widgets.QMessageBox.No)
            if answer!=self.widgets.QMessageBox.Yes:
                return
            self._launch_worker(
                python,arguments,
                {"source":path,"python":python,"capture_path":str(capture),
                 "output_parent":str(parent),"mode":"existing"})
        except Exception as exc:
            message=type(exc).__name__+": "+str(exc)
            self._log("Dragon Capture preflight FAILED: "+message)
            self._warning("Dragon Capture blocked",message)


    def start_prompt(self):
        """Explicit opt-in billable prompt -> generated MP4 -> final iProject."""
        if self.running():
            self._warning("Dragon capture","Finish the current capture first.")
            return
        action,ok=self.widgets.QInputDialog.getMultiLineText(
            self.panel,"Dragon AI Video Action",
            "Describe the Dragon movement in English:",
            "the dragon spreads its wings, roars, then returns to rest.")
        if not ok or not action.strip():
            return
        try:
            # Actor-independent video generation/solving; validate on import.
            reader=REPO/"runs"/"dragon"/"reader.pt"
            reference=REPO/"runs"/"dragon"/"ref"/"ref.png"
            if not reader.is_file():
                chosen,_=self.widgets.QFileDialog.getOpenFileName(
                    self.panel,"Select trained DRAGON reader.pt",
                    str(reader.parent),"Dragon PyTorch reader (*.pt)")
                if not chosen:
                    return
                reader=pathlib.Path(chosen)
            if not reference.is_file():
                chosen,_=self.widgets.QFileDialog.getOpenFileName(
                    self.panel,"Select Dragon original two-camera ref.png",
                    str(reference.parent),"Reference image (*.png *.jpg *.jpeg)")
                if not chosen:
                    return
                reference=pathlib.Path(chosen)
            python=find_external_python(REPO)
            if not python:
                python,_=self.widgets.QFileDialog.getOpenFileName(
                    self.panel,"Select EXTERNAL Python 3.11 or 3.14 (not iClone 3.8)",
                    str(REPO),"Python executable (*.exe);;All files (*)")
                if not python:
                    return
            python_info=preflight_external_python(
                python,("requests","numpy","cv2","torch","transformers","PIL"))
            self._log("External Python ready: "+python_info["executable"]+
                      " (Python "+ ".".join(str(v) for v in python_info.get("version", []))+
                      "; required package names found)")
            parent=self.widgets.QFileDialog.getExistingDirectory(
                self.panel,"Choose parent folder for AI video + iProject",
                str(REPO/"runs"))
            if not parent:
                return
            capture=unique_capture_folder(parent,"dragon_ai_prompt.mp4")
            if not _credential_available():
                raise ContractError(
                    "FAL_KEY missing. Video generation is billable: set "
                    "FAL_KEY in the external environment or repo .env.")
            _prompt_plan(action,str(capture),reader,reference,
                         require_credentials=False,require_supported_python=False)
            arguments=prompt_worker_arguments(
                action,capture,reader,reference)
            text=(
                "OPTIONAL PAID AI VIDEO GENERATION\n\n"
                "Model: MiniMax H3 Max on fal.ai\n"
                "Duration: 5 seconds; Seed: 1\n"
                "Action: %s\n\n"
                "Video provider charges your fal.ai account according to "
                "its CURRENT pricing. This is a BILLABLE external request; "
                "model cost and availability may change.\n\n"
                "If approved, generate the two-camera Dragon video, then "
                "run DINOv2 + Motion Solver on external Python 3.11 or 3.14. "
                "After a successful quality check the THEN-selected Dragon "
                "will be verified; you will be asked AGAIN before the iClone "
                "scene is modified. If the target is unsuitable, motion.json "
                "is preserved for later import.\n\n"
                "Proceed with the paid video-generation request?"%
                action.strip())
            answer=self.widgets.QMessageBox.question(
                self.panel,"Confirm fal.ai external charge",text,
                self.widgets.QMessageBox.Yes |
                self.widgets.QMessageBox.No,
                self.widgets.QMessageBox.No)
            if answer!=self.widgets.QMessageBox.Yes:
                self._log("No paid video-generation request was sent.")
                return
            self._launch_worker(
                python,arguments,
                {"source":"Dragon Prompt: "+action.strip(),
                 "python":python,"capture_path":str(capture),
                 "output_parent":str(parent),
                 "mode":"prompt"})
        except Exception as exc:
            error=type(exc).__name__+": "+str(exc)
            self._log("Prompt Capture preflight failed: "+error)
            self._warning("Prompt Capture blocked",error)

    def _launch_worker(self,python,arguments,job):
        self._worker_log_tail=""
        self.job=job
        process=self.core.QProcess(self.panel)
        process.setProcessChannelMode(self.core.QProcess.MergedChannels)
        process.setWorkingDirectory(str(REPO))
        process.setProgram(str(python))
        process.setArguments(arguments)
        process.readyReadStandardOutput.connect(self._flush)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self.process=process
        self._log("Starting external Dragon Capture Worker: "+
                  " ".join(arguments))
        process.start()

    def _flush(self):
        if self.process is None:
            return
        raw=bytes(self.process.readAllStandardOutput())
        if raw:
            message=raw.decode("utf-8",errors="replace").rstrip("\r\n")
            self._worker_log_tail=(
                self._worker_log_tail+"\n"+message)[-4000:]
            self._log(message)

    def _process_error(self,error):
        # Process-start failure can occur with a wrong Python executable.
        if self.process is not None and error==self.core.QProcess.FailedToStart:
            self._log("ERROR: external Python process could not start. "
                      "Select a valid Python 3.11 or 3.14 environment.")
            self._warning("External Python unavailable",
                          "The external worker failed to start. Install "
                          "Python 3.11 or 3.14 with PyTorch and requirements.txt.")

    def _finished(self,exit_code,exit_status):
        self._flush()
        job=self.job
        self.process=None
        if not job:
            return
        capture=pathlib.Path(job["capture_path"])
        receipt=capture/RECEIPT
        data=None
        if receipt.is_file():
            try:
                with open(str(receipt),encoding="utf-8") as f:
                    data=json.load(f)
            except Exception as exc:
                self._log("Could not inspect Capture receipt: "+str(exc))
        # A solver quality gate is NOT a capture crash. The worker has
        # already written motion.json and its independent quality receipt.
        # Keep the strict 15 px reference threshold and require the user
        # to deliberately review/import the preserved motion, never accept
        # borderline quality automatically or rerun a paid video request.
        if data and data.get("status")=="quality_gate_blocked":
            motion=pathlib.Path(data.get("motion_file") or
                                (capture/OUTPUT_MOTION))
            if motion.is_file() and motion.stat().st_size>0:
                summary=data.get("motion_summary") or {}
                reasons=summary.get("quality_issues") or [
                    data.get("error") or "quality gate not met"]
                reproj=summary.get("reprojection_px") or {}
                reproj_text=(
                    "Front X reprojection: %s px (limit 10 px)\n"
                    "Side reprojection: %s px (limit 15 px)\n"%
                    (reproj.get("front_x_only","not reported"),
                     reproj.get("side","not reported")))
                msg=(
                    "Motion Solver FINISHED, but the result NEEDS REVIEW.\n"
                    "Quality issues:\n- %s\n\n"
                    "Frames: %s\n%s\n"
                    "Saved motion.json (NOT discarded):\n%s\n\n"
                    "The iClone scene has NOT been modified.\n"
                    "Do not pay to generate another video or rerun the "
                    "solver for this quality warning.\n\n"
                    "Inspect the source video, joint overlay (if available), "
                    "and resulting animation before use. For a manual test "
                    "on a fresh CLEAN 42-Bone Dragon (100x FBX, 24 FPS, "
                    "Frame 0, no Motion Clips), select 'IMPORT AnyMove "
                    "motion.json'. Manual import does NOT certify this "
                    "motion's quality and bypasses the solver quality gate."
                    %("\n- ".join(str(x) for x in reasons),
                      summary.get("frame_count","unknown"),
                      reproj_text,str(motion)))
                self._log("Dragon motion solved; QUALITY REVIEW required. "+
                          "\n"+msg)
                self._warning("Dragon motion ready - quality review",msg)
                return
        if (exit_code!=0 or
            exit_status!=self.core.QProcess.NormalExit or
            not data or data.get("status")!="motion_json_ready"):
            status=data.get("status") if data else "no receipt"
            if data:
                reason=data.get("error") or (
                    "Worker exited before producing a valid motion.json. "
                    "Review the worker output in the panel.")
            else:
                reason=(
                    "Worker exited BEFORE creating capture_receipt.json.\n"
                    "External Python: %s\n"
                    "Exit code: %s; process status: %s\n\n"
                    "Worker output (last 1600 characters):\n%s\n\n"
                    "If any generated MP4 already exists, inspect it "
                    "before retrying a paid fal.ai request."%
                    (job.get("python"),exit_code,exit_status,
                     self._worker_log_tail[-1600:] or "(no worker output)"))
            msg=("Capture failed or needs quality review: %s\n%s\n\n"
                 "Partial results (including det.npz if created):\n%s\n\n"
                 "No changes were made to the iClone scene."%
                 (status,reason,str(capture)))
            self._log(msg)
            self._warning("Dragon Capture not imported",msg)
            return
        try:
            motion=pathlib.Path(data.get("motion_file") or
                                (capture/OUTPUT_MOTION))
            if not motion.is_file():
                raise ContractError("Capture reported success but no motion.json exists")
            try:
                plan=prepare_direct_import(str(motion),rlpy=self.rlpy)
            except Exception as exc:
                reason=type(exc).__name__+": "+str(exc)
                self._log(
                    "Dragon motion.json READY; automatic iClone import DEFERRED: "
                    +reason+"\nPreserved motion.json: "+str(motion))
                self._warning(
                    "Dragon motion ready - import deferred",
                    "Video capture and motion solving completed successfully.\n"
                    "Saved motion.json:\n%s\n\n"
                    "Automatic iClone import could not be prepared:\n%s\n\n"
                    "No animation was written to the iClone scene. "
                    "When ready, select a clean original 42-Bone Dragon "
                    "(100x, 24 FPS, Frame 0, no prior Motion Clips), then use "
                    "'IMPORT AnyMove motion.json -> iClone + iProject'."%
                    (str(motion),reason))
                return
            quality=(data.get("motion_summary") or {}).get("quality_status")
            if quality is None and data.get("local_capture_receipt"):
                with open(data["local_capture_receipt"],encoding="utf-8") as f:
                    local=json.load(f)
                quality=(local.get("motion_summary") or {}).get("quality_status")
            question=(
                "Dragon Solve complete: motion.json is READY\n"
                "Quality: %s\nFrames: %d\n"
                "Rotation bones: %d\nRoot Position: %d\n"
                "Controller scalar samples: %d\n\n"
                "Now write all frames to the CLEAN selected Dragon scene, "
                "read back every controller sample and save a native iProject?"
                "\n\nThe scene WILL be modified. Do not run twice on one scene."%
                (quality,plan["frames"],plan["rotation_bones"],
                 plan["position_bones"],plan["expected_scalar_samples"])
            )
            reply=self.widgets.QMessageBox.question(
                self.panel,"Write solved Dragon Motion into iClone",question,
                self.widgets.QMessageBox.Yes |
                self.widgets.QMessageBox.No,
                self.widgets.QMessageBox.No)
            if reply!=self.widgets.QMessageBox.Yes:
                self._log("Motion solved but NOT imported. File preserved: "+
                          str(motion))
                return
            result=execute_direct_import(
                plan,str(capture),rlpy=self.rlpy,save_project=True)
            self._log(json.dumps({
                "import_status":result["status"],
                "iProject":result["iProject_path"],
                "audit_directory":result["output_directory"],
                "verified_samples":result.get("verified_scalar_samples"),
                "warnings":result.get("warnings",[])
            },ensure_ascii=False,indent=2))
            if result["status"]=="completed_full_frame_verified_project_saved":
                self.widgets.QMessageBox.information(
                    self.panel,"AnyMove Dragon iClone Import Complete",
                    "Source -> motion.json -> iClone project complete.\n\n"
                    "Project:\n%s\n\nReports:\n%s"%
                    (result["iProject_path"],result["output_directory"]))
            else:
                self._warning(
                    "iClone import requires review",
                    "Motion was solved, but the iClone final stage did not "
                    "complete native saving.\nStatus: %s\n\n"
                    "Check import_receipt.json under:\n%s\n\n"
                    "Do NOT retry in this keyed scene."%
                    (result["status"],result["output_directory"]))
        except Exception as exc:
            message=type(exc).__name__+": "+str(exc)
            self._log("Dragon Motion produced but iClone import failed: "+message)
            self._warning(
                "iClone integration failed",message+
                "\n\nOriginal solved motion.json remains at:\n"+
                str(pathlib.Path(data.get("motion_file")) if data and
                                 data.get("motion_file") else capture/OUTPUT_MOTION)+
                "\nDo not retry in the current scene if keys were already written.")
