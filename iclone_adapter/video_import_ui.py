"""iClone UI coordinator: Dragon MP4 or det.npz -> solved Motion -> iProject.

The iClone 8 Python 3.8 process remains RLPy/PySide2 ONLY.
Expensive video decoding, DINOv2 CUDA inference and torch fitting execute
in a separate Python 3.11 QProcess, with live progress and receipt files.
When motion.json is ready, the previously verified same-Dragon-rig
direct_import code validates and writes in the iClone process.

User confirms the scene change explicitly after motion capture finishes.
No paid fal.ai video-generation calls are made by this workflow.
"""
import json
import pathlib

from .direct_import import _selected_and_fps, prepare_direct_import, execute_direct_import
from .rlpy_reader import selected_object, _get_skeleton, snapshot
from .scene_guard import check_clean_clip
from .schema import ContractError
from .verify_import import DEFAULT_SOURCE, audit_rig
from .video_capture import _input_plan, OUTPUT_MOTION, RECEIPT

REPO=pathlib.Path(__file__).resolve().parents[1]


def preflight_video_target(rlpy):
    """Read-only guard BEFORE launching costly external video processing."""
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
            target=preflight_video_target(self.rlpy)
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
                    self.panel,"Select EXTERNAL Python 3.11 executable "
                               "(NOT iClone 3.8)",
                    str(REPO),"Python interpreter (python.exe python);;All files (*)")
                if not python:
                    return
            parent=self.widgets.QFileDialog.getExistingDirectory(
                self.panel,"Choose folder for new capture and native iProject",
                str(REPO/"runs"))
            if not parent:
                return
            capture=unique_capture_folder(parent,path)
            _input_plan(path,reader,reference,capture,require_python311=False)
            arguments=worker_arguments(path,capture,reader,reference)
            text=(
                "Dragon Motion Capture from an EXISTING two-camera source\n\n"
                "Selected Dragon: %s\n"
                "Source: %s\n"
                "Python: %s (must be 3.11)\n"
                "Capture folder: %s\n\n"
                "The worker runs original mocap.detect (for MP4), then "
                "mocap.solve in an external Python 3.11 process.\n"
                "It reads GPU-trained Dragon DINOv2 Reader for MP4; "
                "an existing det.npz skips Reader inference.\n"
                "No fal.ai or paid video requests are made.\n"
                "No iClone scene change occurs during capture. "
                "After motion.json is produced, iClone will ask for "
                "separate confirmation before writing animation.\n\n"
                "Start capture?"%
                (target["object_name"],path,python,str(capture))
            )
            answer=self.widgets.QMessageBox.question(
                self.panel,"Start Dragon Motion Capture",text,
                self.widgets.QMessageBox.Yes |
                self.widgets.QMessageBox.No,
                self.widgets.QMessageBox.No)
            if answer!=self.widgets.QMessageBox.Yes:
                return
            self.job={"source":path,"python":python,"capture_path":str(capture),
                      "output_parent":str(parent),"target":target}
            process=self.core.QProcess(self.panel)
            process.setProcessChannelMode(self.core.QProcess.MergedChannels)
            process.setWorkingDirectory(str(REPO))
            process.setProgram(python)
            process.setArguments(arguments)
            process.readyReadStandardOutput.connect(self._flush)
            process.finished.connect(self._finished)
            process.errorOccurred.connect(self._process_error)
            self.process=process
            self._log("Starting external Dragon Capture Worker: "+
                      " ".join(arguments))
            process.start()
        except Exception as exc:
            message=type(exc).__name__+": "+str(exc)
            self._log("Dragon Capture preflight FAILED: "+message)
            self._warning("Dragon Capture blocked",message)

    def _flush(self):
        if self.process is None:
            return
        raw=bytes(self.process.readAllStandardOutput())
        if raw:
            self._log(raw.decode("utf-8",errors="replace").rstrip("\r\n"))

    def _process_error(self,error):
        # Process-start failure can occur with a wrong Python executable.
        if self.process is not None and error==self.core.QProcess.FailedToStart:
            self._log("ERROR: external Python process could not start. "
                      "Select a valid Python 3.11 environment.")
            self._warning("Python 3.11 unavailable",
                          "The external worker failed to start. Install "
                          "Python 3.11 with PyTorch and requirements.txt.")

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
        if (exit_code!=0 or
            exit_status!=self.core.QProcess.NormalExit or
            not data or data.get("status")!="motion_json_ready"):
            status=data.get("status") if data else "no receipt"
            reason=data.get("error") if data else "external worker exited"
            msg=("Capture failed or needs quality review: %s\n%s\n\n"
                 "Partial results (including det.npz if created):\n%s\n\n"
                 "No changes were made to the iClone scene." %
                 (status,reason,str(capture)))
            self._log(msg)
            self._warning("Dragon Capture not imported",msg)
            return
        try:
            motion=capture/OUTPUT_MOTION
            plan=prepare_direct_import(str(motion),rlpy=self.rlpy)
            quality=data.get("motion_summary",{}).get("quality_status")
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
                    "Video/NPZ -> motion.json -> iClone project complete.\n\n"
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
                str(capture/OUTPUT_MOTION)+
                "\nDo not retry in the current scene if keys were already written.")
