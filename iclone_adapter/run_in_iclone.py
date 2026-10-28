"""Minimal in-app P0/P1 diagnostics. Heavy PyTorch runs outside iClone."""
import json

from .motion_contract import make_sine_test
from .motion_writer import write_selected
from .p1_inspector import inspect_selected_animation
from .fk_pose_validator import validate_live_fk_pose
from .animation_bone_audit import audit_animation_bones
from .diagnostics import save_failure_report, raw_skeleton_report
from .bone_registry import build_bone_registry
from . import __version__
from .rlpy_reader import capability_report, export_selected, selected_object

_WINDOW = None


def show():
    import RLPy
    from PySide2 import QtWidgets

    class Panel(QtWidgets.QDialog):
        def __init__(self):
            super(Panel, self).__init__()
            self.setWindowTitle("AnyMove for iClone - P0/P1/P2 Diagnostics v" + __version__)
            self.resize(620, 420)
            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(QtWidgets.QLabel(
                "P0: read existing Skin Bones. P1: direct FK channel diagnostics ONLY.\n"
                "Test on an expendable scene; one default Motion Clip is OK.\n"
                "Use converted FK preview JSON, NOT raw Blender motion.json."))
            self.out = QtWidgets.QPlainTextEdit()
            self.out.setReadOnly(True)
            layout.addWidget(self.out)
            for label, callback in [
                ("Inspect selected skeleton", self.inspect),
                ("Export current-pose skeleton snapshot", self.export),
                ("Save raw Skin Bone diagnostics (no export)", self.debug_bones),
                ("Inspect existing motion clips / FK key counts", self.inspect_clips),
                ("Create visible 48-frame FK motion (30 deg)", self.synthetic),
                ("Write FK motion to selected avatar (modifies scene)", self.import_motion),
                ("Validate Dragon FK animation vs preview (read-only)", self.validate_dragon),
                ("Audit iClone 8 FK Bone IDs / Keys (read-only)", self.audit_motion_bones),
                ("Inspect Pelvis FK Root Position controls (read-only)", self.inspect_root),
            ]:
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(callback)
                layout.addWidget(button)

        def report(self, callback):
            try:
                self.out.appendPlainText(json.dumps(callback(), ensure_ascii=False, indent=2))
            except Exception as exc:
                self.out.appendPlainText(type(exc).__name__ + ": " + str(exc))

        def inspect(self):
            self.report(lambda: {
                "build": __version__,
                "panel_module": __file__,
                "capability": capability_report(selected_object(RLPy))
            })

        def export(self):
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export skeleton snapshot", "skeleton.json", "JSON (*.json)")
            if path:
                try:
                    data = export_selected(path, RLPy)
                    self.out.appendPlainText("Saved skeleton.json: %s (%d bones)" %
                                             (path, len(data["bones"])))
                except Exception as exc:
                    message = type(exc).__name__ + ": " + str(exc)
                    try:
                        debug_path = save_failure_report(path, RLPy, exc)
                    except Exception as debug_exc:
                        debug_path = "Debug report also failed: " + repr(debug_exc)
                    self.out.appendPlainText("Export failed: " + message +
                                             "\\nDiagnostic report: " + debug_path)
                    QtWidgets.QMessageBox.critical(
                        self, "Skeleton export failed",
                        message + "\\n\\nDiagnostic report: " + debug_path)

        def debug_bones(self):
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save raw skeleton diagnostics", "skeleton_debug.json", "JSON (*.json)")
            if path:
                self.report(lambda: self._save_debug(path))

        def _save_debug(self, path):
            report = raw_skeleton_report(RLPy)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            return {"written": path, "skin_bones": report.get("skin_bone_count"),
                    "issues": report.get("issues", [])}

        def inspect_clips(self):
            self.report(lambda: inspect_selected_animation(RLPy))

        def synthetic(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Choose exported skeleton JSON", "", "JSON (*.json)")
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    skeleton = json.load(f)
                names = ["%s : %s" % (b["id"], b["path"]) for b in skeleton["bones"]]
                preferred = ("CC_Base_L_Forearm", "CC_Base_L_Upperarm",
                             "CC_Base_R_Forearm", "CC_Base_Head")
                default = 0
                for wanted in preferred:
                    match = next((i for i, b in enumerate(skeleton["bones"])
                                  if b["name"] == wanted), None)
                    if match is not None:
                        default = match
                        break
                choice, ok = QtWidgets.QInputDialog.getItem(
                    self, "Diagnostic bone", "Choose a visible deforming Bone", names,
                    default, False)
                if not ok:
                    return
                bone_id = choice.split(" : ", 1)[0]
                target, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save diagnostic", "test_fk.json", "JSON (*.json)")
                if target:
                    motion = make_sine_test(skeleton, bone_id,
                                            amplitude_deg=30.0)
                    with open(target, "w", encoding="utf-8") as f:
                        json.dump(motion, f, indent=2)
                    self.out.appendPlainText(
                        "Wrote 24fps 48-frame 30-degree FK diagnostic: " + target)
            except Exception as exc:
                self.out.appendPlainText(type(exc).__name__ + ": " + str(exc))

        def validate_dragon(self):
            baseline, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select UNPOSED original Dragon skeleton.json", "", "JSON (*.json)")
            if not baseline:
                return
            preview, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select the SAME Dragon FK preview JSON you already wrote", "", "JSON (*.json)")
            if not preview:
                return
            obj = selected_object(RLPy)
            skel = obj.GetSkeletonComponent()
            try:
                count = int(skel.GetClipCount())
                if count == 0:
                    raise ValueError("No Motion Clips on Dragon")
                index, ok = QtWidgets.QInputDialog.getInt(
                    self, "Select Motion Clip to evaluate",
                    "Motion Clip index (last is usually the new test):",
                    count-1, 0, count-1)
                if not ok:
                    return
                with open(baseline, "r", encoding="utf-8") as f:
                    rest = json.load(f)
                with open(preview, "r", encoding="utf-8") as f:
                    motion = json.load(f)
                report = validate_live_fk_pose(rest, motion, clip_index=index, rlpy=RLPy)
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save detailed Dragon FK validation report",
                    "dragon_fk_validation.json", "JSON (*.json)")
                if not path:
                    self.out.appendPlainText(json.dumps({
                        k:v for k,v in report.items()
                        if k!="sample_details"},ensure_ascii=False,indent=2))
                    return
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                self.out.appendPlainText(json.dumps({
                    "saved_validation_report":path,
                    "result":report["result"],
                    "meaningful_motion_bone_count":report["meaningful_motion_bone_count"],
                    "consistent_bone_count":report["consistent_bone_count"],
                    "axis_testable_bone_count":report["axis_testable_bone_count"],
                    "static_node_bone_count":report["static_node_bone_count"],
                    "validity_warning":report["validity_warning"],
                    "worst_bones":report["by_bone"][:10],
                },ensure_ascii=False,indent=2))
            except Exception as exc:
                message = "%s: %s" % (type(exc).__name__, exc)
                self.out.appendPlainText(message)
                QtWidgets.QMessageBox.warning(self,"Dragon FK validation failed",message)

        def audit_motion_bones(self):
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Dragon FK preview JSON already written to this scene",
                "", "JSON (*.json)")
            if not file_path:
                return
            try:
                with open(file_path,"r",encoding="utf-8") as f:
                    motion=json.load(f)
                obj=selected_object(RLPy)
                count=int(obj.GetSkeletonComponent().GetClipCount())
                if not count:
                    raise ValueError("selected Dragon has no Motion Clips")
                index, ok=QtWidgets.QInputDialog.getInt(
                    self, "Select Clip", "Motion Clip index:",count-1,0,count-1)
                if not ok:
                    return
                report=audit_animation_bones(motion,index,rlpy=RLPy)
                path, _=QtWidgets.QFileDialog.getSaveFileName(
                    self,"Save read-only Bone / FK diagnostic",
                    "dragon_bone_channel_audit.json","JSON (*.json)")
                if path:
                    with open(path,"w",encoding="utf-8") as f:
                        json.dump(report,f,ensure_ascii=False,indent=2)
                summary={"saved_report":path or None,"clip_index":index,
                         "source_motion_frame_count":motion["frame_count"],
                         "channel_integrity":report["channel_integrity"]}
                for key,info in report["node_sets"].items():
                    summary[key]={
                        "available":info.get("available"),
                        "sdk_method":info.get("method"),
                        "node_count":info.get("node_count"),
                        "matching_target_names":info.get("matching_target_names"),
                        "target_bones_with_any_fk_key":
                            info.get("target_bones_with_any_fk_key"),
                        "target_bones_readback_matches_1e3":
                            info.get("target_bones_readback_matches_1e3"),
                        "error":info.get("error")}
                summary["different_id_counts"]={
                    key:len(val) for key,val in
                    report["same_name_identity_differences"].items()}
                self.out.appendPlainText(json.dumps(
                    summary,ensure_ascii=False,indent=2))
            except Exception as exc:
                message="%s: %s"%(type(exc).__name__,exc)
                self.out.appendPlainText(message)
                QtWidgets.QMessageBox.warning(
                    self,"Bone / Motion Clip diagnostic failed",message)

        def inspect_root(self):
            try:
                obj=selected_object(RLPy)
                skel=obj.GetSkeletonComponent()
                bones,_=build_bone_registry(skel.GetSkinBones() or [])
                pelvis=[b for b in bones if b["name"]=="pelvis"]
                if len(pelvis)!=1:
                    raise ValueError("exactly one Pelvis bone required")
                count=int(skel.GetClipCount())
                if count<1:
                    self.out.appendPlainText(
                        "Root channel inspection: no Clip yet. This is normal "
                        "for a fresh scene. Writer will probe before writing.")
                    return
                clip=skel.GetClip(0)
                control=clip.GetControl("Layer",pelvis[0]["node"])
                if control is None:
                    raise ValueError("Pelvis has no Layer control")
                block=control.GetDataBlock()
                if block is None:
                    raise ValueError("Pelvis Layer has no DataBlock")
                result={"bone":"pelvis","clips":count,"clip_index":0,
                        "root_position_channels":{},
                        "root_rotation_channels":{}}
                for kind in ("Rotation","Position"):
                    for axis in "XYZ":
                        key=kind+"/"+kind+axis
                        channel=block.GetControl(key)
                        result["root_"+kind.lower()+"_channels"][axis]={
                            "available":channel is not None,
                            "key_count":int(channel.GetKeyCount()) if channel is not None else None}
                self.out.appendPlainText(json.dumps(result,indent=2))
            except Exception as exc:
                self.out.appendPlainText("Root channel inspection failed: "+
                                         type(exc).__name__+": "+str(exc))

        def import_motion(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select explicit iClone FK motion", "", "JSON (*.json)")
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    motion = json.load(f)
            except Exception as exc:
                self.out.appendPlainText(str(exc))
                return
            preview_notice = ""
            if str(motion.get("calibration_status","")).startswith("unverified_"):
                preview_notice = (
                    "\nEXPERIMENTAL DRAGON PREVIEW: FK channel axes are not "
                    "production calibrated. Strict unposed-scene guard will "
                    "reject reused animation projects.\n"
                )
            root=motion.get("root_motion")
            if root:
                preview_notice += (
                    "\nUNVERIFIED Pelvis ROOT MOTION: mode %s, position "
                    "scale %s, position space %s. Test only; Root Translation "
                    "units and channel space are not yet calibrated.\n" %
                    (root.get("components"),root.get("root_position_scale"),
                     root.get("root_position_space"))
                )
            message = (
                "Confirm project FPS is %s and timeline is at Frame 0.\n" %
                motion.get("fps")
            )
            message += (
                "Use a disposable TEST project only.\n"
                "Experimental Dragon clips require a matching unposed "
                "reference and no existing animation.\n"
                "Use a clean project; this writer will not append a Dragon "
                "preview over previous clips.\n"
                "The playhead moves to the peak test pose."
            )
            message += preview_notice + "\nContinue?"
            answer = QtWidgets.QMessageBox.warning(
                self, "Experimental FK clip write", message,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
            try:
                result = write_selected(motion, fps_confirmed=True, rlpy=RLPy)
                self.out.appendPlainText(json.dumps(result, ensure_ascii=False, indent=2))
                if result.get("status") != "fk_keys_verified_pose_changed":
                    QtWidgets.QMessageBox.warning(
                        self, "FK keys exist - visual result not confirmed",
                        "FK keyframe data was written, but the chosen bone's "
                        "world pose did not visibly change or could not be checked. "
                        "Use Inspect existing motion clips and review the output. "
                        "Active IK may override FK.")
            except Exception as exc:
                message = "%s: %s" % (type(exc).__name__, exc)
                self.out.appendPlainText(message)
                QtWidgets.QMessageBox.critical(self, "FK Motion write failed", message)

    global _WINDOW
    if _WINDOW is not None:
        _WINDOW.close()
    _WINDOW = Panel()
    _WINDOW.show()
    return _WINDOW
