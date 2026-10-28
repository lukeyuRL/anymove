# coding: utf-8
"""STANDALONE iClone P0 diagnostic. Load this file via Script > Load Python.

Deliberately does not import iclone_adapter or PySide2 before creating
%TEMP%/anymove_iclone_probe.json. It is READ-ONLY for the iClone scene.
This script runs immediately when loaded (no console/import assumptions).
"""
import collections
import datetime
import json
import os
import sys
import tempfile
import traceback

BUILD = "P0-STANDALONE-20261028-01"
NAME = "anymove_iclone_probe.json"


def _safe(fn):
    try:
        value = fn()
        return str(value) if value is not None else None
    except Exception as exc:
        return "<%s: %s>" % (type(exc).__name__, exc)


def _output():
    candidates = [
        os.environ.get("TEMP"),
        tempfile.gettempdir(),
        os.path.expanduser("~"),
        os.getcwd(),
    ]
    failures = []
    for folder in candidates:
        if not folder:
            continue
        try:
            os.makedirs(folder, exist_ok=True)
            target = os.path.join(folder, NAME)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write('{"status":"started"}\n')
            return target
        except Exception as exc:
            failures.append("%s: %s" % (folder, exc))
    raise RuntimeError("cannot create diagnostic JSON: " + "; ".join(failures))


def run():
    script_file = os.path.abspath(globals().get("__file__", ""))
    root = os.path.dirname(script_file)
    info = {
        "build": BUILD,
        "status": "started",
        "started_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "script_file": script_file,
        "repo_root": root,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "previously_loaded_modules": {
            name: getattr(mod, "__file__", None)
            for name, mod in list(sys.modules.items())
            if name == "iclone_adapter" or name.startswith("iclone_adapter.")
        },
        "steps": [],
        "errors": [],
        "bone_rows": [],
    }
    path = None

    def flush():
        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(info, handle, ensure_ascii=False, indent=2, default=str)

    def stage(name, func):
        try:
            result = func()
            info["steps"].append({"stage": name, "status": "ok"})
            flush()
            return result
        except Exception as exc:
            info["steps"].append({"stage": name, "status": "failed"})
            info["errors"].append({
                "stage": name,
                "exception": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            })
            flush()
            return None

    try:
        path = _output()
        info["diagnostic_file"] = path
        flush()  # committed before importing RLPy, PySide2, or anymove.
        print("ANYMOVE_DIAG_START", BUILD, path, flush=True)

        def load_rlpy():
            import RLPy
            info["rlpy_module"] = str(getattr(RLPy, "__file__", "<embedded>"))
            return RLPy

        rlpy = stage("import_rlpy", load_rlpy)
        if rlpy is not None:
            def selected():
                items = list(rlpy.RScene.GetSelectedObjects() or [])
                info["selected_count"] = len(items)
                info["selected_names"] = [_safe(item.GetName) for item in items]
                if len(items) != 1:
                    raise RuntimeError("select exactly one rigged character")
                return items[0]

            obj = stage("selected_object", selected)
            if obj is not None:
                def skin_bones():
                    skel = obj.GetSkeletonComponent()
                    if skel is None:
                        raise RuntimeError("GetSkeletonComponent() returned None")
                    bones = list(skel.GetSkinBones() or [])
                    info["skin_bone_count"] = len(bones)
                    if not bones:
                        raise RuntimeError("GetSkinBones() returned zero bones")
                    return skel, bones

                bundle = stage("skin_bone_enumeration", skin_bones)
                if bundle is not None:
                    skel, bones = bundle

                    def collect_raw():
                        for index, bone in enumerate(bones):
                            entry = {
                                "index": index,
                                "name": _safe(bone.GetName),
                                "native_id": _safe(bone.GetID),
                                "python_proxy_type": type(bone).__name__,
                            }
                            try:
                                parent = bone.GetParent()
                                entry["parent_name"] = _safe(parent.GetName) if parent is not None else None
                                entry["parent_native_id"] = (
                                    _safe(parent.GetID) if parent is not None else None)
                            except Exception as exc:
                                entry["parent_error"] = "%s: %s" % (type(exc).__name__, exc)
                            info["bone_rows"].append(entry)
                        dupes = collections.Counter(row["native_id"] for row in info["bone_rows"])
                        info["duplicate_scene_id_groups"] = {
                            k: v for k, v in dupes.items() if v > 1
                        }
                        return True

                    stage("raw_bone_capture", collect_raw)

                    def module_paths():
                        # Avoid relying on the running plugin package in any
                        # earlier step. This import happens AFTER raw evidence.
                        if root and root not in sys.path:
                            sys.path.insert(0, root)
                        import iclone_adapter
                        import iclone_adapter.bone_registry as registry
                        import iclone_adapter.rlpy_reader as reader
                        info["module_paths"] = {
                            "package": getattr(iclone_adapter, "__file__", ""),
                            "bone_registry": getattr(registry, "__file__", ""),
                            "reader": getattr(reader, "__file__", ""),
                            "package_version": getattr(iclone_adapter, "__version__", ""),
                        }
                        return registry, reader

                    modules = stage("import_adapter", module_paths)
                    if modules is not None:
                        registry, reader = modules

                        def test_matrix():
                            info["matrix_layout"] = reader.probe_matrix_layout(rlpy)
                            return True

                        stage("matrix_layout_probe", test_matrix)

                        def test_registry():
                            entries, stats = registry.build_bone_registry(bones)
                            info["registry_stats"] = stats
                            info["registry_count"] = len(entries)
                            info["registry_paths_sample"] = [
                                entry["path"] for entry in entries[:10]
                            ]
                            return True

                        stage("bone_registry", test_registry)

                        def test_full_export():
                            data = reader.snapshot(obj, rlpy=rlpy)
                            skeleton_file = os.path.join(os.path.dirname(path),
                                                        "anymove_iclone_skeleton.json")
                            with open(skeleton_file, "w", encoding="utf-8") as f:
                                json.dump(data, f, ensure_ascii=False, indent=2)
                            info["skeleton_file"] = skeleton_file
                            info["export_bone_count"] = len(data["bones"])
                            return True

                        stage("full_skeleton_export", test_full_export)
        info["status"] = "completed_with_errors" if info["errors"] else "completed"
    except Exception as exc:
        info["status"] = "internal_error"
        info["errors"].append({
            "stage": "standalone_runner",
            "exception": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        try:
            flush()
        except Exception as exc:
            print("ANYMOVE_DIAG_WRITE_FAILURE", repr(exc), flush=True)
        print("ANYMOVE_DIAG_DONE", info.get("status"), path, flush=True)

    try:
        from PySide2 import QtWidgets
        status = info.get("status", "unknown")
        summary = ("AnyMove standalone P0 diagnostic: %s\n\n"
                   "Build: %s\nReport: %s\n\n"
                   "Please upload the JSON report." % (status, BUILD, path))
        QtWidgets.QMessageBox.information(
            None, "AnyMove P0 diagnostic", summary)
    except Exception as exc:
        print("ANYMOVE_DIAG_UI_UNAVAILABLE", repr(exc), flush=True)
    return info


run()
