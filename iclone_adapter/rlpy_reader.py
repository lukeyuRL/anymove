"""Read the selected iClone 8/9 Skin Bones without changing scene state.

The runtime probes the RLPy matrix translation convention before converting to
column-vector matrices. Current pose snapshots are NOT skin bind matrices.
This module imports RLPy lazily to support tests in stock Python.
"""
import json

from .schema import VERSION, ContractError, skeleton_hash, validate_skeleton
from .bone_registry import build_bone_registry


def _get_skeleton(obj):
    fn = getattr(obj, "GetSkeletonComponent", None)
    if fn is None:
        raise ContractError("selected object has no GetSkeletonComponent")
    skel = fn()
    if skel is None:
        raise ContractError("selected object has no skeleton component")
    return skel


def selected_object(rlpy):
    selection = list(rlpy.RScene.GetSelectedObjects() or [])
    if len(selection) != 1:
        raise ContractError("select exactly one iClone Avatar or rigged Prop")
    return selection[0]


def _read_matrix_rows(mat):
    """Read RMatrix4 through RVector4.__getitem__ (returns float).

    Legacy RMatrix4.M(row, col) has a SWIG float& overload and may return
    SwigPyObject rather than a Python number. GetRow()->RVector4[i] has the
    explicitly documented float return type and avoids raw pointers.
    """
    return [[float(mat.GetRow(row)[col]) for col in range(4)]
            for row in range(4)]


def probe_matrix_layout(rlpy):
    """Measure whether RMatrix4 stores translation in row 3 or column 3.

    The provided Legacy RLPy wrapper documents GetRow()->RVector4[i],
    MakeIdentity, SetTranslate and RVector3. Probe
    with a non-zero synthetic translation rather than trusting a guess.
    """
    try:
        mat = rlpy.RMatrix4()
        mat.MakeIdentity()
        mat.SetTranslate(rlpy.RVector3(13.0, 17.0, 19.0))
        vals = [x for row in _read_matrix_rows(mat) for x in row]
    except Exception as exc:
        raise ContractError("RLPy matrix layout probe failed: %s" % exc)
    expected = (13.0, 17.0, 19.0)
    row = (vals[12], vals[13], vals[14])
    col = (vals[3], vals[7], vals[11])
    is_row = all(abs(a-b) < 1e-5 for a,b in zip(row, expected))
    is_col = all(abs(a-b) < 1e-5 for a,b in zip(col, expected))
    if is_row == is_col:
        raise ContractError("RLPy matrix translation convention is ambiguous")
    return "transpose" if is_row else "direct"


def _native_matrix(transform, layout):
    """Convert an RMatrix4 to canonical row-major/column-vector format."""
    if layout not in ("transpose", "direct"):
        raise ContractError("unverified native matrix layout")
    mat = transform.Matrix()
    rows = _read_matrix_rows(mat)
    return [rows[c][r] if layout == "transpose" else rows[r][c]
            for r in range(4) for c in range(4)]


def snapshot(obj, rlpy=None):
    if rlpy is None:
        import RLPy as rlpy
    matrix_layout = probe_matrix_layout(rlpy)
    skel = _get_skeleton(obj)
    registry, identity_diagnostics = build_bone_registry(skel.GetSkinBones() or [])
    records = []
    for entry in registry:
        node = entry["node"]
        record = {
            "id": entry["id"],
            "path": entry["path"],
            "name": entry["name"],
            "parent_id": entry["parent_id"],
            "native_scene_id": entry["native_scene_id"],
            "snapshot_local_matrix": _native_matrix(node.LocalTransform(), matrix_layout),
            "snapshot_world_matrix": _native_matrix(node.WorldTransform(), matrix_layout),
            "rest_local_matrix": None,
        }
        try:
            record["can_animate"] = bool(skel.IsBoneAnimated(node))
        except (AttributeError, TypeError, RuntimeError):
            record["can_animate"] = None
        records.append(record)

    data = {
        "schema_version": VERSION,
        "source": {"application": "iClone", "object_name": str(obj.GetName()),
                   "object_type": str(obj.GetType())},
        "coordinates": {"matrix_convention": "column_vector_row_major",
                        "raw_unit": "iClone_native_unverified", "meters_per_unit": None,
                        "native_matrix_layout": matrix_layout,
                        "world_up": "+Z", "world_forward": "-Y"},
        "reference": {"kind": "current_snapshot",
                      "warning": "NOT a bind/rest pose; iClone GetBoneTPosePosition provides position only."},
        "bones": sorted(records, key=lambda b: b["path"]),
        "identity_diagnostics": identity_diagnostics,
    }
    validate_skeleton(data)
    data["skeleton_hash"] = skeleton_hash(data)
    return data


def export_selected(path, rlpy=None):
    if rlpy is None:
        import RLPy as rlpy
    data = snapshot(selected_object(rlpy), rlpy=rlpy)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def capability_report(obj):
    """Non-mutating SDK capability report. Do not create or alter clips."""
    skel = _get_skeleton(obj)
    result = {"object": str(obj.GetName()), "skin_bones": len(skel.GetSkinBones() or []),
              "api": {name: callable(getattr(skel, name, None)) for name in (
                  "AddClip", "GetClip", "GetClipCount", "GetBoneTPosePosition",
                  "IsBoneAnimated", "BakeFkToIk", "SampleMotionClip")}}
    try:
        result["existing_clip_count"] = int(skel.GetClipCount())
    except (AttributeError, TypeError, RuntimeError):
        result["existing_clip_count"] = None
    return result
