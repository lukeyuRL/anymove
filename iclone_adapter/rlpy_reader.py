"""Read the selected iClone 8/9 Skin Bones without changing scene state.

The runtime probes the RLPy matrix translation convention before converting to
column-vector matrices. Current pose snapshots are NOT skin bind matrices.
This module imports RLPy lazily to support tests in stock Python.
"""
import json

from .schema import VERSION, ContractError, skeleton_hash, validate_skeleton


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


def probe_matrix_layout(rlpy):
    """Measure whether RMatrix4 stores translation in row 3 or column 3.

    The provided Legacy RLPy wrapper documents M(row,col), MakeIdentity,
    SetTranslate and RVector3, but does not specify storage convention. Probe
    with a non-zero synthetic translation rather than trusting a guess.
    """
    try:
        mat = rlpy.RMatrix4()
        mat.MakeIdentity()
        mat.SetTranslate(rlpy.RVector3(13.0, 17.0, 19.0))
        vals = [float(mat.M(r, c)) for r in range(4) for c in range(4)]
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
    return [float(mat.M(c, r) if layout == "transpose" else mat.M(r, c))
            for r in range(4) for c in range(4)]


def snapshot(obj, rlpy=None):
    if rlpy is None:
        import RLPy as rlpy
    matrix_layout = probe_matrix_layout(rlpy)
    skel = _get_skeleton(obj)
    nodes = list(skel.GetSkinBones() or [])
    if not nodes:
        raise ContractError("no accessible Skin Bones")
    node_ids = [str(node.GetID()) for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ContractError("duplicate scene bone IDs")
    ids = set(node_ids)
    # Explicitly require unique IDs in a snapshot; parent may be an object,
    # so only parents contained in GetSkinBones are traversed.
    records = {}
    for node, key in zip(nodes, node_ids):
        parent = node.GetParent()
        p = str(parent.GetID()) if parent is not None else None
        if p not in ids:
            p = None
        records[key] = {
            "id": key, "name": str(node.GetName()), "parent_id": p,
            "snapshot_local_matrix": _native_matrix(node.LocalTransform(), matrix_layout),
            "snapshot_world_matrix": _native_matrix(node.WorldTransform(), matrix_layout),
        }

    def path(key, stack=None):
        stack = set() if stack is None else stack
        if key in stack:
            raise ContractError("cycle in skin bones")
        b = records[key]
        parent = b["parent_id"]
        # IDs ensure that duplicate names are not confused during this scene.
        label = b["name"].replace("/", "%2F") + "#" + key
        return path(parent, stack | {key}) + "/" + label if parent else "/" + label

    for key, b in records.items():
        b["path"] = path(key)
        try:
            b["can_animate"] = bool(skel.IsBoneAnimated(nodes[node_ids.index(key)]))
        except (AttributeError, TypeError, RuntimeError):
            b["can_animate"] = None
        b["rest_local_matrix"] = None

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
        "bones": sorted(records.values(), key=lambda b: b["path"]),
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
