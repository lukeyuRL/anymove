"""Read the selected iClone 8/9 Skin Bones without changing scene state.

No Blender assumptions: RLPy matrices are transposed to a column-vector
contract. Current pose transforms are snapshots, NOT skin bind matrices.
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


def _native_matrix(transform):
    """Transpose RMatrix4's row-vector matrix into 16 column-vector floats."""
    mat = transform.Matrix()
    return [float(mat.M(c, r)) for r in range(4) for c in range(4)]


def snapshot(obj):
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
            "snapshot_local_matrix": _native_matrix(node.LocalTransform()),
            "snapshot_world_matrix": _native_matrix(node.WorldTransform()),
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
    data = snapshot(selected_object(rlpy))
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
