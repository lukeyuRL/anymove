"""Versioned, dependency-free skeleton contract and matrix utilities.

All serialized matrices are row-major arrays of 16 floats and use COLUMN vectors:
a point p is transformed by M @ p, translation occupies indices 3,7,11.
The Reallusion RMatrix4 uses row-vector notation and must be transposed on import.
No unverified snapshot is ever silently labelled as the bind/rest pose.
"""
import hashlib
import json
import math

VERSION = "anymove.iclone.skeleton/1"
IDENTITY = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]


class ContractError(ValueError):
    pass


def matrix(value):
    if not isinstance(value, (list, tuple)) or len(value) != 16:
        raise ContractError("matrix must contain 16 values")
    out = [float(v) for v in value]
    if not all(math.isfinite(v) for v in out):
        raise ContractError("matrix must be finite")
    if any(abs(out[i] - want) > 1e-5 for i, want in [(12, 0), (13, 0), (14, 0), (15, 1)]):
        raise ContractError("matrix must be affine (column-vector convention)")
    return out


def mul(a, b):
    a, b = matrix(a), matrix(b)
    return [sum(a[4*r+k] * b[4*k+c] for k in range(4))
            for r in range(4) for c in range(4)]


def inverse(a):
    """Gauss-Jordan inverse, including invertible affine scale/shear."""
    a = matrix(a)
    rows = [[a[4*r+c] for c in range(4)] +
            [float(r == c) for c in range(4)] for r in range(4)]
    for j in range(4):
        pivot = max(range(j, 4), key=lambda r: abs(rows[r][j]))
        if abs(rows[pivot][j]) < 1e-12:
            raise ContractError("singular skeleton transform")
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [x / scale for x in rows[j]]
        for i in range(4):
            if i != j:
                factor = rows[i][j]
                rows[i] = [x - factor*y for x, y in zip(rows[i], rows[j])]
    return matrix([rows[r][c+4] for r in range(4) for c in range(4)])


def relative(parent_world, child_world):
    return mul(inverse(parent_world), child_world)


def world_matrices(bones, local_key="rest_local_matrix"):
    """General FK; bone list need not be sorted, and may have multiple roots."""
    lookup = {b["id"]: b for b in bones}
    if len(lookup) != len(bones):
        raise ContractError("duplicate bone id")
    output, visiting = {}, set()

    def resolve(key):
        if key in output:
            return output[key]
        if key in visiting:
            raise ContractError("cycle in skeleton hierarchy")
        b = lookup[key]
        visiting.add(key)
        local = matrix(b[local_key])
        parent = b.get("parent_id")
        result = mul(resolve(parent), local) if parent is not None else local
        visiting.remove(key)
        output[key] = result
        return result

    for key in lookup:
        resolve(key)
    return output


def validate_skeleton(data):
    if data.get("schema_version") != VERSION:
        raise ContractError("unsupported skeleton schema version")
    bones = data.get("bones")
    if not isinstance(bones, list) or not bones:
        raise ContractError("nonempty bones list required")
    ids, paths = set(), set()
    for b in bones:
        key, path = b.get("id"), b.get("path")
        if not isinstance(key, str) or not key or key in ids:
            raise ContractError("duplicate or invalid bone id")
        if not isinstance(path, str) or not path or path in paths:
            raise ContractError("duplicate or invalid bone path")
        ids.add(key)
        paths.add(path)
        matrix(b["snapshot_local_matrix"])
        matrix(b["snapshot_world_matrix"])
        rest = b.get("rest_local_matrix")
        if rest is not None:
            matrix(rest)
    for b in bones:
        p = b.get("parent_id")
        if p is not None and p not in ids:
            raise ContractError("orphan parent: " + str(p))
    world_matrices([{"id": b["id"], "parent_id": b.get("parent_id"),
                     "rest_local_matrix": b["snapshot_local_matrix"]} for b in bones])
    if data.get("reference", {}).get("kind") not in ("current_snapshot", "verified_rest"):
        raise ContractError("reference.kind must be current_snapshot or verified_rest")
    if data["reference"]["kind"] == "verified_rest":
        if any(b.get("rest_local_matrix") is None for b in bones):
            raise ContractError("verified_rest requires all rest_local_matrix")
    return True


def skeleton_hash(data):
    """Hash only in-scene hierarchy identity; unchanged when a pose is animated."""
    validate_skeleton(data)
    payload = {
        "bones": [{"id": b["id"], "path": b["path"],
                   "parent_id": b.get("parent_id")}
                  for b in sorted(data["bones"], key=lambda b: b["path"])],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
