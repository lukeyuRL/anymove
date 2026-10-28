"""Resolve generic Skin Bone identities without assuming RINode.GetID() is unique.

In iClone, GetSkinBones() can contain multiple bones sharing a scene GetID().
The stable-ish export key is a hierarchy path, not the raw scene identifier.
For duplicate bone names, parent chain disambiguates the path; duplicate
siblings are numbered in the order returned by GetSkinBones(). If the SDK
does not provide enough information to identify a parent uniquely, fail
rather than silently constructing an incorrect rig.
"""
from collections import Counter, defaultdict
from urllib.parse import quote

from .schema import ContractError


def _lineage(node):
    """Name chain from scene root to node, including non-skin helper nodes."""
    names, seen_ids, retained_proxies = [], set(), []
    for _ in range(512):
        if node is None:
            return tuple(reversed(names))
        name = str(node.GetName())
        names.append(name)
        # Retain each proxy: otherwise CPython may reuse an id() of a
        # freshly returned transient SWIG wrapper, yielding a FALSE cycle.
        proxy_id = id(node)
        if proxy_id in seen_ids:
            raise ContractError("cycle while traversing RINode parents")
        seen_ids.add(proxy_id)
        retained_proxies.append(node)
        node = node.GetParent()
    raise ContractError("bone ancestry exceeds 512 nodes")


def build_bone_registry(nodes):
    nodes = list(nodes)
    if not nodes:
        raise ContractError("no accessible Skin Bones")
    names = [str(b.GetName()) for b in nodes]
    scene_ids = [str(b.GetID()) for b in nodes]
    if any(not n for n in names):
        raise ContractError("Skin Bone has an empty name")
    names_to_index = defaultdict(list)
    scene_id_to_index = defaultdict(list)
    for i, (name, scene_id) in enumerate(zip(names, scene_ids)):
        names_to_index[name].append(i)
        scene_id_to_index[scene_id].append(i)

    # Only compute expensive lineages when a duplicate name requires it.
    lineage_cache = {}

    def lineage(index):
        if index not in lineage_cache:
            lineage_cache[index] = _lineage(nodes[index])
        return lineage_cache[index]

    def resolve(proxy):
        # Exact Python proxy identity if a parent is returned from the same
        # object instance. SWIG normally returns a different wrapper.
        for i, original in enumerate(nodes):
            if proxy is original:
                return i
        # SDK's native equality can sometimes compare the underlying pointer.
        matched = []
        for i, original in enumerate(nodes):
            try:
                if proxy == original:
                    matched.append(i)
            except (TypeError, RuntimeError):
                continue
        if len(matched) == 1:
            return matched[0]
        sid = str(proxy.GetID())
        if len(scene_id_to_index.get(sid, ())) == 1:
            return scene_id_to_index[sid][0]
        name = str(proxy.GetName())
        choices = names_to_index.get(name, ())
        if len(choices) == 1:
            return choices[0]
        if not choices:
            return None
        chain = _lineage(proxy)
        choices = [i for i in choices if lineage(i) == chain]
        if len(choices) == 1:
            return choices[0]
        if choices:
            raise ContractError(
                "ambiguous duplicate parent bone %r (scene IDs are not unique); "
                "cannot reliably recover hierarchy" % name)
        # A proxy with a matching name but different parent chain is not
        # necessarily the matching Skin Bone.
        return None

    parents = []
    skipped_helpers = 0
    for index, node in enumerate(nodes):
        ancestor = node.GetParent()
        visited, found = set(), None
        for _ in range(512):
            if ancestor is None:
                break
            # Prevent accidental cyclic helper paths without trusting GetID.
            signature = _lineage(ancestor)
            if signature in visited:
                raise ContractError("cycle in skin bone helper hierarchy")
            visited.add(signature)
            candidate = resolve(ancestor)
            if candidate is not None:
                if candidate == index:
                    raise ContractError("Skin Bone cannot be its own parent")
                found = candidate
                break
            skipped_helpers += 1
            ancestor = ancestor.GetParent()
        else:
            raise ContractError("Skin Bone parent chain exceeds 512 nodes")
        parents.append(found)

    children = defaultdict(list)
    for i, parent in enumerate(parents):
        children[parent].append(i)
    names_by_parent = {}
    for parent, siblings in children.items():
        counts = Counter(names[i] for i in siblings)
        occurrences = defaultdict(int)
        for i in siblings:
            occurrences[names[i]] += 1
            label = quote(names[i], safe="")
            if counts[names[i]] > 1:
                label += "~%d" % occurrences[names[i]]
            names_by_parent[i] = label

    paths, active = {}, set()

    def path(index):
        if index in paths:
            return paths[index]
        if index in active:
            raise ContractError("cycle in reconstructed Skin Bone hierarchy")
        active.add(index)
        p = parents[index]
        result = (path(p) if p is not None else "") + "/" + names_by_parent[index]
        active.remove(index)
        paths[index] = result
        return result

    registry = []
    for i, node in enumerate(nodes):
        bp = path(i)
        registry.append({
            "id": "skin:" + bp,
            "path": bp,
            "name": names[i],
            "parent_id": "skin:" + path(parents[i]) if parents[i] is not None else None,
            "native_scene_id": scene_ids[i],
            "node": node,
            "index": i,
        })
    stats = {
        "native_id_unique_count": len(scene_id_to_index),
        "native_id_duplicate_groups": sum(len(v) > 1 for v in scene_id_to_index.values()),
        "native_id_duplicate_nodes": sum(len(v) for v in scene_id_to_index.values() if len(v) > 1),
        "duplicate_name_groups": sum(len(v) > 1 for v in names_to_index.values()),
        "skipped_non_skin_ancestors": skipped_helpers,
        "identity_strategy": "hierarchy_path",
        "note": "GetID is diagnostic only; identical sibling names use skin list order.",
    }
    return registry, stats
