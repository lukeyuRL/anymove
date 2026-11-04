"""Fit the rig to the joints read from both views: one pose per frame, smooth over time, inside the joint limits.

python -m mocap.solve DETECTIONS.npz rig_info.json OUT.json
Before fitting, the side view's left and right legs are checked against the front view: a joint has the same height
in both halves, so if swapping a pair of legs in the side view makes the heights agree better, the side reader
mixed up the near and far leg and they are swapped back. The same height test gives the sync score: how far apart
the two halves disagree, and the time lag between them that agrees best.

The fit runs Blender's forward kinematics in torch (pose = parent pose x rest offset x Euler XYZ), with the pelvis
free to turn and move, every other bone limited to its range, and minimises the confidence-weighted distance between
the rig's projected joints and the readings in both views, plus smoothness and a small pull toward the rest pose.
"""
import json, math, sys

import numpy as np
import torch

from mocap import layout
from mocap.camera_geometry import triangulate_orthographic, verify_record_angles
from mocap.confidence import bounded_peak_weights

LEGS = {"F": ["scap", "shoulder", "elbow", "wrist", "fpaw", "ftoe"], "H": ["hip", "knee", "hock", "hpaw", "htoe"]}


def leg_pairs(names):
    """Index pairs (left, right) of every leg joint, grouped per front/hind pair."""
    out = {}
    for k, stems in LEGS.items():
        pairs = []
        for s in stems:
            if f"{s}_L" in names:
                pairs.append((names.index(f"{s}_L"), names.index(f"{s}_R")))
        out[k] = pairs
    return out


def fix_swaps(d, names, margin=4.0):
    """Swap near/far legs in the side view where the heights say the reader mixed them up."""
    fv, sv = d["front_uv"][..., 1], d["side_uv"][..., 1]
    c = np.minimum(d["front_conf"], d["side_conf"]).clip(0, 1)
    swapped = {}
    for k, pairs in leg_pairs(names).items():
        L, R = np.array(pairs).T
        same = (c[:, L] * np.abs(sv[:, L] - fv[:, L]) + c[:, R] * np.abs(sv[:, R] - fv[:, R])).sum(1)
        swap = (c[:, L] * np.abs(sv[:, R] - fv[:, L]) + c[:, R] * np.abs(sv[:, L] - fv[:, R])).sum(1)
        gain = (same - swap) / len(pairs)
        gain = np.array([np.median(gain[max(0, t - 2):t + 3]) for t in range(len(gain))])
        do = gain > margin
        for t in np.where(do)[0]:
            d["side_uv"][t, L], d["side_uv"][t, R] = d["side_uv"][t, R].copy(), d["side_uv"][t, L].copy()
            d["side_conf"][t, L], d["side_conf"][t, R] = d["side_conf"][t, R].copy(), d["side_conf"][t, L].copy()
        swapped[k] = int(do.sum())
    return swapped


def height_features(uv, conf, names):
    """Per-frame heights that do not care about left/right labels: middle joints as they are, each leg pair as its
    mean height and its height difference. Returns (T, K) features and (T, K) weights."""
    v = uv[..., 1]
    feats, w = [], []
    for i, n in enumerate(names):
        if not n.endswith(("_L", "_R")):
            feats.append(v[:, i]); w.append(conf[:, i])
        elif n.endswith("_L"):
            a, b = i, names.index(n[:-2] + "_R")
            c = np.minimum(conf[:, a], conf[:, b])
            feats += [(v[:, a] + v[:, b]) / 2, np.abs(v[:, a] - v[:, b])]
            w += [c, c]
    return np.stack(feats, 1), np.stack(w, 1).clip(0, 1)


def align_views(d, names, band=48, step_penalty=1.0):
    """Video models often play the two halves out of step (one view runs ahead). Match the halves frame to frame
    with dynamic time warping on joint heights, which both views share, and resample the front view onto the side
    view's timeline. Returns how far each side frame's match was moved, in frames."""
    F, Fw = height_features(d["front_uv"], d["front_conf"], names)
    S, Sw = height_features(d["side_uv"], d["side_conf"], names)
    T = len(F)
    C = np.full((T, T), np.inf)
    for i in range(T):
        lo, hi = max(0, i - band), min(T, i + band + 1)
        w = np.minimum(Fw[i][None], Sw[lo:hi])
        C[i, lo:hi] = (w * np.abs(F[i][None] - S[lo:hi])).sum(1) / w.sum(1).clip(1e-6)
    D = np.full((T, T), np.inf)
    D[0, 0] = C[0, 0]
    for i in range(T):
        for j in range(max(0, i - band), min(T, i + band + 1)):
            if i == j == 0:
                continue
            best = min(D[i - 1, j - 1] if i and j else np.inf, D[i - 1, j] + step_penalty if i else np.inf,
                       D[i, j - 1] + step_penalty if j else np.inf)
            D[i, j] = C[i, j] + best
    i, j, path = T - 1, T - 1, [(T - 1, T - 1)]
    while i or j:
        cands = [(D[i - 1, j - 1], i - 1, j - 1) if i and j else (np.inf, 0, 0),
                 (D[i - 1, j], i - 1, j) if i else (np.inf, 0, 0), (D[i, j - 1], i, j - 1) if j else (np.inf, 0, 0)]
        _, i, j = min(cands)
        path.append((i, j))
    path = np.array(path[::-1])
    match = np.array([path[path[:, 1] == j, 0].mean() for j in range(T)])  # front frame for each side frame
    match = np.convolve(np.pad(match, 3, mode="edge"), np.ones(7) / 7, mode="valid")  # no stair steps
    t = np.arange(T)
    for k in ("front_uv", "front_conf"):
        a = d[k].reshape(T, -1)
        d[k] = np.stack([np.interp(match, t, a[:, c]) for c in range(a.shape[1])], 1).reshape(d[k].shape)
    return match - t


def feet():
    """The joints whose heights make the sync score: creature.json "feet" when MOCAP_CREATURE is set, else the wolf's."""
    import os, pathlib
    cfg = pathlib.Path(os.environ.get("MOCAP_CREATURE", "")) / "creature.json"
    if os.environ.get("MOCAP_CREATURE") and cfg.exists():
        return json.load(open(cfg)).get("feet", [])
    return ["fpaw_L", "fpaw_R", "hpaw_L", "hpaw_R"]


def paw_agreement(d, names):
    """Correlation of each paw's height between the two views over the clip: +1 when both halves show the same
    footfalls, near 0 or negative when the video model animated the views differently."""
    out = {}
    for k in feet():
        if k in names:
            j = names.index(k)
            a, b = d["front_uv"][:, j, 1], d["side_uv"][:, j, 1]
            out[k] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0.5 and b.std() > 0.5 else None
    vals = [v for v in out.values() if v is not None]
    out["score"] = float(np.mean(vals)) if vals else None
    return out


def sync_report(d):
    """Mean height disagreement between the halves (layout px) and the lag (frames) that minimises it."""
    fv, sv = d["front_uv"][..., 1], d["side_uv"][..., 1]
    c = np.minimum(d["front_conf"], d["side_conf"]).clip(0, 1)
    err = (c * np.abs(fv - sv)).sum(1) / c.sum(1).clip(1e-6)
    best = None
    T = len(fv)
    for lag in range(-min(12,T-1), min(12,T-1)+1):
        a = fv[max(0, lag):T + min(0, lag)]
        b = sv[max(0, -lag):T + min(0, -lag)]
        e = np.abs(a - b).mean()
        if best is None or e < best[1]:
            best = (lag, e)
    return {"mean_height_gap_px": float(err.mean()), "worst_frame_gap_px": float(err.max()),
            "per_frame_gap_px": err.round(2).tolist(), "best_lag_frames": best[0], "gap_at_best_lag_px": float(best[1])}


def euler_xyz(a):
    """Blender Euler XYZ (applied X, then Y, then Z) to rotation matrices: Rz @ Ry @ Rx."""
    x, y, z = a.unbind(-1)
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    o, l = torch.zeros_like(x), torch.ones_like(x)
    Rx = torch.stack([l, o, o, o, cx, -sx, o, sx, cx], -1).view(*x.shape, 3, 3)
    Ry = torch.stack([cy, o, sy, o, l, o, -sy, o, cy], -1).view(*x.shape, 3, 3)
    Rz = torch.stack([cz, -sz, o, sz, cz, o, o, o, l], -1).view(*x.shape, 3, 3)
    return Rz @ Ry @ Rx


def axis_angle(v):
    th = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = v / th
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, dtype=v.dtype, device=v.device) + th.sin() * K + (1 - th.cos()) * K @ K


class Rig:
    def __init__(self, info, dev="cpu"):
        self.names = [b["name"] for b in info["bones"]]
        self.parent = [b["parent"] for b in info["bones"]]
        B = torch.tensor([b["matrix_local"] for b in info["bones"]], dtype=torch.float64, device=dev)
        self.B = B
        self.L = torch.tensor([b["length"] for b in info["bones"]], dtype=torch.float64, device=dev)
        # rest offset of each bone in its parent's frame
        self.off = torch.stack([B[i] if p < 0 else torch.linalg.inv(B[p]) @ B[i] for i, p in enumerate(self.parent)])
        lim = np.radians(np.array([b["limits_deg"] for b in info["bones"]], float))
        self.lo = torch.tensor(lim[..., 0], device=dev)
        self.hi = torch.tensor(lim[..., 1], device=dev)
        self.kp = [(k["bone"], k["end"]) for k in info["keypoints"]]
        self.kp_names = [k["name"] for k in info["keypoints"]]
        self.rest_kp = torch.tensor([k["rest"] for k in info["keypoints"]], dtype=torch.float64, device=dev)

    def forward(self, trans, root_rot, eul):
        """trans (T,3) world offset, root_rot (T,3,3) pelvis basis, eul (T,nb,3) Euler for every bone (root ignored)."""
        T, nb = eul.shape[:2]
        R = euler_xyz(eul)
        P = [None] * nb
        for i, p in enumerate(self.parent):
            basis = torch.zeros(T, 4, 4, dtype=eul.dtype, device=eul.device)
            basis[:, :3, :3] = root_rot if p < 0 else R[:, i]
            basis[:, 3, 3] = 1
            if p < 0:
                M = self.off[i].expand(T, 4, 4) @ basis
                M = M.clone()
                M[:, :3, 3] += trans
            else:
                M = P[p] @ self.off[i] @ basis
            P[i] = M
        P = torch.stack(P, 1)
        joints = []
        for b, end in self.kp:
            if end == "head":
                joints.append(P[:, b, :3, 3])
            else:
                joints.append(P[:, b, :3, 1] * self.L[b] + P[:, b, :3, 3])
        return torch.stack(joints, 1), P


def project(J):
    """Torch (and differentiable) counterpart to the shared Blender/NumPy projection."""
    v = layout.H / 2 - (J[..., 2] - layout.Z_CENTER) / layout.M_PER_PX
    f_angle = math.radians(layout.FRONT_AZIMUTH_DEG)
    s_angle = math.radians(layout.SIDE_AZIMUTH_DEG)
    f = torch.stack([layout.W_FRONT / 2 + (math.cos(f_angle) * J[..., 0] - math.sin(f_angle) * J[..., 1]) / layout.M_PER_PX, v], -1)
    s = torch.stack([layout.W_SIDE / 2 + (math.cos(s_angle) * J[..., 0] - math.sin(s_angle) * J[..., 1]) / layout.M_PER_PX, v], -1)
    return f, s


def effective_front_height_weight(value=None):
    """Preserve legacy Front-height downweight; treat oblique slots symmetrically."""
    if value is not None:
        return float(value)
    return (0.25 if layout.FRONT_AZIMUTH_DEG == 0.0
            and layout.SIDE_AZIMUTH_DEG == 90.0 else 1.0)


def triangulate(d, front_height_weight=None):
    """Calibrated 2-view XYZ reconstruction; no hardcoded Front X / Side Y."""
    front_height_weight = effective_front_height_weight(front_height_weight)
    return triangulate_orthographic(d["front_uv"], d["side_uv"], d["front_conf"], d["side_conf"],
                                   front_width=layout.W_FRONT, side_width=layout.W_SIDE,
                                   height=layout.H, metres_per_pixel=layout.M_PER_PX,
                                   z_center=layout.Z_CENTER,
                                   front_angle_deg=layout.FRONT_AZIMUTH_DEG,
                                   side_angle_deg=layout.SIDE_AZIMUTH_DEG,
                                   front_height_weight=front_height_weight)


def apply_temporal_joint_guard(d, names, frame_indices, calibration):
    """Opt-in HUMAN 67-marker P1 guard before the FK fitting stage.

    Returns new 2D observations and confidence factors. Unreliable derived
    points receive downweighted evidence; skeleton lengths remain physically
    enforced later by the rig's FK optimizer. Never replaces the raw det.npz.
    This is not a generic creature tracker: the explicit calibration must
    match 315/45, 672px and the exact input marker names.
    """
    from mocap.temporal_skeleton_tracker import track_two_view
    required=("front_uv","side_uv","front_conf","side_conf")
    if any(k not in d for k in required):
        raise ValueError("Missing two-view Reader observations")
    if not isinstance(names,(list,tuple)) or len(names)!=67:
        raise ValueError("P1 human guard requires exactly 67 named joints")
    obs={k:np.asarray(d[k]).copy() for k in required}
    result=track_two_view(obs["front_uv"],obs["side_uv"],
                          frame_indices,calibration,names)
    repaired={k:v.copy() for k,v in obs.items()}
    repaired["front_uv"]=result["front_uv"]
    repaired["side_uv"]=result["side_uv"]
    repaired["front_conf"]=(np.clip(obs["front_conf"],0,1)*
                            result["front_reliability"]).clip(0,1)
    repaired["side_conf"]=(np.clip(obs["side_conf"],0,1)*
                           result["side_reliability"]).clip(0,1)
    report={
        "schema":"anymove.human_p1_temporal_joint_guard/1",
        "camera_azimuth_deg":calibration["camera_azimuth_deg"],
        "frames":len(frame_indices),
        "markers":len(names),
        "source_blend_sha256":calibration["source_blend_sha256"],
        "train_calibration_clip_count":calibration["calibration_clip_count"],
        "derived_joint_confidence_is_heuristic":True,
        "source_detections_preserved":True,
        "tracker_diagnostics":result["diagnostics"],
        "sparse_flags_by_frame":[{
            "frame":int(frame_indices[t]),
            "front_inferred":[names[j] for j in np.flatnonzero(result["front_inferred"][t])],
            "side_inferred":[names[j] for j in np.flatnonzero(result["side_inferred"][t])],
            "skeleton_untrusted":[names[j] for j in np.flatnonzero(result["skeleton_untrusted"][t])],
        } for t in range(len(frame_indices)) if (
            result["front_inferred"][t].any() or
            result["side_inferred"][t].any() or
            result["skeleton_untrusted"][t].any())],
    }
    return repaired,report


def rotation_only(M):
    """Orthonormal factor of a Blender rest basis, excluding uniform object scale."""
    U,_,Vt=np.linalg.svd(M)
    D=np.diag([1.,1.,float(np.sign(np.linalg.det(U@Vt)))])
    return U@D@Vt


def root_from_triangulation(tri, conf, rig, names, info):
    """Per-frame root rotation (as the basis of the root bone, axis-angle) from the points that ride rigidly on it
    (its two ends and the heads of the bones hung straight off it), by weighted Kabsch against the rest pose;
    smoothed over time."""
    bones = info["bones"]
    ride = {bones[0]["head_joint"], bones[0]["tail_joint"]} | {b["head_joint"] for b in bones if b["parent"] == 0}
    body = sorted(names.index(n) for n in ride if n in names)
    h = names.index(bones[0]["head_joint"])
    rest = rig.rest_kp.cpu().numpy()
    a = rest[body] - rest[h]
    Brot = rotation_only(rig.B[0, :3, :3].cpu().numpy())
    if len(body)<3 or np.linalg.matrix_rank(a, tol=1e-7)<2:
        # The CC human synthetic RL_BoneRoot has only one physical child;
        # its axis is not observable from two rest points. Avoid an unstable
        # arbitrary Kabsch spin, allowing the data term to solve orientation.
        return np.zeros((len(tri),3),dtype=np.float64)
    out = []
    for t in range(len(tri)):
        b = tri[t, body] - tri[t, h]
        w = conf[t, body]
        U, _, Vt = np.linalg.svd((a * w[:, None]).T @ b)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        W = Vt.T @ D @ U.T
        R = Brot.T @ W @ Brot
        ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        out.append(axis / max(np.linalg.norm(axis), 1e-8) * ang if ang > 1e-6 else np.zeros(3))
    out = np.array(out)
    k = np.ones(5) / 5
    return np.stack([np.convolve(np.pad(out[:, i], 2, mode="edge"), k, mode="valid") for i in range(3)], 1)


def euler_from_matrix(R):
    """Inverse of euler_xyz for one 3x3 matrix (Blender XYZ: R = Rz Ry Rx)."""
    b = -np.arcsin(np.clip(R[2, 0], -1, 1))
    return np.array([np.arctan2(R[2, 1], R[2, 2]), b, np.arctan2(R[1, 0], R[0, 0])])


def rot_between(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def ik_init(tri, root_aa, info, rig):
    """Point every bone at its triangulated child joint, down the hierarchy, with no twist and inside its limits:
    a starting pose close to the answer, so the fit does not have to find a sit or a rear from a standing pose."""
    names = rig.kp_names
    bones = info["bones"]
    off = rig.off.cpu().numpy()
    Brot = rotation_only(rig.B[0, :3, :3].cpu().numpy())
    lo, hi = rig.lo.cpu().numpy(), rig.hi.cpu().numpy()
    T, nb = len(tri), len(bones)
    eul = np.zeros((T, nb, 3))
    rest=rig.rest_kp.cpu().numpy()
    rest_rot=[rotation_only(b[:3,:3]) for b in rig.B.cpu().numpy()]
    for t in range(T):
        aa = root_aa[t]
        ang = np.linalg.norm(aa)
        k = aa / ang if ang > 1e-9 else np.array([1.0, 0, 0])
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        W = [None] * nb
        W[0] = Brot @ (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K)
        for i in range(1, nb):
            p = bones[i]["parent"]
            base = W[p] @ off[i][:3, :3]
            if bones[i]["head_joint"] not in names or bones[i]["tail_joint"] not in names:
                W[i] = base  # e.g. fur bones, whose root is not a tracked joint: start at rest
                continue
            d = tri[t, names.index(bones[i]["tail_joint"])] - tri[t, names.index(bones[i]["head_joint"])]
            if np.linalg.norm(d) < 1e-6:
                W[i] = base
                continue
            head=names.index(bones[i]["head_joint"])
            tail=names.index(bones[i]["tail_joint"])
            rest_vec=rest[tail]-rest[head]
            if np.linalg.norm(rest_vec)<1e-8:
                W[i]=base
                continue
            # Rest bone-to-marker vector is not always bone-local +Y,
            # especially on disconnected Character Creator limb chains.
            a=rest_rot[i].T@rest_vec
            a=a/np.linalg.norm(a)
            b=base.T@(d/np.linalg.norm(d))
            b=b/np.linalg.norm(b)
            R = rot_between(a,b)
            e = np.clip(euler_from_matrix(R), lo[i], hi[i])
            eul[t, i] = e
            W[i] = base @ euler_xyz(torch.tensor(e)).numpy()
    return eul


def time_normalized_joint_acceleration(J, frame_deltas):
    """Second position derivative in source-frame units, respecting sparse samples.

    For unit intervals this is the original J[t+1]-2*J[t]+J[t-1].
    For linearly moving joints it is zero at any positive irregular cadence.
    """
    dt=frame_deltas
    if len(J)<3:
        return J.new_zeros((0,)+tuple(J.shape[1:]))
    if dt.ndim!=1 or len(dt)!=len(J)-1 or torch.any(dt<=0):
        raise ValueError('Invalid source-frame deltas')
    v0=(J[1:-1]-J[:-2])/dt[:-1,None,None]
    v1=(J[2:]-J[1:-1])/dt[1:,None,None]
    return 2*(v1-v0)/(dt[:-1,None,None]+dt[1:,None,None])


def fit(d, info, iters=1500, dev="cpu", log=True, front_height_weight=None):
    front_height_weight = effective_front_height_weight(front_height_weight)
    rig = Rig(info, dev)
    names = rig.kp_names
    T = len(d["front_uv"])
    frame_idx=np.asarray(d.get("frame_indices",np.arange(T)),dtype=np.float64)
    if frame_idx.shape!=(T,) or not np.isfinite(frame_idx).all() or np.any(np.diff(frame_idx)<=0):
        raise ValueError("Solver frame_indices must increase monotonically")
    dt=torch.tensor(np.diff(frame_idx),dtype=torch.float64,device=dev).clamp_min(1e-6)
    human_mode=(info.get("schema")=="anymove.human_v3_nose_hr_v2_rig_solver_contract/1")
    f_obs = torch.tensor(d["front_uv"], dtype=torch.float64, device=dev)
    s_obs = torch.tensor(d["side_uv"], dtype=torch.float64, device=dev)
    f_c = torch.tensor(d["front_conf"], dtype=torch.float64, device=dev).clamp(0, 1) ** 2
    s_c = torch.tensor(d["side_conf"], dtype=torch.float64, device=dev).clamp(0, 1) ** 2
    tri_np = triangulate(d, front_height_weight)
    tri = torch.tensor(tri_np, dtype=torch.float64, device=dev)
    hips = names.index(info["bones"][0]["head_joint"])
    trans = (tri[:, hips] - rig.rest_kp[hips]).clone().requires_grad_(True)
    conf = np.minimum(d["front_conf"], d["side_conf"]).clip(1e-3, 1)
    root0 = root_from_triangulation(tri_np, conf, rig, names, info) + 1e-4
    root = torch.tensor(root0, dtype=torch.float64, device=dev).requires_grad_(True)
    nb = len(rig.names)
    # Euler = lo + (hi - lo) * sigmoid(phi), starting from the analytic IK pose
    e0 = torch.tensor(ik_init(tri_np, root0, info, rig), dtype=torch.float64, device=dev)
    z = ((e0 - rig.lo) / (rig.hi - rig.lo)).clamp(1e-3, 1 - 1e-3)
    phi = torch.logit(z).clone().requires_grad_(True)
    paws = [names.index(n) for n in names if any(n.startswith(p) for p in ("fpaw", "ftoe", "hpaw", "htoe"))]
    sigma = 6.0  # Human with good IK initialization uses 6px throughout; legacy anneals from 40px.
    acc_strength=0.002 if human_mode else 0.02
    vel_strength=0.1 if human_mode else 2.0
    prior_strength=0.005 if human_mode else 0.05
    observation_strength=4.0 if human_mode else 1.0
    # Original 0/90 Front height was experimentally downweighted. For arbitrary
    # oblique camera slots the default is symmetric, with confidence per view.
    f_axis = torch.tensor([1.0, front_height_weight], dtype=torch.float64, device=dev)

    def loss_fn(stage):
        nonlocal sigma
        eul = rig.lo + (rig.hi - rig.lo) * torch.sigmoid(phi)
        J, _ = rig.forward(trans, axis_angle(root), eul)
        pf, ps = project(J)
        rf = ((pf - f_obs) ** 2 * f_axis).sum(-1)
        rs = ((ps - s_obs) ** 2).sum(-1)
        data = observation_strength*((f_c * torch.log1p(rf / sigma ** 2)).sum() + (s_c * torch.log1p(rs / sigma ** 2)).sum())
        # Time-normalized derivatives: sparse 7-frame samples must not receive
        # 7^4 times larger acceleration penalty than contiguous frames.
        if T>=3:
            accel=time_normalized_joint_acceleration(J,dt)
            acc=(accel**2).sum()/layout.M_PER_PX**2*acc_strength
        else:
            acc=torch.zeros((),dtype=torch.float64,device=dev)
        vel=(((eul[1:]-eul[:-1])/dt[:,None,None])**2).sum()*vel_strength + (((root[1:]-root[:-1])/dt[:,None])**2).sum()*vel_strength
        prior=(eul**2).sum()*prior_strength
        ground = (torch.relu(-J[:, paws, 2]) ** 2).sum() / layout.M_PER_PX ** 2 * 0.5
        total = data + acc + vel + prior + ground
        return total, {"data": data.item(), "acc": acc.item(), "vel": vel.item(), "prior": prior.item(), "ground": ground.item()}, J, pf, ps, eul

    opt = torch.optim.Adam([{"params": [trans], "lr": 0.001 if human_mode else 0.01}, {"params": [root], "lr": 0.002 if human_mode else 0.03}, {"params": [phi], "lr": 0.005 if human_mode else 0.05}])
    best_human=None
    best_score=float('inf')
    for it in range(iters):
        sigma = 6.0 if human_mode else 6.0 + 34.0 * max(0.0, 1 - it / (0.6 * iters))
        if it == int(iters * 0.7):
            for g in opt.param_groups:
                g["lr"] *= 0.3
        total, parts, J, pf, ps, eul = loss_fn(it)
        if human_mode and bool(torch.isfinite(total)) and total.item()<best_score:
            best_score=float(total.item())
            best_human=(trans.detach().clone(),root.detach().clone(),phi.detach().clone())
        opt.zero_grad()
        total.backward()
        if human_mode:
            torch.nn.utils.clip_grad_norm_([trans,root,phi],max_norm=10.0)
        opt.step()
        if log and it % 250 == 0:
            print(f"  iter {it} loss {total.item():.1f}", {k: round(v, 1) for k, v in parts.items()}, flush=True)
    sigma = 6.0
    if human_mode:
        with torch.no_grad():
            total,_,_,_,_,_=loss_fn(iters)
            if bool(torch.isfinite(total)) and total.item()<best_score:
                best_score=float(total.item())
                best_human=(trans.detach().clone(),root.detach().clone(),phi.detach().clone())
            if best_human is None:
                raise RuntimeError('Human Solver did not find a finite optimization checkpoint')
            for variable,saved in zip((trans,root,phi),best_human):
                variable.copy_(saved)
    with torch.no_grad():
        total, parts, J, pf, ps, eul = loss_fn(iters)
        err_f = (pf - f_obs).norm(dim=-1)
        err_fx = (pf - f_obs)[..., 0].abs()
        err_s = (ps - s_obs).norm(dim=-1)
    return {"trans": trans.detach().cpu().numpy(), "root_rotmat": axis_angle(root).detach().cpu().numpy(),
            "euler": eul.detach().cpu().numpy(), "joints": J.cpu().numpy(),
            "reproj_px": {"front": float(err_f.mean()), "front_x_only": float(err_fx.mean()), "side": float(err_s.mean())},
            "bones": rig.names, "joint_names": names}


def to_quat(Rm):
    q = []
    for R in Rm:
        w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        x = np.copysign(np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
        y = np.copysign(np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
        z = np.copysign(np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
        q.append([w, x, y, z])
    return np.array(q)


def main():
    det_path, info_path, out = sys.argv[1:4]
    temporal_calibration_path=None
    iterations=1500
    if "--iters" in sys.argv[4:]:
        ipos=sys.argv.index("--iters")
        if ipos+1>=len(sys.argv):
            raise ValueError("--iters requires an integer")
        iterations=int(sys.argv[ipos+1])
        if not 1<=iterations<=10000:
            raise ValueError("--iters must be 1..10000")
    if "--temporal-calibration" in sys.argv[4:]:
        pos=sys.argv.index("--temporal-calibration")
        if pos+1>=len(sys.argv):
            raise ValueError("--temporal-calibration requires a JSON calibration file")
        temporal_calibration_path=sys.argv[pos+1]
    raw = np.load(det_path)
    d = {k: raw[k].copy() for k in ("front_uv", "side_uv", "front_conf", "side_conf")}
    # The reader decodes raw (unbounded) linear heatmap peaks. Weights for
    # optimization are bounded without rewriting or corrupting det.npz.
    for name in ("front_conf","side_conf"):
        d[name],peak_stats=bounded_peak_weights(d[name],name)
        print("Reader %s raw peak range %.4g..%.4g, clipped %d/%d "
              "to [0,1] for solver weights"%
              (name,peak_stats["raw_min"],peak_stats["raw_max"],
               peak_stats["clipped_samples"],peak_stats["total_samples"]),
              flush=True)
    info = json.load(open(info_path))
    if (info.get("schema")=="anymove.human_v3_nose_hr_v2_rig_solver_contract/1"
        and str(info.get("limits_status","")).startswith("PROVISIONAL")
        and "--allow-provisional-human-limits" not in sys.argv):
        raise ValueError("HUMAN_RIG_LIMITS_UNCALIBRATED: this V3 Nose export uses provisional broad Euler limits. "
                         "Use --allow-provisional-human-limits only for disposable research QA; "
                         "do not import it directly into iClone.")
    # Fail before optimization if imported rig and camera settings disagree.
    # Without MOCAP_CREATURE, mocap.layout silently defaults to Wolf.
    from mocap.layout_guard import verify_layout_for_rig
    active_layout=verify_layout_for_rig(info_path,layout)
    camera_angles=[layout.FRONT_AZIMUTH_DEG,layout.SIDE_AZIMUTH_DEG]
    observed_angles=raw["camera_azimuth_deg"] if "camera_azimuth_deg" in raw.files else None
    camera_provenance=verify_record_angles(observed_angles,camera_angles,"detections.npz")
    print("Solver camera layout:",active_layout,"camera provenance:",camera_provenance,flush=True)
    names = [k["name"] for k in info["keypoints"]]
    before = sync_report(d)
    paws = paw_agreement(d, names)
    print("paw height agreement between views:", {k: (None if v is None else round(v, 2)) for k, v in paws.items()})
    print(f"sync as generated: mean height gap {before['mean_height_gap_px']:.1f} px, worst {before['worst_frame_gap_px']:.1f} px, "
          f"best constant lag {before['best_lag_frames']} frames")
    shift = None
    if "--no-align" not in sys.argv:
        trial = {k: v.copy() for k, v in d.items()}
        s = align_views(trial, names)
        gain = 1 - sync_report(trial)["mean_height_gap_px"] / max(before["mean_height_gap_px"], 1e-6)
        if gain > 0.15:  # only re-time when the halves really were out of step
            d.update(trial)
            shift = s
            print(f"time-aligned the front view onto the side view: mean shift {np.abs(shift).mean():.1f} frames, "
                  f"largest {np.abs(shift).max():.1f} frames (height gap down {gain:.0%})")
        else:
            print(f"views already in step (re-timing would only cut the height gap by {gain:.0%}); left as generated")
    # painted legs make the reader's left/right labels trustworthy, so swaps are only fixed on request
    swapped = fix_swaps(d, names) if "--fix-swaps" in sys.argv else {}
    temporal_report=None
    if temporal_calibration_path is not None:
        # Never silently infer a sparse clip's frame spacing: motion speed
        # constraints are calibrated in source-frame units.
        if "frame_indices" not in raw.files:
            raise ValueError("P1 temporal guard requires frame_indices in detections.npz")
        with open(temporal_calibration_path,encoding="utf-8") as f:
            calibration=json.load(f)
        if (int(calibration["image_width"])!=layout.W_FRONT or
            int(calibration["image_height"])!=layout.H or
            int(calibration["image_width"])!=layout.W_SIDE or
            not np.isclose(float(calibration["metres_per_pixel"]),layout.M_PER_PX,rtol=1e-6)):
            raise ValueError("P1 calibration image dimensions/scale differ from active Solver layout")
        if "names" in raw.files and raw["names"].tolist()!=names:
            raise ValueError("P1 detections joint names/order differ from rig_info.json")
        guarded,temporal_report=apply_temporal_joint_guard(d,names,raw["frame_indices"],calibration)
        import hashlib
        from pathlib import Path
        temporal_report["calibration_sha256"]=hashlib.sha256(Path(temporal_calibration_path).read_bytes()).hexdigest()
        d.update(guarded)
        print("P1 temporal/skeleton guard:",temporal_report["tracker_diagnostics"],flush=True)
    sync = sync_report(d)
    print("side-view leg swaps fixed (frames):", swapped)
    print(f"sync after: mean height gap {sync['mean_height_gap_px']:.1f} px, worst {sync['worst_frame_gap_px']:.1f} px")
    sync["before"] = {k: v for k, v in before.items() if k != "per_frame_gap_px"}
    sync["paw_agreement"] = paws
    sync["front_shift_frames"] = None if shift is None else shift.round(2).tolist()
    if "frame_indices" in raw.files:
        d["frame_indices"]=raw["frame_indices"].copy()
    res = fit(d, info, iters=iterations, dev="cuda" if torch.cuda.is_available() else "cpu")
    print("reprojection error px:", {k: round(v, 2) for k, v in res["reproj_px"].items()})
    # Preserve the exact detection file and camera Layout provenance.
    # These two values prevent the old "same 124 frames / same 50 names"
    # ambiguity when a user later diagnoses an incorrect motion.
    import hashlib
    det_sha = hashlib.sha256()
    with open(det_path, "rb") as source_det:
        for chunk in iter(lambda: source_det.read(1048576), b""):
            det_sha.update(chunk)
    motion = {"fps": float(raw["fps"]), "bones": res["bones"], "trans": res["trans"].round(5).tolist(),
              "root_quat": to_quat(res["root_rotmat"]).round(6).tolist(), "euler": res["euler"].round(5).tolist(),
              "joints": res["joints"].round(4).tolist(), "joint_names": res["joint_names"],
              "reproj_px": res["reproj_px"], "sync": sync, "swaps": swapped,
              "solver_camera_layout": active_layout,
              "solver_camera_azimuth_deg": camera_angles,
              "source_detections_sha256": det_sha.hexdigest(),
              "source_frame_indices": (raw["frame_indices"].astype(int).tolist() if "frame_indices" in raw.files else None),
              "source_frames_contiguous": bool("frame_indices" not in raw.files or np.all(np.diff(raw["frame_indices"])==1)),
              "temporal_joint_tracking":temporal_report,
              "solver_rig_limits_status":info.get("limits_status"),
              "motion_space_warning":"BLENDER_LOCAL_EULER_NOT_ICLONE_FK_CHANNELS"}
    json.dump(motion, open(out, "w"))
    print("saved", out)


if __name__ == "__main__":
    main()
