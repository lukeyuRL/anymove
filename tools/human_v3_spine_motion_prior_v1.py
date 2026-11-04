"""Spine/Root motion-prior GT source for a single V3 Nose humanoid.

Purpose: expand synthetic 24 FPS local *rotation* and root-turn coverage,
not increase repeated single-frame Reader PNGs. Every take animates all
30 finger phalanges, legs, wrists and core. 12 meaningfully distinct
families include yaw turns, pitch bends, lateral lean and counter-rotation.

All motion is deterministic and PROCEDURAL, NOT clinical joint limits,
human motion capture, physical contact or a learned prior.
"""
from __future__ import annotations

import math

from tools import human_v3_fullbody_motion_v2 as v2

FPS=24
FRAMES=107
FAMILIES=(
    "pivot_turn_left",
    "pivot_turn_right",
    "walk_turn_left",
    "walk_turn_right",
    "waist_twist_left",
    "waist_twist_right",
    "lateral_bend_left",
    "lateral_bend_right",
    "reach_counterrotate",
    "squat_and_rotate",
    "balance_turn",
    "lean_recover",
)
BASE=v2.BASE
FINGER_NAMES=v2.FINGER_NAMES
finger_tip_names=v2.finger_tip_names
toe_tip_names=v2.toe_tip_names

SOURCE_V2=(
    "turn_and_step","turn_and_step","march_open_close","lateral_step_spread",
    "balance_point","finger_independent_roll",
    "lunge_reach","lunge_reach","lunge_reach","squat_palm_open",
    "balance_point","cross_step_grip",
)


def _smooth(u:float)->float:
    """C1 zero-velocity ends, useful for a turn without sudden Root jerk."""
    return u*u*(3.0-2.0*u)


def pose_changes(family:str,seed:int,frame:int,frame_count:int=FRAMES):
    """Return (local_delta_by_bone, Root XYZ offset in meters, Root XYZ Euler rad).

    The Root XYZ angles are WORLD-space Yaw(Z), Pitch(X), Roll(Y), applied
    to Blender Armature.matrix_world BEFORE the non-uniform-free base rest.
    """
    if family not in FAMILIES:
        raise ValueError("Unknown Spine dataset motion family: "+str(family))
    if (not isinstance(seed,int) or seed<0 or not isinstance(frame,int)
        or frame<0 or frame>=frame_count or frame_count<40):
        raise ValueError("Invalid Spine motion seed/frame")
    idx=FAMILIES.index(family)
    delta0,offset=v2.pose_changes(SOURCE_V2[idx],seed,frame,frame_count)
    delta={name:list(values) for name,values in delta0.items()}
    u=frame/(frame_count-1)
    bell=math.sin(math.pi*u)**2
    period=math.sin(2*math.pi*u)
    phase=(seed%47)*.019
    second=math.sin(4*math.pi*u+phase)
    freq_scale=.82+.06*(seed%7)
    side=-1. if ("left" in family) else 1.
    amp=.86+.03*(seed%9)
    yaw=0.
    pitch=0.
    roll=0.
    # These families intentionally exercise different ROOT angular
    # trajectories so a statistical prior must NOT anchor yaw to 0.
    if family.startswith("pivot_turn"):
        yaw=side*math.radians(52)*amp*bell
        pitch=.036*bell*math.sin(2*math.pi*u+phase)
        roll=.035*period
    elif family.startswith("walk_turn"):
        yaw=side*math.radians(65)*amp*_smooth(u)
        pitch=.025*bell*second
        roll=.025*bell*math.cos(2*math.pi*u)
    elif family.startswith("waist_twist"):
        yaw=side*math.radians(17)*amp*bell
        roll=.04*period
    elif family.startswith("lateral_bend"):
        yaw=side*math.radians(11)*amp*bell
        roll=side*.14*bell
        pitch=.055*period
    elif family=="reach_counterrotate":
        yaw=.26*amp*period
        pitch=.045*bell*math.cos(4*math.pi*u)
        roll=.08*bell
    elif family=="squat_and_rotate":
        yaw=.32*amp*bell
        pitch=.10*bell
        roll=.03*period
    elif family=="balance_turn":
        yaw=-.37*amp*bell
        roll=.045*bell*math.sin(2*math.pi*u)
    elif family=="lean_recover":
        yaw=.21*amp*period
        pitch=.13*bell
        roll=-.055*period
    else:
        raise AssertionError("Unreachable motion family")

    # Body must cover active torsional Y, sagittal X and lateral Z axes.
    # Distribute turn across root (global), waist and two spine bones.
    mid=math.sin(2*math.pi*u+phase)
    rootsign=(1 if yaw>=0 else -1)
    torso_yaw=(.13+.045*idx/len(FAMILIES))*amp*bell*side
    if family.startswith("waist_twist"):
        torso_yaw=side*.40*amp*bell
    elif family=="reach_counterrotate":
        torso_yaw=-.27*amp*period
    elif family=="squat_and_rotate":
        torso_yaw=.21*amp*bell
    torso_pitch=.12*amp*bell*mid if "lateral" not in family else .17*amp*bell
    if family=="lean_recover":
        torso_pitch=.29*amp*bell
    torso_roll=.12*amp*bell*math.cos(2*math.pi*u+phase)
    if family.startswith("lateral_bend"):
        torso_roll=side*.27*amp*bell
    def turn(name,axis,delta_angle):
        if not math.isfinite(delta_angle):
            raise ValueError("Nonfinite local angular position")
        v=delta.setdefault(name,[0.,0.,0.])
        v[axis]+=float(delta_angle)
        if abs(v[axis])>1.7:
            raise RuntimeError("Excessively rotated bone "+name)

    turn("CC_Base_Pelvis",1,torso_yaw*.12)
    turn("CC_Base_Pelvis",2,torso_roll*.10)
    turn("CC_Base_Waist",0,torso_pitch*.65)
    turn("CC_Base_Waist",1,torso_yaw*.52)
    turn("CC_Base_Waist",2,torso_roll*.55)
    turn("CC_Base_Spine01",0,torso_pitch*.40)
    turn("CC_Base_Spine01",1,torso_yaw*.33)
    turn("CC_Base_Spine01",2,torso_roll*.33)
    turn("CC_Base_Spine02",0,torso_pitch*.19)
    turn("CC_Base_Spine02",1,torso_yaw*.18)
    turn("CC_Base_Spine02",2,torso_roll*.28)
    turn("CC_Base_NeckTwist01",1,-.16*torso_yaw)
    turn("CC_Base_NeckTwist02",1,-.12*torso_yaw)
    turn("CC_Base_Head",1,-.11*torso_yaw)
    # Subtle shoulder compensation with matched arm motions.
    for si,side_name in enumerate(("L","R")):
        sign=1. if si==0 else -1.
        turn("CC_Base_"+side_name+"_Clavicle",0,sign*.025*bell*mid)
        turn("CC_Base_"+side_name+"_Clavicle",2,sign*.03*bell*math.cos(2*math.pi*u))

    # World motion stays small, to isolate rotations from large translations.
    root_offset=(float(offset[0]+.035*bell*period),
                 float(offset[1]+.023*bell*math.cos(2*math.pi*u)),
                 float(offset[2]+.003*bell))
    root_xyz=(float(pitch),float(roll),float(yaw))
    if not all(math.isfinite(x) for x in root_offset+root_xyz):
        raise ValueError("Nonfinite Spine / Root motion")
    return delta,root_offset,root_xyz


def all_unique_clips():
    """Strictly balanced Train/Validation/Test by entire independent take."""
    rows=[]
    for family_index,family in enumerate(FAMILIES):
        for take in range(12):
            split="train" if take<8 else ("validation" if take<10 else "test")
            seed=400000+family_index*2003+take*719
            rows.append({"clip":f"spine_{split}_{family}_s{seed}",
                         "family":family,"seed":seed,
                         "take":take,"split":split})
    if len(rows)!=144 or len({r["seed"] for r in rows})!=144:
        raise RuntimeError("Spine dataset must have 144 distinct independent seeds")
    return rows
