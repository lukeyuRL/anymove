"""V2 human synthetic motion families, extending V1 without changing its contract.

New family names and additional finger/toe variations remain deterministic.
Procedural GT is not physically accurate mocap or natural human gait.
"""
import math

from tools import human_v3_fullbody_motion as v1

FINGER_NAMES = v1.FINGER_NAMES
BASE = v1.BASE
finger_tip_names = v1.finger_tip_names
toe_tip_names = v1.toe_tip_names
FAMILIES = v1.FAMILIES + (
    "finger_independent_roll",
    "heel_raise_turn",
    "cross_step_grip",
    "squat_palm_open",
)
BASE_FOR_NEW = {
    "finger_independent_roll": "balance_point",
    "heel_raise_turn": "turn_and_step",
    "cross_step_grip": "lateral_step_spread",
    "squat_palm_open": "crouch_fist",
}

def pose_changes(family, seed, frame, frame_count=107):
    if family not in FAMILIES:
        raise ValueError("Unsupported V2 motion family: " + str(family))
    base = BASE_FOR_NEW.get(family, family)
    delta, offset = v1.pose_changes(base, seed, frame, frame_count)
    delta = {name: list(v) for name, v in delta.items()}
    u = frame / (frame_count-1)
    envelope = math.sin(math.pi*u)**2
    phase = (seed % 43)*.11
    omega = 2*math.pi*(.68+(seed%19)*.043)*frame/24 + phase
    d = math.sin(omega)
    q = math.cos(omega*.82+.61)
    def turn(bone, axis, angle):
        vals = delta.setdefault(bone, [0.0, 0.0, 0.0])
        vals[axis] += float(angle)
        if abs(vals[axis]) > 1.7:
            raise RuntimeError("Excessive V2 pose rotation for "+bone)
    # Seed-specific, alternating motions for every hand digit; the V1
    # baseline already animates all phalanges. These give distinct finger
    # articulation dynamics, not mere whole-hand translation.
    for si, side in enumerate(("L", "R")):
        for digit_i, digit in enumerate(FINGER_NAMES):
            wave = math.sin(omega*(1.2+digit_i*.11)+digit_i*.63+si*.57)
            for segment in (1,2,3):
                name="CC_Base_"+side+"_"+digit+str(segment)
                mag=(.24,.18,.12)[segment-1]
                turn(name,0,envelope*mag*wave)
                if segment==1:
                    turn(name,2,envelope*.042*(digit_i-2)*math.cos(omega+si))
        # Independent ankle/toe motion, with the main V1 leg swing retained.
        turn("CC_Base_"+side+"_ToeBase",0,envelope*.09*math.sin(2*omega+si*1.77))
        turn("CC_Base_"+side+"_Foot",0,envelope*.07*math.sin(omega+.45+si*math.pi))
    if family=="finger_independent_roll":
        for si,side in enumerate(("L","R")):
            turn("CC_Base_"+side+"_Upperarm",2,(1 if si==0 else -1)*.25*envelope)
            turn("CC_Base_"+side+"_Hand",2,(1 if si==0 else -1)*.13*envelope*q)
            for k,digit in enumerate(FINGER_NAMES):
                turn("CC_Base_"+side+"_"+digit+"1",0,.40*envelope*math.sin(omega+k*.92+si))
    elif family=="heel_raise_turn":
        for side in ("L","R"):
            turn("CC_Base_"+side+"_Foot",0,.16*envelope*(.5+.5*d))
            turn("CC_Base_"+side+"_ToeBase",0,-.12*envelope*(.5+.5*d))
        turn("CC_Base_Waist",1,.15*envelope*q)
        turn("CC_Base_Spine01",1,.10*envelope*q)
    elif family=="cross_step_grip":
        for si,side in enumerate(("L","R")):
            turn("CC_Base_"+side+"_Thigh",2,(-1 if si==0 else 1)*.22*envelope*q)
            turn("CC_Base_"+side+"_Calf",0,.13*envelope*(.5+.5*q))
            for digit in FINGER_NAMES:
                turn("CC_Base_"+side+"_"+digit+"1",0,.22*envelope*(.5+.5*d))
    elif family=="squat_palm_open":
        for side in ("L","R"):
            turn("CC_Base_"+side+"_Thigh",0,-.16*envelope)
            turn("CC_Base_"+side+"_Calf",0,.25*envelope)
            turn("CC_Base_"+side+"_Upperarm",2,(1 if side=="L" else -1)*.22*envelope)
            for digit in FINGER_NAMES:
                turn("CC_Base_"+side+"_"+digit+"1",0,-.25*envelope*(.5+.5*q))
    delta_root=(.011*envelope*math.sin(.88*omega),
                .009*envelope*math.sin(omega+.67),
                .006*envelope*(.5+.5*q))
    root=tuple(float(offset[i]+delta_root[i]) for i in range(3))
    return delta, root
