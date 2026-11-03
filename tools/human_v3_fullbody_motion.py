"""Deterministic synthetic whole-body Human V3 Nose poses, including all fingers.

The motion is explicitly procedural, NOT captured human gait. Every generated
clip animates both lower and upper body, torso/head and finger phalanges.
Family and seed parameters provide clip-disjoint training/validation/test
without reusing one clip's frames in a different split.
"""
import math

FINGER_NAMES=("Pinky","Ring","Mid","Index","Thumb")
FAMILIES=(
    "march_open_close",
    "high_knee_grasp",
    "lateral_step_spread",
    "crouch_fist",
    "lunge_reach",
    "turn_and_step",
    "kick_and_wave",
    "balance_point",
)
BASE={"CC_Base_L_Upperarm":(0,0,-1.10),
      "CC_Base_R_Upperarm":(0,0,+1.10)}
def pose_changes(family,seed,frame,frame_count=107):
    if family not in FAMILIES:
        raise ValueError("Unknown human procedural family: "+str(family))
    u=frame/(frame_count-1)
    env=math.sin(math.pi*u)**2
    freq=.78+(seed%11)*.048
    phase=(seed%29)*.13
    w=2*math.pi*freq*(frame/24)+phase
    s=math.sin(w)
    c=math.cos(w)
    q=math.sin(w+1.11)
    delta={}
    def turn(name,ax,rot):
        if not math.isfinite(rot):raise ValueError("Nonfinite pose")
        if abs(rot)>1.8:raise ValueError("Excessive local rotation")
        vals=delta.setdefault(name,[0.,0.,0.])
        vals[ax]+=float(rot)
    def joint(side,part,ax,rot):
        turn("CC_Base_"+side+"_"+part,ax,rot)
    # Every sequence has active arms, head, pelvis/torso, both legs, hands
    # and each of the 30 finger phalange bones.
    k=FAMILIES.index(family)
    leg_amp=(.34,.64,.26,.38,.48,.32,.58,.44)[k]
    arm_amp=(.32,.25,.30,.44,.55,.30,.41,.34)[k]
    torso_amp=(.09,.08,.12,.14,.13,.18,.09,.16)[k]
    for side,sign,ph in (("L",1,0.),("R",-1,math.pi)):
        cycle=math.sin(w+ph)
        lift=max(0.,math.sin(w+ph+.22))
        # Local X thigh raises/lifts forward, calf flex compensates foot;
        # mirror side phase for alternating steps.
        thigh=env*(leg_amp*cycle+.10*max(0.,cycle))
        calf=env*(.22+.42*lift)*(1 if k!=6 else 1.15)
        joint(side,"Thigh",0,thigh)
        joint(side,"Calf",0,calf)
        joint(side,"Foot",0,-.26*calf-.11*thigh)
        joint(side,"ToeBase",0,.12*env*lift)
        # Small lateral step component, mirrored by side.
        joint(side,"Thigh",2,sign*env*(.07+.13*(k==2))*q)
        joint(side,"Upperarm",0,arm_amp*env*cycle)
        joint(side,"Upperarm",2,sign*env*(.22+.35*(k in (4,6,7)))*(0.5+0.5*c))
        joint(side,"Forearm",0,env*(.25+.17*lift))
        joint(side,"Hand",0,env*.13*math.sin(2*w+ph))
        joint(side,"Hand",2,sign*env*.12*math.cos(w+ph))
        # Animate 30 finger joints, not only the knuckle or hand object.
        for digit_idx,digit in enumerate(FINGER_NAMES):
            rhythm=(math.sin(w*1.5 + ph + digit_idx*.45)+1)/2
            flex=env*(.25+.56*rhythm)
            if k==3:
                flex=env*(.58+.18*rhythm)
            elif k==7 and digit=="Index":
                flex=env*(.06+.08*rhythm)  # pointing index while other fingers curl
            elif k==2:
                flex=env*(.12+.27*rhythm)
            for segment in (1,2,3):
                root="CC_Base_"+side+"_"+digit+str(segment)
                turn(root,0,(.52,.40,.26)[segment-1]*flex)
                if segment==1:
                    # Subtle splay, mirrored across the midline.
                    turn(root,2,sign*env*(digit_idx-2)*.025*math.sin(w+.2))
    turn("CC_Base_Waist",1,torso_amp*env*s)
    turn("CC_Base_Spine01",1,.70*torso_amp*env*math.sin(w+.27))
    turn("CC_Base_Spine02",2,.50*torso_amp*env*q)
    turn("CC_Base_NeckTwist01",0,.04*env*s)
    turn("CC_Base_NeckTwist02",1,.045*env*q)
    turn("CC_Base_Head",1,.11*env*math.sin(w+.37))
    # Upper-body emphasis in coordinated reaching/lunging.
    if k==4:
        turn("CC_Base_L_Upperarm",2,.37*env)
        turn("CC_Base_R_Upperarm",2,-.25*env)
    if k==6:
        turn("CC_Base_R_Upperarm",2,-.38*env)
    if k==3:
        turn("CC_Base_L_Thigh",0,-.10*env)
        turn("CC_Base_R_Thigh",0,-.10*env)
        turn("CC_Base_L_Calf",0,.15*env)
        turn("CC_Base_R_Calf",0,.15*env)
    # Sway and modest bob in world metres; floor clearance is added after FK.
    root_x=.045*env*math.sin(w+.43)
    root_y=.028*env*math.cos(w+.71)
    root_z=.02*env*(.5+.5*math.cos(2*w))
    return delta,(root_x,root_y,root_z)

def finger_tip_names():
    return ["CC_Base_"+side+"_"+digit+"Tip" for side in ("L","R") for digit in FINGER_NAMES]

def toe_tip_names():
    return ["CC_Base_"+side+"_ToeTip" for side in ("L","R")]
