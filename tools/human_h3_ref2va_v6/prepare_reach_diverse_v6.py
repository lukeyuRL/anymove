"""Generate separate Human V6 Reach motion source and immutable sample plan.

Exactly 16 new training clips + 4 new heldout, 107 frames/24fps.
All use same Human SkinMax V3 source rig, reusing original V2 render
engine. Four new 3D motion families: FORWARD, DIAGONAL, HIGH, BACKWARD.
No fal paid jobs. Original V2/V5 unaffected.

Do not equate these procedural controlled motions with natural human mocap.
"""
import copy,hashlib,json,pathlib
ROOT=pathlib.Path(r"F:\anymove")
HERE=ROOT/"runs"/"human"/"lora_synth_A"
SOURCE_SCRIPT=HERE/"render_human_conditioned_v2.py"
NEW_SCRIPT=HERE/"render_reach_diverse_v6.py"
V6=HERE/"v6_reach_source"
MODES={
 "forward":("reach_forward",44,66),
 "diagonal":("reach_diagonal",48,67),
 "high":("reach_high",45,70),
 "backward":("reach_backward",40,60)}
CAPTIONS={
 "reach_forward":"extends the {side} arm FORWARD towards the person's chest-facing direction, while the other arm stays lowered; then returns to neutral",
 "reach_diagonal":"raises the {side} arm FORWARD and OUTWARD diagonally away from the torso, then lowers it while the other arm remains stationary",
 "reach_high":"raises the {side} arm UPWARD and OUTWARD above shoulder level, then lowers it; the other arm stays stationary",
 "reach_backward":"extends the {side} arm BACKWARD behind the torso, then returns to neutral; the other arm remains stationary"}
def change(s,needle,replacement):
 if s.count(needle)!=1:raise RuntimeError("Patch anchor must be unique "+needle[:130]+" occurrences="+str(s.count(needle)))
 return s.replace(needle,replacement)
script=SOURCE_SCRIPT.read_text(encoding="utf8")
script=change(script,'FAMILIES=(\n    "lean_left","lean_right","lean_alternating",\n    "reach_left","reach_right","root_yaw_left","root_yaw_right")',
 '''FAMILIES=(
    "lean_left","lean_right","lean_alternating",
    "reach_left","reach_right","root_yaw_left","root_yaw_right",
    "reach_forward_left","reach_forward_right",
    "reach_diagonal_left","reach_diagonal_right",
    "reach_high_left","reach_high_right",
    "reach_backward_left","reach_backward_right")''')
insert='''    "reach_forward_left":"the person extends the anatomical LEFT arm FORWARD toward the front of their body while the RIGHT arm remains lowered, then returns to neutral",
    "reach_forward_right":"the person extends the anatomical RIGHT arm FORWARD toward the front of their body while the LEFT arm remains lowered, then returns to neutral",
    "reach_diagonal_left":"the person extends the anatomical LEFT arm FORWARD and OUTWARD diagonally while the RIGHT arm remains lowered, then returns to neutral",
    "reach_diagonal_right":"the person extends the anatomical RIGHT arm FORWARD and OUTWARD diagonally while the LEFT arm remains lowered, then returns to neutral",
    "reach_high_left":"the person raises the anatomical LEFT arm UPWARD and OUTWARD above shoulder level while the RIGHT arm remains lowered, then returns to neutral",
    "reach_high_right":"the person raises the anatomical RIGHT arm UPWARD and OUTWARD above shoulder level while the LEFT arm remains lowered, then returns to neutral",
    "reach_backward_left":"the person extends the anatomical LEFT arm BACKWARD behind their torso while the RIGHT arm remains lowered, then returns to neutral",
    "reach_backward_right":"the person extends the anatomical RIGHT arm BACKWARD behind their torso while the LEFT arm remains lowered, then returns to neutral",
'''
script=change(script,'    "root_yaw_right":"the person slowly pivots their entire body about a vertical world axis toward their own RIGHT, returning to neutral; no walking or stepping",',insert+'    "root_yaw_right":"the person slowly pivots their entire body about a vertical world axis toward their own RIGHT, returning to neutral; no walking or stepping",')
script=change(script,'    elif family.startswith("reach_"):\n        side="L" if family=="reach_left" else "R"',
                    '    elif family in ("reach_left","reach_right"):\n        side="L" if family=="reach_left" else "R"')
newcase='''    elif family.startswith("reach_"):
        # Verified on the immutable Human source via Blender V6 axis probe:
        # shoulder LOCAL +X moves hand WORLD -Y (forward),
        # shoulder LOCAL -X moves hand WORLD +Y (backward),
        # shoulder LOCAL +/-Z moves actor L/R arm outward and up.
        # Opposite side remains fixed; procedural timing has two
        # distinct shapes, not simply different amplitude/seed noise.
        typ=family.split("_")[1]
        side="L" if family.endswith("_left") else "R"
        sign=1.0 if side=="L" else -1.0
        u=frame/(count-1)
        motion=hold_curve(u,start=0.0,
                          rise=(.16 if seed%2 else .36),
                          fall=(.56 if seed%2 else .76),end=1.0)
        value=math.radians(angle_deg)*motion
        weights={
            "forward":(1.0,.10),
            "diagonal":(.70,.66),
            "high":(.10,1.0),
            "backward":(-.82,.30)}
        if typ not in weights:raise ValueError("Unknown Reach V6 type "+typ)
        forward,lateral=weights[typ]
        put(f"CC_Base_{side}_Upperarm",0,forward*value)
        put(f"CC_Base_{side}_Upperarm",2,sign*lateral*value)
        put(f"CC_Base_{side}_Forearm",0,(.10 if typ!="backward" else -.06)*motion)
        put(f"CC_Base_{side}_Hand",2,sign*.06*motion)
'''
script=change(script,'    elif family.startswith("root_yaw_"):',newcase+'    elif family.startswith("root_yaw_"):')
# Preserve generator data type/contract. Source planned in new folder only.
compile(script,str(NEW_SCRIPT),"exec")
if NEW_SCRIPT.exists() and NEW_SCRIPT.read_text(encoding="utf8")!=script:
 raise FileExistsError("Existing V6 source renderer differs; refuse overwrite")
NEW_SCRIPT.write_text(script,encoding="utf8")
plan=[]
for typ,(_,lo,hi) in MODES.items():
 for side in ("left","right"):
  for n,angle in enumerate((lo,hi)):
   seed=7301+n
   family=f"reach_{typ}_{side}"
   stem=f"full_train_reach_{typ}_{side}_a{angle}_s{seed}"
   plan.append({"split":"train","family":family,"seed":seed,"amplitude_deg":angle,
                "id":stem,"path":str(V6/stem)})
chall=[
 ("reach_forward_left",55,8201),("reach_diagonal_right",58,8202),
 ("reach_high_left",62,8203),("reach_backward_right",52,8204)]
for family,angle,seed in chall:
 stem=f"full_heldout_{family}_a{angle}_s{seed}"
 plan.append({"split":"heldout","family":family,"seed":seed,"amplitude_deg":angle,
              "id":stem,"path":str(V6/stem)})
assert len(plan)==20 and sum(x["split"]=="train" for x in plan)==16
assert len({x["id"] for x in plan})==20
V6.mkdir(parents=True,exist_ok=True)
outfile=V6/"v2_dataset_plan.json"
record={
 "schema":"anymove.human_reach_diverse_v6_source_plan/1",
 "status":"PLANNED_RENDER_NOT_YET_ACCEPTED",
 "target_train_clips":16,"target_heldout_clips":4,
 "complete_train_clips":16,"complete_heldout_clips":4,
 "source_55bone_human_blend_immutable":True,
 "source_V2_and_H3_V5_unchanged":True,
 "new_paid_jobs":0,
 "procedural_motion_not_natural_mocap":True,
 "results":plan}
if outfile.exists() and json.loads(outfile.read_text(encoding="utf8"))!=record:
 raise RuntimeError("V6 dataset plan exists and differs")
outfile.write_text(json.dumps(record,indent=2),encoding="utf8")
print(json.dumps({"renderer":str(NEW_SCRIPT),"plan":str(outfile),
   "new_train":16,"new_heldout":4,"new_types":list(MODES),
   "new_paid_jobs":0},indent=2))
