import json,pathlib
root=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\conditioned_backend\fal_h3_ref2va\remote\h3_ref2va_v6_28_fourview_400step\heldout_ab_v1")
source=root/"forward_left"/"inputs"/"identity.json"
target=root/"backward_right"/"inputs"/"identity.json"
p=json.loads(source.read_text(encoding="utf8"))
b=json.loads((root/"backward_right"/"plan.json").read_text(encoding="utf8"))
assert p["source_sha256"]==b["reference_image_sha256"]
assert p["url"].startswith("https://")
if target.exists():
 assert json.loads(target.read_text(encoding="utf8"))==p
else:
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(json.dumps(p,indent=2),encoding="utf8")
print("REUSED_APPROVED_IDENTICAL_FOURCAM_IDENTITY_REFERENCE",p["source_sha256"],flush=True)
