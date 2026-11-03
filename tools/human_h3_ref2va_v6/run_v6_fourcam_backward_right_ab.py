"""Exactly two user-approved fal H3 Ref2VA Heldout Side Lean A/B videos.

A/B is a one-seed controlled experiment: one fal endpoint, same Pose Reference
video and identity reference image, same prompt/seed/duration/resolution/weights;
only LoRA scale changes from 0.0 to 1.0.

The submit operation is idempotent: records a marker before every paid POST.
An ambiguous network error leaves its marker; NEVER retry automatically.
collect/status cannot submit jobs. No additional AI videos or training.
"""
import copy
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import requests
from PIL import Image

ROOT=pathlib.Path(r"F:\anymove")
sys.path.insert(0,str(ROOT))
from mocap import fal
BASE=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"
PILOT=BASE/"remote"/"h3_ref2va_v6_28_fourview_400step"
PLAN=PILOT/"heldout_ab_v1"/"backward_right"/"plan.json"
WEIGHTS=PILOT/"model"/"human_v6_fourcam_h3_400.safetensors"
VALIDATION=PILOT/"MODEL_VALIDATION.json"
PILOT_RESPONSE=PILOT/"result.json"
OUT=PILOT/"heldout_ab_v1"/"backward_right"
ENDPOINT="minimax/h3/reference-to-video/lora"
FFPROBE=pathlib.Path(r"C:\ffmpeg\bin\ffprobe.exe")
CANDIDATE="full_heldout_reach_backward_right_a52_s8204"

def read(p):return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def save(p,record):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=pathlib.Path(str(p)+".tmp")
    tmp.write_text(json.dumps(record,indent=2),encoding="utf8")
    tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):h.update(part)
    return h.hexdigest()

def ffprobe(p):
    query=[str(FFPROBE),"-hide_banner","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,nb_frames,r_frame_rate:format=duration",
        "-of","json",str(p)]
    proc=subprocess.run(query,check=True,capture_output=True,text=True)
    data=json.loads(proc.stdout)
    s=data["streams"][0]
    result={
        "width":int(s["width"]),"height":int(s["height"]),
        "nb_frames":int(s["nb_frames"]),
        "fps":s["r_frame_rate"],
        "duration_s":float(data["format"]["duration"])
    }
    return result

def preflight():
    doc=read(PLAN)
    val=read(VALIDATION)
    lora=read(PILOT_RESPONSE)["lora_file"]
    if doc["fal_endpoint"]!=ENDPOINT or doc["candidate_heldout_case"]!=CANDIDATE:
        raise ValueError("Unexpected A/B endpoint or heldout case")
    if doc["matched_inference_requests_if_approved"]!=2:
        raise ValueError("Only 2 new paid A/B inference videos authorized")
    if val["status"]!="PASS_400STEP_MODEL_ARTIFACTS" or val["steps"]!=400 or val["config"]["training_type"]!="ref2va":
        raise ValueError("400-Step Ref2VA weights not validated")
    if sha(WEIGHTS)!=val["weights"]["sha256"]:
        raise ValueError("400-Step LoRA has changed after validation")
    model_url=lora["url"]
    if val["weights"]["bytes"]!=lora["file_size"]:
        raise ValueError("Model metadata length mismatch")
    image=pathlib.Path(doc["reference_image_local"])
    video=pathlib.Path(doc["reference_video_local"])
    if not image.is_file() or sha(image)!=doc["reference_image_sha256"]:
        raise ValueError("Heldout first-frame image hash mismatch")
    if not video.is_file() or sha(video)!=doc["reference_video_sha256"]:
        raise ValueError("Heldout Pose Reference video hash mismatch")
    with Image.open(image) as img:
        if img.size!=(896,896):raise ValueError("Expected identity reference 768x448")
    source_profile=ffprobe(video)
    if source_profile["width"]!=896 or source_profile["height"]!=896 or source_profile["nb_frames"]!=107 or source_profile["fps"]!="24/1":
        raise ValueError("Unexpected heldout pose video 107 frames/24fps/768x448")
    if not (4.0 < source_profile["duration_s"] < 5.0):
        raise ValueError("Heldout video must be 2-15 sec; expected ~4.5s")
    r0=copy.deepcopy(doc["requests"]["scale0"])
    r1=copy.deepcopy(doc["requests"]["scale1"])
    for label,request,scale in (("scale0",r0,0.0),("scale1",r1,1.0)):
        if request.get("prompt","").count("FOURCAM")!=1:
            raise ValueError("TWOCAM trigger missing/repeated")
        if request.get("seed")!=42 or request.get("duration")!=5 or request.get("resolution")!="768P" or request.get("aspect_ratio")!="1:1" or request.get("prompt_expansion_mode")!="disabled":
            raise ValueError("Unauthorized fal input parameters")
        if len(request.get("loras",[]))!=1 or request["loras"][0]["path"]!=model_url or request["loras"][0]["scale"]!=scale:
            raise ValueError("Wrong Ref2VA checkpoint or LoRA scale")
        if request.get("reference_image_urls")!=["<UPLOAD_SAME_IDENTITY_IMAGE_ONCE>"] or request.get("reference_video_urls")!=["<UPLOAD_SAME_POSE_VIDEO_ONCE>"]:
            raise ValueError("Reference placeholders are not the agreed data")
        allowed={"prompt","seed","duration","resolution","aspect_ratio","prompt_expansion_mode","loras","reference_image_urls","reference_video_urls"}
        if set(request)!=allowed:raise ValueError("Unexpected/unreviewed fal API fields")
    aa=copy.deepcopy(r0);bb=copy.deepcopy(r1);aa.pop("loras");bb.pop("loras")
    if aa!=bb:raise ValueError("Scale0/1 controls differ beyond LoRA scale")
    if not fal.key():raise ValueError("Fal API key is unavailable")
    info={
        "status":"PREFLIGHT_PASS",
        "endpoint":ENDPOINT,"model_400_sha256":sha(WEIGHTS),
        "case":CANDIDATE,"seed":42,"duration_s":5,"resolution":"768P",
        "source_reference_frames":source_profile["nb_frames"],
        "same_pose_video_and_identity":True,
        "paid_video_requests_authorized":2,
        "existing_requests":{label:((OUT/label/"submitted.json").exists()) for label in ("scale0","scale1")},
        "uncertain_submission_flags":{label:((OUT/label/"submission_in_progress.json").exists()) for label in ("scale0","scale1")},
        "uploaded_image":(OUT/"inputs"/"identity.json").exists(),
        "uploaded_video":(OUT/"inputs"/"pose.json").exists(),
    }
    return doc,info

def upload_once(filename,file,expected):
    path=OUT/"inputs"/filename
    if path.is_file():
        value=read(path)
        if value.get("source_sha256")!=expected or not value.get("url","").startswith("https://"):
            raise ValueError("Uploaded ref file has wrong hash or URL")
        return value["url"]
    url=fal.upload(file)
    if not url.startswith("https://"):raise ValueError("fal returned invalid uploaded reference URL")
    save(path,{"url":url,"source_sha256":expected,"local_file":str(file),"bytes":file.stat().st_size})
    print("UPLOADED_ONCE",filename,flush=True)
    return url

def submit():
    doc,info=preflight()
    print("MATCHED_REF2VA_AB_PREFLIGHT",json.dumps(info,indent=2),flush=True)
    # If either paid POST was attempted ambiguously, do not proceed to submit
    # anything else without reconciliation.
    for label in ("scale0","scale1"):
        if (OUT/label/"submission_in_progress.json").exists():
            raise RuntimeError("Ambiguous prior submission at "+label+"; refusing all new paid jobs")
    if all((OUT/lab/"submitted.json").exists() for lab in ("scale0","scale1")):
        print("BOTH_ALREADY_SUBMITTED_NO_PAID_DUPLICATE",flush=True);return
    image_file=pathlib.Path(doc["reference_image_local"])
    video_file=pathlib.Path(doc["reference_video_local"])
    image_url=upload_once("identity.json",image_file,doc["reference_image_sha256"])
    video_url=upload_once("pose.json",video_file,doc["reference_video_sha256"])
    for label,scale in (("scale0",0.0),("scale1",1.0)):
        folder=OUT/label
        submitted=folder/"submitted.json"
        marker=folder/"submission_in_progress.json"
        request=copy.deepcopy(doc["requests"][label])
        request["reference_image_urls"]=[image_url]
        request["reference_video_urls"]=[video_url]
        if submitted.exists():
            old=read(folder/"request.json")["request"]
            if old!=request:raise ValueError("Previous paid request parameters differ "+label)
            print("REUSE_ACCEPTED",label,read(submitted)["request_id"],flush=True)
            continue
        if marker.exists():raise RuntimeError("Ambiguous existing paid request "+label)
        save(folder/"request.json",{"endpoint":ENDPOINT,"request":request,
                                   "same_control_experiment":"heldout_v6_fourcam_backward_right_400_ref2va"})
        save(marker,{"endpoint":ENDPOINT,"request":request,
            "warning":"May have been accepted: never retry another paid HTTP POST until reconciled"})
        print("SUBMITTING_PAID_REF2VA_AB",label,flush=True)
        response=requests.post("https://queue.fal.run/"+ENDPOINT,
              headers=fal.headers(),json=request,timeout=90)
        if response.status_code>=400:
            save(folder/"submit_error.json",{"status_code":response.status_code,
                                             "body":response.text[:1600]})
            if response.status_code in (400,401,403,404,422):
                marker.unlink(missing_ok=True)
            raise RuntimeError("fal returned "+str(response.status_code)+" for "+label+": "+response.text[:600])
        data=response.json()
        if not all(data.get(k) for k in ("request_id","status_url","response_url")):
            save(folder/"unknown_submit_response.json",data)
            raise RuntimeError("Unrecognized submit acceptance: must not retry "+label)
        save(submitted,{**data,"endpoint":ENDPOINT})
        marker.unlink(missing_ok=True)
        print("ACCEPTED_REF2VA_AB",label,data["request_id"],flush=True)
    print("TWO_AUTHORIZED_REF2VA_AB_REQUESTS_SUBMITTED",flush=True)

def validate_video(p):
    v=ffprobe(p)
    if v["width"]<720 or v["height"]<720 or v["width"]%2 or v["height"]%2 or v["width"]!=v["height"]:
        raise ValueError("Unexpected generated video dimensions "+str(v))
    if v["nb_frames"]<100 or not (4.4<=v["duration_s"]<=5.5):
        raise ValueError("Truncated generated video "+str(v))
    return {**v,"file_size":p.stat().st_size,"sha256":sha(p)}

def collect():
    _,info=preflight()
    all_done=True
    for label in ("scale0","scale1"):
        folder=OUT/label
        sub_file=folder/"submitted.json"
        if not sub_file.exists():
            print("NOT_SUBMITTED",label,flush=True);all_done=False;continue
        sub=read(sub_file)
        movie=folder/("h3_ref2va_"+label+".mp4")
        if movie.exists() and (folder/"validation.json").exists():
            print("ALREADY_COMPLETED",label,read(folder/"validation.json")["sha256"],flush=True)
            continue
        if (folder/"response.json").exists():
            result=read(folder/"response.json")
        else:
            res=requests.get(sub["status_url"],headers=fal.headers(),timeout=60)
            res.raise_for_status()
            st=res.json()
            save(folder/"last_status.json",st)
            print("CURRENT_FAL_STATUS",label,st.get("status"),sub["request_id"],flush=True)
            if st.get("status")!="COMPLETED":
                if st.get("status") not in ("IN_PROGRESS","IN_QUEUE"):
                    print("POSSIBLE_FAILED_JOB",label,json.dumps(st)[:1500],flush=True)
                all_done=False;continue
            res=requests.get(sub["response_url"],headers=fal.headers(),timeout=120)
            res.raise_for_status()
            result=res.json()
            save(folder/"response.json",result)
        video=result.get("video",{})
        link=video.get("url") if isinstance(video,dict) else None
        if not link or not link.startswith("https://"):
            raise ValueError("No valid output video URL for "+label)
        if not movie.exists():
            tmp=folder/(movie.name+".partial.mp4")
            print("DOWNLOADING",label,flush=True)
            fal.download(link,tmp)
            meta=validate_video(tmp)
            tmp.replace(movie)
        else:
            meta=validate_video(movie)
        if video.get("file_size") and meta["file_size"]!=int(video["file_size"]):
            raise ValueError("fal video size mismatch")
        save(folder/"validation.json",meta)
        print("DOWNLOADED_REF2VA_AB",label,json.dumps(meta),flush=True)
    return all_done

def status():
    _,info=preflight()
    print(json.dumps(info,indent=2))
    for label in ("scale0","scale1"):
        folder=OUT/label
        sub=folder/"submitted.json"
        if sub.is_file():
            print("SUBMITTED",label,read(sub)["request_id"])
        if (folder/"last_status.json").exists():
            print("LAST_STATUS",label,read(folder/"last_status.json").get("status"))

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "status"
    {"preflight":status,"status":status,"submit":submit,"collect":collect}[mode]()
