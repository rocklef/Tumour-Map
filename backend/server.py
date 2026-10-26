"""Local web server for the tumour segmentation app.

Run from the tumour-map folder:  python backend/server.py   ->  http://localhost:8210
"""
import hashlib
import json
import queue
import secrets
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import uvicorn
from fastapi import Cookie, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STUDIES = DATA / "studies"
STUDIES.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "db.json"
SEED = json.loads((ROOT / "backend" / "seed.json").read_text())
lock = threading.Lock()


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()


def load_db():
    if DB_PATH.exists():
        return json.loads(DB_PATH.read_text())
    salt = secrets.token_hex(16)
    u = SEED["demo_user"]
    return {"users": [{"email": u["email"], "name": u["name"], "salt": salt, "hash": hash_pw(u["password"], salt)}], "patients": [], "studies": []}


db = load_db()


def save():
    with lock:
        tmp = DB_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(db, indent=1))
        tmp.replace(DB_PATH)


sessions = {}  # token -> email
app = FastAPI(title="Tumour Map")


def user_for(token):
    email = sessions.get(token or "")
    if not email:
        raise HTTPException(401, "Sign in to continue.")
    return next(u for u in db["users"] if u["email"] == email)


# ------------------------------------------------------------------ jobs
jobs = queue.Queue()


def set_study(sid, **kw):
    with lock:
        s = next(s for s in db["studies"] if s["id"] == sid)
        s.update(kw)
    save()


def worker():
    while True:
        sid = jobs.get()
        study = next(s for s in db["studies"] if s["id"] == sid)
        d = STUDIES / sid
        try:
            set_study(sid, status="processing", step="Starting", started=time.time())
            files = {k: d / "input" / v for k, v in study["files"].items()}
            report = pipeline.run(d, files, progress=lambda s: set_study(sid, step=s))
            set_study(sid, status="done", step="Complete", finished=time.time(),
                      summary={"whole": report["volumes_ml"]["whole"], "detected": report["detected"], "hemisphere": report.get("hemisphere"),
                               "lobe": report["lobes"][0]["name"] if report.get("lobes") else None})
        except Exception as e:  # surface the reason to the doctor instead of failing silently
            traceback.print_exc()
            set_study(sid, status="error", step=str(e))


threading.Thread(target=worker, daemon=True).start()


def new_study(pid, files_src, label, copy=True):
    sid = uuid.uuid4().hex[:12]
    d = STUDIES / sid / "input"
    d.mkdir(parents=True)
    names = {}
    for k, src in files_src.items():
        name = f"{k}.nii.gz" if str(src).endswith(".gz") else f"{k}.nii"
        if copy:
            shutil.copy(src, d / name)
        names[k] = name
    study = {"id": sid, "patient": pid, "label": label, "created": time.time(), "status": "queued", "step": "Waiting in queue", "files": names}
    with lock:
        db["studies"].append(study)
    save()
    jobs.put(sid)
    return study


def seed_demo():
    if db["patients"]:
        return
    for p in SEED["demo_patients"]:
        pid = uuid.uuid4().hex[:10]
        db["patients"].append({"id": pid, "name": p["name"], "mrn": p["mrn"], "age": p["age"], "sex": p["sex"], "notes": p["notes"], "created": time.time()})
        case = ROOT / "samples" / p["case"]
        files = {s: next(case.glob(f"*_{s}.nii*")) for s in pipeline.SEQS}
        seg = list(case.glob("*_seg.nii*"))
        if seg:
            files["seg"] = seg[0]
        new_study(pid, files, p["study_label"])
    save()


# requeue anything interrupted by a restart
for s in db["studies"]:
    if s["status"] in ("queued", "processing"):
        s.update(status="queued", step="Waiting in queue")
        jobs.put(s["id"])
seed_demo()


# ------------------------------------------------------------------ auth
class Login(BaseModel):
    email: str
    password: str


@app.post("/api/login")
def login(body: Login, response: Response):
    u = next((u for u in db["users"] if u["email"].lower() == body.email.strip().lower()), None)
    if not u or hash_pw(body.password, u["salt"]) != u["hash"]:
        raise HTTPException(401, "That email and password don't match an account.")
    token = secrets.token_urlsafe(24)
    sessions[token] = u["email"]
    response.set_cookie("tm_session", token, httponly=True, samesite="strict")
    return {"email": u["email"], "name": u["name"]}


@app.post("/api/login/demo")
def login_demo(response: Response):
    u = SEED["demo_user"]
    return login(Login(email=u["email"], password=u["password"]), response)


@app.post("/api/logout")
def logout(response: Response, tm_session: str = Cookie(None)):
    sessions.pop(tm_session or "", None)
    response.delete_cookie("tm_session")
    return {"ok": True}


@app.get("/api/me")
def me(tm_session: str = Cookie(None)):
    u = user_for(tm_session)
    return {"email": u["email"], "name": u["name"]}


# ------------------------------------------------------------------ patients & studies
class PatientIn(BaseModel):
    name: str
    mrn: str = ""
    age: int | None = None
    sex: str = ""
    notes: str = ""


@app.get("/api/patients")
def patients(tm_session: str = Cookie(None)):
    user_for(tm_session)
    out = []
    for p in db["patients"]:
        st = sorted([s for s in db["studies"] if s["patient"] == p["id"]], key=lambda s: -s["created"])
        out.append({**p, "studies": st})
    return sorted(out, key=lambda p: -p["created"])


@app.post("/api/patients")
def add_patient(body: PatientIn, tm_session: str = Cookie(None)):
    user_for(tm_session)
    if not body.name.strip():
        raise HTTPException(400, "Enter the patient's name.")
    p = {"id": uuid.uuid4().hex[:10], **body.model_dump(), "created": time.time()}
    with lock:
        db["patients"].append(p)
    save()
    return p


@app.post("/api/patients/{pid}/studies")
async def upload_study(pid: str, label: str = Form("MRI brain"), t1: UploadFile = File(...), t1ce: UploadFile = File(...),
                       t2: UploadFile = File(...), flair: UploadFile = File(...), tm_session: str = Cookie(None)):
    user_for(tm_session)
    if not any(p["id"] == pid for p in db["patients"]):
        raise HTTPException(404, "Patient not found.")
    tmp = DATA / "upload" / uuid.uuid4().hex
    tmp.mkdir(parents=True)
    files = {}
    for k, f in {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair}.items():
        if not (f.filename.endswith(".nii") or f.filename.endswith(".nii.gz")):
            raise HTTPException(400, f"{k.upper()} must be a NIfTI file (.nii or .nii.gz). Convert DICOM series with dcm2niix first.")
        dst = tmp / (k + (".nii.gz" if f.filename.endswith(".gz") else ".nii"))
        dst.write_bytes(await f.read())
        files[k] = dst
    study = new_study(pid, files, label)
    shutil.rmtree(tmp, ignore_errors=True)
    return study


@app.post("/api/patients/{pid}/studies/sample")
def sample_study(pid: str, case: str = Form("BraTS2021_00014"), tm_session: str = Cookie(None)):
    user_for(tm_session)
    d = ROOT / "samples" / case
    if not d.is_dir():
        raise HTTPException(404, "Sample case not found.")
    files = {s: next(d.glob(f"*_{s}.nii*")) for s in pipeline.SEQS}
    seg = list(d.glob("*_seg.nii*"))
    if seg:
        files["seg"] = seg[0]
    return new_study(pid, files, f"Sample MRI ({case})")


@app.get("/api/samples")
def samples(tm_session: str = Cookie(None)):
    user_for(tm_session)
    return sorted(p.name for p in (ROOT / "samples").iterdir() if p.is_dir())


@app.get("/api/studies/{sid}")
def study(sid: str, tm_session: str = Cookie(None)):
    user_for(tm_session)
    s = next((s for s in db["studies"] if s["id"] == sid), None)
    if not s:
        raise HTTPException(404, "Study not found.")
    out = dict(s)
    rp = STUDIES / sid / "report.json"
    if s["status"] == "done" and rp.exists():
        out["report"] = json.loads(rp.read_text())
    out["patient"] = next(p for p in db["patients"] if p["id"] == s["patient"])
    return out


BIN = {"t1", "t1ce", "t2", "flair", "labels", "truth", "meshes"}


@app.get("/api/studies/{sid}/{name}.bin")
def study_bin(sid: str, name: str, tm_session: str = Cookie(None)):
    user_for(tm_session)
    if name not in BIN or not all(c.isalnum() for c in sid):
        raise HTTPException(404)
    f = STUDIES / sid / (f"vol_{name}.bin" if name in pipeline.SEQS else f"{name}.bin")
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="application/octet-stream", headers={"Content-Encoding": "gzip", "Cache-Control": "private, max-age=3600"})


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8210)
