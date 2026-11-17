"""Myeye AI - FastAPI application (backend HTTP API + web UI)."""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .vision.faces import get_engine
from .vision.pipeline import VideoEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("myeye")

app = FastAPI(title="Myeye AI", version="1.0.0")
engine = VideoEngine()

WEB_DIR = config.BASE_DIR / "web"


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()
    db.prune_events()
    # warm up OCR engine + face models in the background so first use is fast
    import threading

    def warm():
        try:
            engine._reader.engine_name
        except Exception:
            pass
        try:
            get_engine().prepare()  # downloads face models once (38 MB)
        except Exception:
            pass

    threading.Thread(target=warm, daemon=True).start()


# ------------------------------------------------------------------ sources
@app.post("/api/source")
def api_source(body: dict):
    stype = body.get("type", "")
    value = body.get("value")
    if stype == "demo":
        if not config.DEMO_VIDEO.exists():
            raise HTTPException(404, "demo video not found")
        stype, value = "file", str(config.DEMO_VIDEO)
    elif stype == "webcam":
        value = int(value if str(value).strip().isdigit() else 0)
    elif stype in ("file", "rtsp"):
        value = str(value or "").strip()
        if not value:
            raise HTTPException(400, "missing source path/URL")
    else:
        raise HTTPException(400, "type must be webcam | file | rtsp | demo")
    return engine.start(stype, value)


@app.post("/api/stop")
def api_stop():
    return engine.stop()


@app.get("/api/status")
def api_status():
    return {**engine.status(), "stats": engine.stats()}


@app.get("/api/stats")
def api_stats():
    return engine.stats()


@app.get("/api/stream")
def api_stream():
    def gen():
        boundary = b"--frame\r\n"
        while True:
            jpg = engine.latest_jpeg()
            if jpg is None:
                import time as _t
                _t.sleep(0.08)
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n\r\n"
                continue
            yield (boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            import time as _t
            _t.sleep(0.03)

    return StreamingResponse(gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/frame")
def api_frame():
    jpg = engine.latest_jpeg()
    if jpg is None:
        raise HTTPException(404, "no frame yet - start a source first")
    return Response(jpg, media_type="image/jpeg")


# ---------------------------------------------------------------- settings
@app.get("/api/settings")
def api_settings_get():
    return engine.get_settings()


@app.post("/api/settings")
def api_settings_post(patch: dict):
    return engine.set_settings(patch)


# ------------------------------------------------------------------ events
@app.get("/api/events")
def api_events(type: str | None = None, search: str | None = None,
               limit: int = 60, offset: int = 0):
    rows, total = db.query_events(type, search, limit, offset)
    return {"rows": rows, "total": total}


@app.get("/api/events/export")
def api_events_export(type: str | None = None):
    rows, _ = db.query_events(type, None, 100000, 0)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "timestamp", "type", "label", "details", "thumbnail"])
    for r in rows:
        writer.writerow([r["id"],
                         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])),
                         r["type"], r["label"], r["extra"] or "", r["thumb"] or ""])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition":
                             f"attachment; filename=myeye_events_{int(time.time())}.csv"})


@app.delete("/api/events")
def api_events_clear(type: str | None = None):
    db.clear_events(type)
    return {"ok": True}


@app.get("/api/top_plates")
def api_top_plates():
    return {"plates": db.top_plates(10)}


# ------------------------------------------------------------------- faces
@app.get("/api/faces")
def api_faces():
    faces = db.list_faces()
    for f in faces:
        f["embedding"] = f"<{len(f['embedding'])} bytes>"
    return {"rows": faces}


@app.delete("/api/faces/{face_id}")
def api_faces_delete(face_id: int):
    if not db.delete_face(face_id):
        raise HTTPException(404, "face not found")
    return {"ok": True}


def _embed_image(img: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns [(embedding, aligned_row)] for every face found in the image."""
    eng = get_engine()
    if not eng.prepare():
        raise HTTPException(503, "face engine unavailable (model download failed?)")
    out = []
    for row in eng.detect(img):
        vec = eng.embed(img, row)
        if vec is not None:
            out.append((vec, row))
    return out


def _read_upload_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image (use jpg/png)")
    if max(img.shape[:2]) > 1600:
        s = 1600.0 / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    return img


def _save_face_thumb(img: np.ndarray, row: np.ndarray) -> str:
    eng = get_engine()
    try:
        aligned = eng._recognizer.alignCrop(img, row)
        crop = aligned
    except Exception:
        h_, w_ = img.shape[:2]
        x, y, fw, fh = row[:4]
        crop = img[max(0, int(y)):int(y + fh), max(0, int(x)):int(x + fw)]
    config.ensure_dirs()
    name = f"{time.time_ns()}_face.jpg"
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        (config.FACES_DIR / name).write_bytes(buf.tobytes())
        return f"faces/{name}"
    return ""


@app.post("/api/faces/enroll")
def api_faces_enroll(name: str = Form(...), files: list[UploadFile] = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "name is required")
    enrolled = 0
    for f in files:
        data = f.file.read()
        img = _read_upload_image(data)
        faces = _embed_image(img)
        if not faces:
            continue
        vec, row = faces[0]  # largest face (YuNet orders by score)
        thumb = _save_face_thumb(img, row)
        db.add_face(name, vec, thumb)
        enrolled += 1
    if enrolled == 0:
        raise HTTPException(400, "no face detected in the uploaded image(s)")
    return {"ok": True, "enrolled": enrolled}


@app.post("/api/faces/capture")
def api_faces_capture(name: str = Form(...)):
    frame = engine.latest_raw()
    if frame is None:
        raise HTTPException(400, "no live frame - start a source first")
    faces = _embed_image(frame)
    if not faces:
        raise HTTPException(400, "no face visible in the current frame")
    vec, row = faces[0]
    thumb = _save_face_thumb(frame, row)
    db.add_face(name.strip(), vec, thumb)
    return {"ok": True}


@app.post("/api/faces/search")
def api_faces_search(file: UploadFile = File(...), k: int = 5):
    img = _read_upload_image(file.file.read())
    faces = _embed_image(img)
    if not faces:
        return {"matches": [], "message": "no face found in the uploaded image"}
    vec, row = faces[0]
    eng = get_engine()
    known = db.iter_face_embeddings()
    scored = sorted(((eng.similarity(vec, fvec), fid, fname)
                     for fid, fname, fvec in known), reverse=True)[: max(1, k)]
    matches = []
    for sim, fid, fname in scored:
        m = {"face_id": fid, "name": fname, "similarity": round(sim, 3),
             "match": eng.is_match(sim)}
        for f in db.list_faces():
            if f["id"] == fid:
                m["thumb"] = f["thumb"]
                break
        matches.append(m)
    return {"matches": matches}


# ----------------------------------------------------------------- uploads
@app.post("/api/upload")
def api_upload(file: UploadFile = File(...)):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(file.filename or "video.mp4").name)
    config.ensure_dirs()
    dest = config.UPLOADS_DIR / f"{int(time.time())}_{safe}"
    size = 0
    with open(dest, "wb") as out:
        while True:
            chunk = file.file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, "empty upload")
    return {"path": str(dest), "size": size, "name": file.filename}


# ------------------------------------------------------------------ system
@app.get("/api/system")
def api_system():
    import platform
    data_dir = config.DATA_DIR
    total = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file()) \
        if data_dir.exists() else 0
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "data_dir": str(data_dir),
        "data_size_mb": round(total / 1e6, 2),
        "demo_video": config.DEMO_VIDEO.exists(),
        "face_models": {"yunet": config.YUNET_PATH.exists(),
                        "sface": config.SFACE_PATH.exists()},
        "engines": engine.stats()["engines"],
    }


# --------------------------------------------------------------- static UI
config.ensure_dirs()  # /media serves thumbnails from the data dir
app.mount("/media", StaticFiles(directory=str(config.DATA_DIR)), name="media")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
