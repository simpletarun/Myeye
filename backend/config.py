"""Myeye AI - paths, defaults and persistent settings."""
import json
import os
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VISIONHUB_DATA", str(BASE_DIR / "data")))
MODELS_DIR = DATA_DIR / "models"
EVENTS_DIR = DATA_DIR / "events"
FACES_DIR = DATA_DIR / "faces"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "visionhub.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
DEMO_VIDEO = BASE_DIR / "assets" / "demo_traffic.mp4"

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# COCO ids: 0=person 2=car 3=motorcycle 5=bus 7=truck
DEFAULT_CLASS_IDS = [0, 2, 3, 5, 7]

DEFAULTS = {
    "confidence": 0.35,
    "model": "yolov8n.pt",
    "detect_objects": True,
    "detect_classes": "traffic",          # traffic = person + vehicles, all = every COCO class
    "detect_plates": True,
    "plate_mode": "vehicle",              # vehicle = search plates inside vehicles, closeup = whole frame
    "plate_min_conf": 0.55,               # below this the OCR text is garbage more often than not
    "recognize_colors": True,
    "recognize_faces": True,
    "face_threshold": 0.40,
    "loop_file": True,
    "max_fps": 25.0,
}

_lock = threading.Lock()
_settings = None


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, EVENTS_DIR, FACES_DIR, UPLOADS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load() -> dict:
    global _settings
    with _lock:
        if _settings is None:
            merged = dict(DEFAULTS)
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    stored = json.load(f)
                for k in DEFAULTS:
                    if k in stored:
                        merged[k] = stored[k]
            except Exception:
                pass
            _settings = merged
        return dict(_settings)


def update(patch: dict) -> dict:
    global _settings
    with _lock:
        cur = dict(load_unlocked())
        for k, v in patch.items():
            if k in DEFAULTS:
                cur[k] = v
        _settings = cur
        try:
            ensure_dirs()
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(cur, f, indent=2)
        except Exception:
            pass
        return dict(cur)


def load_unlocked() -> dict:
    """load() without re-locking (internal use)."""
    global _settings
    if _settings is None:
        merged = dict(DEFAULTS)
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                stored = json.load(f)
            for k in DEFAULTS:
                if k in stored:
                    merged[k] = stored[k]
        except Exception:
            pass
        _settings = merged
    return _settings
