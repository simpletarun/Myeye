"""Face detection (YuNet) + recognition (SFace) via OpenCV - 100% free, fully
local. The two small ONNX models are downloaded once from OpenCV Zoo."""
from __future__ import annotations

import logging
import threading
import urllib.request

import cv2
import numpy as np

from .. import config

log = logging.getLogger("myeye")

_cos_threshold = 0.363  # recommended cosine threshold for SFace


def _download(url: str, dest, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    config.ensure_dirs()
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Myeye"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(dest)
        log.info("downloaded %s", dest.name)
        return True
    except Exception as e:
        log.error("download failed for %s: %s", url, e)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


class FaceEngine:
    """Lazy singleton-style engine; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._detector = None
        self._recognizer = None
        self._ready = False
        self._failed = False
        self._last_size = (0, 0)

    @property
    def ready(self) -> bool:
        return self._ready

    def prepare(self) -> bool:
        """Force model load; safe to call repeatedly."""
        return self._init()

    @property
    def failed(self) -> bool:
        return self._failed

    def _init(self) -> bool:
        if self._ready:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._ready:
                return True
            if not (_download(config.YUNET_URL, config.YUNET_PATH)
                    and _download(config.SFACE_URL, config.SFACE_PATH)):
                self._failed = True
                return False
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    str(config.YUNET_PATH), "", (320, 320),
                    score_threshold=0.70, nms_threshold=0.3, top_k=5000)
                self._recognizer = cv2.FaceRecognizerSF.create(
                    str(config.SFACE_PATH), "")
                self._ready = True
                log.info("face engine ready (YuNet + SFace)")
                return True
            except Exception as e:
                log.error("face engine init failed: %s", e)
                self._failed = True
                return False

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """Each item is a YuNet row: x,y,w,h, 10 landmarks, score."""
        if not self._init():
            return []
        h, w = frame.shape[:2]
        scale = 1.0
        small = frame
        # The pipeline already caps frames at 1280; shrinking further to 960
        # lost real faces in the 80-100 px range. 1280 still runs on CPU.
        if max(h, w) > 1280:  # only for larger API uploads
            scale = 1280.0 / max(h, w)
            small = cv2.resize(frame, None, fx=scale, fy=scale)
        # The detector keeps its input size as mutable state, so the live
        # pipeline and the search/enroll endpoints must not interleave.
        with self._infer_lock:
            if (small.shape[1], small.shape[0]) != self._last_size:
                self._detector.setInputSize((small.shape[1], small.shape[0]))
                self._last_size = (small.shape[1], small.shape[0])
            try:
                ret, faces = self._detector.detect(small)
            except Exception as e:  # pragma: no cover
                log.debug("yunet error: %s", e)
                return []
        out: list[np.ndarray] = []
        if faces is not None:
            for row in faces:
                if scale != 1.0:
                    row = row.copy()
                    row[:14] /= scale  # box AND landmarks; alignCrop needs both
                if row[14] < 0.70:
                    continue
                out.append(row)
        return out

    def embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray | None:
        if not self._ready:
            return None
        try:
            with self._infer_lock:
                aligned = self._recognizer.alignCrop(frame, face_row)
                feat = self._recognizer.feature(aligned)  # 1x128 float32
            return feat.flatten().astype(np.float32)
        except Exception as e:
            log.debug("embed error: %s", e)
            return None

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for SFace embeddings (higher = more alike)."""
        try:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denom) if denom > 0 else -1.0
        except Exception:
            return -1.0

    @staticmethod
    def is_match(sim: float) -> bool:
        return sim >= _cos_threshold


_default_engine: FaceEngine | None = None


def get_engine() -> FaceEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = FaceEngine()
    return _default_engine
