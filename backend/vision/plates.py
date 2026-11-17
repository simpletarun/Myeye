"""License plate localisation + OCR with a chain of free local engines."""
from __future__ import annotations

import logging
import re
import threading

import cv2
import numpy as np

log = logging.getLogger("myeye")


# --------------------------------------------------------------------------- #
# Plate localisation (edge/contour heuristic - works on vehicle crops and on
# close-up plate cameras without any extra model).
# --------------------------------------------------------------------------- #
class PlateLocator:
    def __init__(self, max_plates: int = 3) -> None:
        self.max_plates = max_plates

    def find(self, img: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        h, w = img.shape[:2]
        if h < 30 or w < 60:
            return []
        scale = 1.0
        if max(h, w) > 800:  # contour search gets slow on busy 720p+ frames
            scale = 800.0 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 40, 40)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sx = cv2.normalize(np.abs(sx), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, th = cv2.threshold(sx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 40), 3))
        morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kern, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        cands: list[tuple[tuple[int, int, int, int], float]] = []
        for cnt in cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh < 14 or bh > h * 0.75 or bw < 2.0 * bh or bw > 6.8 * bh:
                continue
            if bw * bh < 0.001 * w * h:
                continue
            roi = th[y:y + bh, x:x + bw]
            density = float(np.count_nonzero(roi)) / float(bw * bh)
            fill = cv2.contourArea(cnt) / float(bw * bh)
            if density < 0.16 or fill < 0.35:
                continue
            cands.append(((x, y, x + bw, y + bh), min(1.0, 0.55 * fill + 0.45 * density)))
        picked = self._nms(sorted(cands, key=lambda c: c[1], reverse=True))
        if scale != 1.0:
            picked = [((int(x1 / scale), int(y1 / scale),
                        int(x2 / scale), int(y2 / scale)), score)
                      for (x1, y1, x2, y2), score in picked]
        return picked

    @staticmethod
    def _nms(cands, iou_thr: float = 0.3):
        picked = []
        for box, score in cands:
            ok = True
            for pbox, _ in picked:
                # IoU misses a small box nested inside a larger one
                if (PlateLocator._iou(box, pbox) > iou_thr
                        or PlateLocator._containment(box, pbox) > 0.4):
                    ok = False
                    break
            if ok:
                picked.append((box, score))
            if len(picked) >= 3:
                break
        return picked

    @staticmethod
    def _containment(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        smallest = min((a[2] - a[0]) * (a[3] - a[1]),
                       (b[2] - b[0]) * (b[3] - b[1]))
        return inter / smallest if smallest > 0 else 0.0

    @staticmethod
    def _iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0


def _clean_plate_text(text: str) -> str | None:
    text = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
    if not (5 <= len(text) <= 12):
        return None
    # Real plates mix letters and digits; this rejects burned-in clocks,
    # dates, and partial numbers that OCR reads off the video timecode.
    letters = sum(c.isalpha() for c in text)
    digits = len(text) - letters
    if letters < 2 or digits < 1:
        return None
    return text


# --------------------------------------------------------------------------- #
# OCR engine chain: EasyOCR -> fast-plate-ocr -> pytesseract. All free and
# fully local; the first available one is used.
# --------------------------------------------------------------------------- #
class PlateReader:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: str | None = None
        self._easy = None
        self._fpo = None
        self._checked = False

    # -- engine discovery ---------------------------------------------------
    def _pick_engine(self) -> str | None:
        if self._checked:
            return self._engine
        self._checked = True
        try:
            import easyocr
            # Build the reader during startup warm-up, not on the first crop.
            self._easy = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._engine = "easyocr"
            return self._engine
        except Exception as e:
            log.info("easyocr unavailable: %s", e)
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            self._fpo = LicensePlateRecognizer(hub_ocr_model="cct-xs-v1-global-model",
                                               device="cpu")
            self._engine = "fast-plate-ocr"
            return self._engine
        except Exception as e:
            log.info("fast-plate-ocr unavailable: %s", e)
        try:
            import pytesseract  # noqa: F401
            pytesseract.get_tesseract_version()
            self._engine = "tesseract"
            return self._engine
        except Exception:
            pass
        self._engine = None
        return None

    @property
    def engine_name(self) -> str | None:
        # Never block the dashboard API while the startup warm-up is
        # downloading/initialising the OCR model.
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._pick_engine()
        finally:
            self._lock.release()

    # -- reading ------------------------------------------------------------
    def read(self, crop: np.ndarray) -> tuple[str | None, float]:
        """Returns (plate_text, confidence 0..1) or (None, 0)."""
        if crop is None or crop.size == 0:
            return None, 0.0
        crop = self._prep(crop)
        # The startup warm-up may still be downloading the OCR model. Never
        # block the capture thread on it: skip this frame's read instead.
        if not self._lock.acquire(blocking=False):
            return None, 0.0
        try:
            engine = self._pick_engine()
            if engine == "easyocr":
                return self._read_easyocr(crop)
            if engine == "fast-plate-ocr":
                return self._read_fast(crop)
            if engine == "tesseract":
                return self._read_tesseract(crop)
            return None, 0.0
        finally:
            self._lock.release()

    @staticmethod
    def _prep(crop: np.ndarray) -> np.ndarray:
        h = crop.shape[0]
        if h < 48:
            scale = min(4.0, 56.0 / max(1, h))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        elif h > 160:
            scale = 160.0 / h
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return crop

    def _read_fast(self, crop: np.ndarray) -> tuple[str | None, float]:
        try:
            preds = self._fpo.run(crop, return_confidence=True)
            if not preds:
                return None, 0.0
            p = preds[0]
            text = getattr(p, "plate", None)
            conf = getattr(p, "confidence", None)
            if conf is None:
                probs = getattr(p, "char_probs", None)
                conf = float(np.mean(probs)) if probs is not None and len(probs) else 0.5
            return _clean_plate_text(text or ""), float(conf)
        except Exception as e:
            log.debug("fast-plate-ocr failed: %s", e)
            return None, 0.0

    def _read_easyocr(self, crop: np.ndarray) -> tuple[str | None, float]:
        try:
            results = self._easy.readtext(crop, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            texts, confs = [], []
            for _, t, c in results:
                texts.append(t)
                confs.append(float(c))
            return _clean_plate_text("".join(texts)), (float(np.mean(confs)) if confs else 0.0)
        except Exception as e:
            log.debug("easyocr failed: %s", e)
            return None, 0.0

    def _read_tesseract(self, crop: np.ndarray) -> tuple[str | None, float]:
        try:
            import pytesseract
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(
                gray, config="--psm 7 -c tessedit_char_whitelist="
                             "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            return _clean_plate_text(text), 0.6 if text.strip() else 0.0
        except Exception as e:
            log.debug("tesseract failed: %s", e)
            return None, 0.0
