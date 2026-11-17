"""YOLO object detection / tracking wrapper with graceful degradation."""
from __future__ import annotations

import logging

log = logging.getLogger("myeye")

try:
    from ultralytics import YOLO  # noqa: F401
    HAS_ULTRALYTICS = True
except Exception:  # pragma: no cover
    HAS_ULTRALYTICS = False

try:
    import lap  # noqa: F401
    HAS_LAP = True
except Exception:  # pragma: no cover
    HAS_LAP = False

# COCO ids: 0=person 2=car 3=motorcycle 5=bus 7=truck
VEHICLE_IDS = [2, 3, 5, 7]
TRAFFIC_IDS = [0, 2, 3, 5, 7]
CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat",
}

# BGR palette used by the annotator
CLASS_COLORS = {
    "person": (46, 204, 113),
    "car": (52, 152, 219),
    "motorcycle": (155, 89, 182),
    "bus": (230, 126, 34),
    "truck": (241, 196, 15),
}


class Detector:
    """Lazy-loading YOLO detector. `infer` returns plain dicts so the rest of
    the pipeline never touches torch tensors."""

    def __init__(self) -> None:
        self._model = None
        self._name: str | None = None
        self.failed = False

    @property
    def available(self) -> bool:
        return HAS_ULTRALYTICS and not self.failed

    def ensure(self, model_name: str) -> bool:
        if not HAS_ULTRALYTICS:
            return False
        if self._model is not None and self._name == model_name:
            return True
        try:  # never block startup on Ultralytics' own package auto-updater
            from ultralytics import settings as _usettings
            _usettings.update({"autoUpdate": False})
        except Exception:
            pass
        try:
            self._model = YOLO(model_name)  # auto-downloads the .pt on first use
            self._name = model_name
            self.failed = False
            log.info("detector loaded: %s", model_name)
            return True
        except Exception as e:  # pragma: no cover
            log.error("detector load failed: %s", e)
            self.failed = True
            self._model = None
            return False

    def infer(self, frame, conf: float, class_ids: list[int] | None = None,
              track: bool = True) -> list[dict]:
        if self._model is None:
            return []
        try:
            kwargs = dict(conf=conf, verbose=False, iou=0.5)
            if class_ids:
                kwargs["classes"] = class_ids
            # Ultralytics' track() tries to pip-install `lap` on first use and
            # blocks the engine thread when offline. Use plain detection
            # instead; the pipeline's IoUTracker assigns ids as a fallback.
            if track and HAS_LAP:
                kwargs["persist"] = True
                results = self._model.track(frame, **kwargs)
            else:
                results = self._model.predict(frame, **kwargs)
        except Exception as e:  # pragma: no cover
            log.warning("inference error: %s", e)
            return []
        out: list[dict] = []
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return out
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            cid = int(clss[i])
            out.append({
                "box": tuple(float(v) for v in xyxy[i]),
                "cls": cid,
                "name": r.names.get(cid, str(cid)) if hasattr(r, "names") else str(cid),
                "conf": float(confs[i]),
                "tid": int(ids[i]) if ids[i] is not None else None,
            })
        return out


class IoUTracker:
    """Minimal IoU tracker used as a fallback when Ultralytics tracking is
    unavailable (still gives stable ids for unique counting)."""

    def __init__(self, iou_thr: float = 0.30, max_age: int = 15) -> None:
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, detections: list[dict]) -> list[dict]:
        unmatched = list(range(len(detections)))
        for tid, tr in list(self.tracks.items()):
            best, best_iou = None, self.iou_thr
            for j in unmatched:
                if detections[j]["name"] != tr["name"]:
                    continue
                i = self._iou(tr["box"], detections[j]["box"])
                if i > best_iou:
                    best, best_iou = j, i
            if best is None:
                tr["age"] += 1
                if tr["age"] > self.max_age:
                    del self.tracks[tid]
            else:
                self.tracks[tid]["box"] = detections[best]["box"]
                self.tracks[tid]["age"] = 0
                detections[best]["tid"] = tid
                unmatched.remove(best)
        for j in unmatched:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"box": detections[j]["box"],
                                "name": detections[j]["name"], "age": 0}
            detections[j]["tid"] = tid
        return detections
