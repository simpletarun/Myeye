"""VideoEngine - the real-time analysis pipeline tying every module together.

A single background thread owns the active capture source, runs the enabled
modules on every frame, produces the annotated MJPEG frame, updates live
statistics and writes deduplicated events to SQLite.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from datetime import datetime
from difflib import SequenceMatcher

import cv2
import numpy as np

from .. import config, db
from .colors import classify_vehicle_color
from .detector import CLASS_COLORS, VEHICLE_IDS, Detector, \
    HAS_ULTRALYTICS, IoUTracker
from .faces import get_engine
from .plates import PlateLocator, PlateReader

log = logging.getLogger("myeye")


def _ts_str(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _enhance_crop(img: np.ndarray, target_h: int) -> np.ndarray:
    """Upscale a small plate/face crop and sharpen it for identification."""
    h, w = img.shape[:2]
    scale = min(4.0, target_h / max(1, h))
    if abs(scale - 1.0) > 0.05:
        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=interp)
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.55, blur, -0.55, 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _save_thumb(img: np.ndarray, ev_type: str, max_h: int = 110,
                enhance: bool = False) -> str | None:
    """Persist an event thumbnail; returns path relative to DATA_DIR."""
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if enhance:
        img = _enhance_crop(img, max_h)
    elif h > max_h:
        scale = max_h / h
        img = cv2.resize(img, (max(1, int(w * scale)), max_h))
    config.ensure_dirs()
    name = f"{time.time_ns()}_{ev_type}_{threading.get_ident() % 10000}.jpg"
    path = config.EVENTS_DIR / name
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        return None
    try:
        buf.tofile(str(path))
    except Exception:
        return None
    return f"events/{name}"


class VideoEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()  # serialises source switches
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._source: dict = {"type": "idle", "value": None}

        self.detector = Detector()
        self._fallback_tracker = IoUTracker()
        self._locator = PlateLocator()
        self._reader = PlateReader()

        self._latest_jpeg: bytes | None = None
        self._latest_raw: np.ndarray | None = None
        self._jpeg_lock = threading.Lock()

        self._started_at: float | None = None
        self._fps: float = 0.0
        self._frame_size: tuple[int, int] = (0, 0)

        self._seen_tids: set[int] = set()
        self._plate_memory: dict[object, dict] = {}  # vehicle key -> best read
        self._plate_tracks: list[dict] = []        # untracked plates matched by box
        self._seen_plates: list[dict] = []         # session-wide plate texts
        self._face_memory: dict[str, float] = {}   # name -> last ts
        self._face_warm_started = False
        self._last_face_ts = 0.0
        self._active: Counter = Counter()
        self._unique: Counter = Counter()
        self._session_events: Counter = Counter()
        self._timeline: deque = deque(maxlen=240)  # (ts, active_objects) 1 Hz
        self._last_bucket = 0
        self._events_feed: deque = deque(maxlen=120)
        self._frame_count = 0

    # ------------------------------------------------------------------ api
    def set_settings(self, patch: dict) -> dict:
        return config.update(patch)

    def get_settings(self) -> dict:
        return config.load()

    def start(self, source_type: str, value) -> dict:
        with self._start_lock:
            # Never join while holding _lock: the old capture thread needs
            # that lock to publish its last event and exit.
            with self._lock:
                old_thread = self._thread
                old_evt = self._stop_evt
            old_evt.set()
            if old_thread and old_thread.is_alive():
                old_thread.join(timeout=6)
            with self._lock:
                # A run that failed to stop in time keeps its own set event,
                # so it cannot see this run's event or reuse its capture.
                self._stop_evt = threading.Event()
                self._source = {"type": source_type, "value": value}
                self._seen_tids.clear()
                self._plate_memory.clear()
                self._plate_tracks.clear()
                self._seen_plates.clear()
                self._face_memory.clear()
                self._fallback_tracker = IoUTracker()
                self._session_events.clear()
                self._started_at = time.time()
                self._thread = threading.Thread(target=self._run, daemon=True,
                                                name="myeye-engine",
                                                args=(self._stop_evt,))
                self._thread.start()
                return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._stop_evt.set()
            self._source = {"type": "idle", "value": None}
            self._started_at = None
        return self.status()

    def status(self) -> dict:
        with self._lock:
            running = bool(self._thread and self._thread.is_alive()
                           and not self._stop_evt.is_set())
            return {
                "running": running,
                "source": dict(self._source),
                "uptime": (time.time() - self._started_at) if self._started_at else 0,
            }

    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg

    def latest_raw(self) -> np.ndarray | None:
        with self._jpeg_lock:
            return None if self._latest_raw is None else self._latest_raw.copy()

    def stats(self) -> dict:
        with self._lock:
            face_engine = get_engine()
            st = self.status()
            # prune timeline buckets older than 4 min
            now = time.time()
            while self._timeline and now - self._timeline[0][0] > 240:
                self._timeline.popleft()
            return {
                **st,
                "fps": round(self._fps, 1),
                "frame_size": list(self._frame_size),
                "active": dict(self._active),
                "active_total": int(sum(self._active.values())),
                "unique": dict(self._unique),
                "timeline": [[t, n] for t, n in self._timeline],
                "frames": self._frame_count,
                "engines": {
                    "detector": HAS_ULTRALYTICS and not self.detector.failed,
                    "ocr": self._reader.engine_name,
                    "faces": face_engine.ready,
                    "faces_failed": face_engine.failed,
                },
                "event_counts": db.event_type_counts(),
            }

    def events_feed(self) -> list[dict]:
        with self._lock:
            return list(self._events_feed)

    def _push_feed(self, ev: dict) -> None:
        with self._lock:
            self._events_feed.append(ev)

    # ------------------------------------------------------------- pipeline
    def _open_capture(self) -> cv2.VideoCapture:
        stype, value = self._source["type"], self._source["value"]
        if stype == "webcam":
            idx = int(value) if str(value).strip().isdigit() else 0
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif stype == "rtsp":
            cap = cv2.VideoCapture(str(value))
        elif stype == "file":
            cap = cv2.VideoCapture(str(value))
        else:
            raise ValueError("invalid source")
        if not cap.isOpened():
            raise IOError(f"cannot open source: {stype} {value}")
        return cap

    def _target_fps(self, cap) -> float:
        base = float(config.load().get("max_fps", 25.0))
        if self._source["type"] == "file":
            try:
                vfps = cap.get(cv2.CAP_PROP_FPS) or 0
                if 1 < vfps <= base:
                    return vfps
            except Exception:
                pass
        return base

    def _class_ids(self, settings: dict) -> list[int] | None:
        if not settings.get("detect_objects", True):
            return []
        if settings.get("detect_classes", "traffic") == "all":
            return None
        return config.DEFAULT_CLASS_IDS

    def _run(self, stop_evt: threading.Event) -> None:
        try:
            cap = self._open_capture()
        except Exception as e:
            log.error("source open failed: %s", e)
            with self._lock:
                # a superseded run must not overwrite the new source status
                if self._stop_evt is stop_evt:
                    self._source = {"type": "error", "value": str(e)}
            return
        with self._lock:
            if self._stop_evt is stop_evt:
                self._cap = cap
        if stop_evt.is_set():
            cap.release()
            return

        loop_file = bool(config.load().get("loop_file", True))
        is_file = self._source["type"] == "file"
        model_name = config.load().get("model", "yolov8n.pt")
        if HAS_ULTRALYTICS:
            self.detector.ensure(model_name)

        log.info("engine started: %s %s", self._source["type"], self._source["value"])
        target = self._target_fps(cap)
        last_frame_time = time.time()
        fail_count = 0
        skip = 0  # file sources: skip frames so playback stays near real time

        while not stop_evt.is_set():
            for _ in range(skip):
                if not cap.grab():
                    break
            skip = 0
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file and loop_file and fail_count < 3:
                    fail_count += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                break
            fail_count = 0

            h, w = frame.shape[:2]
            if max(h, w) > 1920:
                scale = 1920.0 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            self._frame_size = [frame.shape[1], frame.shape[0]]

            with self._jpeg_lock:
                self._latest_raw = frame.copy()

            try:
                annotated = self._process_frame(frame)
            except Exception as e:
                log.exception("frame processing error: %s", e)
                annotated = frame

            ok_j, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok_j:
                with self._jpeg_lock:
                    self._latest_jpeg = buf.tobytes()
                self._frame_count += 1

            # fps + 1 Hz timeline bucket
            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now
            inst = 1.0 / dt if dt > 0 else 0.0
            self._fps = self._fps * 0.9 + inst * 0.1 if self._fps else inst
            bucket = int(now)
            if bucket != self._last_bucket:
                self._last_bucket = bucket
                self._timeline.append((bucket, int(sum(self._active.values()))))
                self._active = Counter()

            # throttle processing rate
            target = self._target_fps(cap)
            elapsed = time.time() - t0
            if is_file:
                source_fps = cap.get(cv2.CAP_PROP_FPS) or target
                # cap the gap so detection still sees most of the action
                skip = max(0, min(5, int(elapsed * source_fps) - 1))
            delay = (1.0 / target) - elapsed
            if delay > 0:
                time.sleep(delay)

        try:
            cap.release()
        except Exception:
            pass
        log.info("engine stopped")

    # ------------------------------------------------------------ per-frame
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        settings = config.load()
        h, w = frame.shape[:2]
        objects: list[dict] = []
        plates_found: list[dict] = []
        faces_found: list[dict] = []
        now = time.time()

        # ---- object detection + tracking --------------------------------
        if settings.get("detect_objects", True) and self.detector.available:
            ids = self._class_ids(settings)
            if ids != []:
                use_track = self._source["type"] in ("file", "rtsp", "webcam")
                objects = self.detector.infer(frame, float(settings["confidence"]),
                                              ids, track=use_track)
                if objects and all(o["tid"] is None for o in objects):
                    objects = self._fallback_tracker.update(objects)

        # ---- vehicle colour + plates inside vehicles ---------------------
        color_on = settings.get("recognize_colors", True)
        plates_on = settings.get("detect_plates", True)
        vehicle_boxes: list[tuple[str, tuple, str | None, int | None]] = []
        if color_on or plates_on:
            for o in objects:
                if o["name"] in ("car", "motorcycle", "bus", "truck"):
                    if color_on:
                        o["color"] = classify_vehicle_color(self._crop(frame, o["box"]))
                    vehicle_boxes.append((o["name"], o["box"], o.get("color"), o.get("tid")))

        if plates_on:
            mode = settings.get("plate_mode", "vehicle")
            if mode == "closeup":
                regions = [((0, 0, w, h), None, "closeup", None)]
            elif not vehicle_boxes:
                # No vehicle found (e.g. the synthetic demo): scan the frame,
                # but skip the broadcast-timecode strip in _looks_like_timecode().
                regions = [((0, 0, w, h), None, "full", None)]
            else:
                # dedupe plates per tracked vehicle, not per class
                regions = [(box, name,
                            ("veh", tid) if tid is not None else tuple(v // 120 for v in box),
                            vcolor)
                           for name, box, vcolor, tid in vehicle_boxes]
            for region_box, veh_name, mem_key, veh_color in regions:
                x1, y1, x2, y2 = [int(v) for v in region_box]
                roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if roi.size == 0:
                    continue
                for pbox, score in self._locator.find(roi):
                    gx1, gy1, gx2, gy2 = pbox
                    if (gx2 - gx1) < 2.0 * (gy2 - gy1):
                        continue  # too square/partial to be a readable plate
                    box_abs = (x1 + gx1, y1 + gy1, x1 + gx2, y1 + gy2)
                    if (box_abs[0] <= 2 or box_abs[1] <= 2
                            or box_abs[2] >= w - 2 or box_abs[3] >= h - 2):
                        continue  # plate runs off-screen: text would be partial
                    if mem_key == "full" and self._looks_like_timecode(box_abs, w, h):
                        continue
                    if mem_key in ("closeup", "full"):
                        if self._plate_recently_emitted(box_abs, now):
                            continue
                    else:
                        entry = self._plate_memory.get(mem_key)
                        if entry is not None and now - entry["emitted"] < 10.0:
                            continue
                    crop = self._crop(roi, (gx1 - int((gx2 - gx1) * 0.08),
                                            gy1 - int((gy2 - gy1) * 0.10),
                                            gx2 + int((gx2 - gx1) * 0.18),
                                            gy2 + int((gy2 - gy1) * 0.10)))
                    text, conf = self._reader.read(crop) if crop is not None else (None, 0)
                    # Keep unreadable candidates too: the live overlay shows
                    # where a plate was found even before OCR succeeds.
                    plates_found.append({
                        "box": box_abs, "text": text or "", "conf": conf,
                        "vehicle": veh_name, "key": mem_key,
                        "color": veh_color, "crop": crop,
                    })

        # ---- faces --------------------------------------------------------
        if settings.get("recognize_faces", True):
            engine = get_engine()
            if not engine.ready and not engine.failed and not self._face_warm_started:
                # never block the capture thread on a 38 MB download
                self._face_warm_started = True
                threading.Thread(target=engine.prepare, daemon=True).start()
            # Face inference is the most expensive stage; 5 Hz is plenty for
            # a 5 s event cooldown and keeps high-FPS sources responsive.
            if engine.ready and now - self._last_face_ts >= 0.2:
                self._last_face_ts = now
                thr = float(settings.get("face_threshold", 0.40))
                known = db.iter_face_embeddings()
                for row in engine.detect(frame):
                    x, y, fw, fh = row[:4]
                    box = (int(x), int(y), int(x + fw), int(y + fh))
                    vec = engine.embed(frame, row)
                    name, sim = "Unknown", -1.0
                    if vec is not None and known:
                        best_id, best_sim = None, -1.0
                        for fid, fname, fvec in known:
                            s = engine.similarity(vec, fvec)
                            if s > best_sim:
                                best_id, best_sim = fid, s
                        if best_sim >= thr:
                            name = next(f[1] for f in known if f[0] == best_id)
                            sim = best_sim
                        else:
                            sim = best_sim
                    faces_found.append({"box": box, "name": name, "sim": sim,
                                        "crop": self._crop(frame, box, pad=0.25),
                                        "vec": vec})

        self._emit_events(frame, objects, plates_found, faces_found, now)
        self._annotate(frame, objects, plates_found, faces_found, settings)
        return frame

    # --------------------------------------------------------------- events
    def _emit_events(self, frame, objects, plates, faces, now) -> None:
        settings = config.load()

        for o in objects:
            tid = o.get("tid")
            if tid is not None and tid not in self._seen_tids:
                self._seen_tids.add(tid)
                self._unique[o["name"]] += 1
                if settings.get("detect_objects", True) and o["name"] in (
                        "person", "car", "motorcycle", "bus", "truck"):
                    crop = self._crop(frame, o["box"])
                    thumb = _save_thumb(crop, "object", max_h=140)
                    ev = {"ts": now, "time": _ts_str(now), "type": "object",
                          "label": o["name"], "extra": {"conf": round(o["conf"], 2),
                                                        "track_id": tid,
                                                        "color": o.get("color")},
                          "thumb": thumb}
                    db.add_event(ev["type"], ev["label"], ev["extra"], thumb)
                    self._push_feed(ev)
            if tid is not None:
                self._active[o["name"]] += 1

        for p in plates:
            if not p["text"]:
                continue
            if p["conf"] < float(settings.get("plate_min_conf", 0.55)):
                continue
            # Confirm a plate over five sightings and keep its best read, so
            # partial/one-frame OCR guesses never reach the counters.
            if p["key"] in ("closeup", "full"):
                p = self._track_plate(p, now)
            else:
                p = self._plate_for_vehicle(p, now)
            if p is None:
                continue
            if self._plate_already_seen(p["text"]):
                continue
            thumb = _save_thumb(p["crop"], "plate", max_h=220, enhance=True)
            ev = {"ts": now, "time": _ts_str(now), "type": "plate",
                  "label": p["text"],
                  "extra": {"conf": round(p["conf"], 2), "vehicle": p["vehicle"],
                            "color": p.get("color")},
                  "thumb": thumb}
            ev_id = db.add_event(ev["type"], ev["label"], ev["extra"], thumb)
            self._remember_plate(p["text"], ev_id)
            self._session_events["plate"] += 1
            self._push_feed(ev)

        for f in faces:
            last = self._face_memory.get(f["name"], 0)
            if now - last < 5.0:
                continue
            self._face_memory[f["name"]] = now
            thumb = _save_thumb(f.get("crop"), "face", max_h=360, enhance=True)
            ev = {"ts": now, "time": _ts_str(now), "type": "face",
                  "label": f["name"],
                  "extra": {"sim": round(f["sim"], 3) if f["sim"] > 0 else None},
                  "thumb": thumb}
            db.add_event(ev["type"], ev["label"], ev["extra"], thumb)
            self._session_events["face"] += 1
            self._push_feed(ev)

    def _plate_already_seen(self, text: str) -> bool:
        """One event per plate per session; later clearer reads only enrich it."""
        for row in self._seen_plates:
            if SequenceMatcher(None, row["text"], text).ratio() >= 0.8:
                if len(text) > len(row["text"]):
                    row["text"] = text
                    if row.get("id"):
                        db.update_event_label(row["id"], text)
                return True
        return False

    def _remember_plate(self, text: str, ev_id: int) -> None:
        self._seen_plates.append({"text": text, "id": ev_id})
        self._seen_plates = self._seen_plates[-200:]

    def _plate_for_vehicle(self, p: dict, now: float) -> dict | None:
        """One confirmed event per tracked vehicle, with its clearest read."""
        mem = self._plate_memory
        key = p["key"]
        entry = mem.get(key)
        if entry is None or now - entry["ts"] > 10.0:
            mem[key] = {"text": p["text"], "conf": p["conf"], "hits": 1,
                        "ts": now, "emitted": 0.0}
            return None
        entry["hits"] += 1
        entry["ts"] = now
        # Prefer the longest plausible read: a 3-frame partial plate often has
        # higher OCR confidence than the full plate seen later.
        if (len(p["text"]) > len(entry["text"])
                or (len(p["text"]) == len(entry["text"])
                    and p["conf"] > entry["conf"])):
            entry["text"], entry["conf"] = p["text"], p["conf"]
        if entry["hits"] < 5 or now - entry["emitted"] < 10.0:
            return None
        entry["emitted"] = now
        p["text"], p["conf"] = entry["text"], entry["conf"]
        return p

    def _plate_recently_emitted(self, box, now: float) -> bool:
        for t in self._plate_tracks:
            if now - t["emitted"] < 20.0 and PlateLocator._iou(box, t["box"]) > 0.2:
                return True
        return False

    def _track_plate(self, p: dict, now: float) -> dict | None:
        """Follow a plate by box overlap when the vehicle has no track id."""
        tracks = self._plate_tracks
        tracks[:] = [t for t in tracks
                     if now - t["ts"] <= 5.0 or now - t["emitted"] < 20.0]
        best, best_iou = None, 0.15
        for t in tracks:
            iou = PlateLocator._iou(p["box"], t["box"])
            if iou > best_iou:
                best, best_iou = t, iou
        if best is None:
            tracks.append({"box": p["box"], "text": p["text"], "conf": p["conf"],
                           "hits": 1, "ts": now, "emitted": 0.0})
            return None
        best["box"], best["ts"] = p["box"], now
        best["hits"] += 1
        if (len(p["text"]) > len(best["text"])
                or (len(p["text"]) == len(best["text"])
                    and p["conf"] > best["conf"])):
            best["text"], best["conf"] = p["text"], p["conf"]
        if best["hits"] < 5 or now - best["emitted"] < 20.0:
            return None
        best["emitted"] = now
        p["text"], p["conf"] = best["text"], best["conf"]
        return p

    # ----------------------------------------------------------- annotation
    def _annotate(self, frame, objects, plates, faces, settings) -> None:
        # Scale labels with the frame so they stay readable on 1080p/1440p.
        k = max(1.0, frame.shape[1] / 1280.0)
        th = max(2, int(round(2 * k)))
        for o in objects:
            x1, y1, x2, y2 = [int(v) for v in o["box"]]
            color = CLASS_COLORS.get(o["name"], (0, 200, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)
            label = o["name"]
            if o.get("color"):
                label += f" · {o['color']}"
            if o.get("tid") is not None:
                label += f" #{o['tid']}"
            _chip(frame, label, x1, y1, color, k)

        for p in plates:
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            if p["text"]:
                color = (80, 220, 80)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)
                _chip(frame, f"{p['text']}  {p['conf']:.2f}", x1, max(0, y1 - 4),
                      color, k)
            else:
                # Locator found a plate but OCR has not confirmed it yet.
                color = (0, 215, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, th - 1))
                _chip(frame, "plate", x1, max(0, y1 - 4), color, k)

        for f in faces:
            x1, y1, x2, y2 = f["box"]
            color = (200, 120, 255) if f["name"] != "Unknown" else (90, 90, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)
            sim_txt = f"  {f['sim']:.2f}" if f["sim"] > 0 else ""
            _chip(frame, f"{f['name']}{sim_txt}", x1, max(0, y1 - 4), color, k)

        # HUD shows this run's reads, not all-time database totals
        chips = [
            f"FPS {self._fps:.0f}",
            f"Objects {sum(self._active.values())}",
            f"Plates {self._session_events['plate']}",
            f"Faces {self._session_events['face']}",
        ]
        x = 14
        for i, c in enumerate(chips):
            (tw, thh), _ = cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, 1)
            cv2.rectangle(frame, (x, 12), (x + tw + 18, 12 + int(30 * k)),
                          (18, 18, 20), -1)
            cv2.putText(frame, c, (x + 9, 12 + int(20 * k)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k,
                        (235, 235, 240), 1, cv2.LINE_AA)
            x += tw + 34

    @staticmethod
    def _looks_like_timecode(box, frame_w: int, frame_h: int) -> bool:
        """Burned-in clocks/dates sit in a top or bottom centre strip."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        centered = 0.12 * frame_w < cx < 0.88 * frame_w
        edge = cy > 0.86 * frame_h or cy < 0.09 * frame_h
        return centered and edge

    @staticmethod
    def _crop(frame, box, pad: float = 0.0):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        if pad > 0:
            pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
            x1, y1, x2, y2 = x1 - pw, y1 - ph, x2 + pw, y2 + ph
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]


def _chip(frame, text: str, x: int, y: int, color, scale: float = 1.0) -> None:
    """Small filled label chip anchored above (x, y)."""
    font = 0.52 * scale
    thick = max(1, int(round(scale)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
    ty = y - int(8 * scale)
    if ty - th - int(6 * scale) < 0:
        ty = y + th + int(14 * scale)
    cv2.rectangle(frame, (x, ty - th - int(6 * scale)),
                  (x + tw + int(12 * scale), ty + int(4 * scale)), color, -1)
    cv2.putText(frame, text, (x + int(6 * scale), ty),
                cv2.FONT_HERSHEY_SIMPLEX, font, (18, 18, 18), thick, cv2.LINE_AA)
