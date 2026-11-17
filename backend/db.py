"""Myeye AI - SQLite storage for events and enrolled faces."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from . import config

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def _get() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.ensure_dirs()
        _conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _init(_conn)
    return _conn


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            type TEXT NOT NULL,
            label TEXT NOT NULL,
            extra TEXT,
            thumb TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(type, ts);
        CREATE TABLE IF NOT EXISTS faces(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            thumb TEXT,
            created REAL
        );
        """
    )
    conn.commit()


def add_event(ev_type: str, label: str, extra: dict | None = None,
              thumb: str | None = None) -> int:
    with _lock:
        cur = _get().execute(
            "INSERT INTO events(ts, type, label, extra, thumb) VALUES(?,?,?,?,?)",
            (time.time(), ev_type, label,
             json.dumps(extra or {}, ensure_ascii=False), thumb))
        _get().commit()
        return cur.lastrowid


def update_event_label(event_id: int, label: str) -> None:
    with _lock:
        _get().execute("UPDATE events SET label = ? WHERE id = ?", (label, event_id))
        _get().commit()


def query_events(ev_type: str | None = None, search: str | None = None,
                 limit: int = 60, offset: int = 0) -> tuple[list[dict], int]:
    with _lock:
        conn = _get()
        where, args = [], []
        if ev_type:
            where.append("type = ?")
            args.append(ev_type)
        if search:
            where.append("label LIKE ?")
            args.append(f"%{search}%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        total = conn.execute(f"SELECT COUNT(*) FROM events {clause}", args).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM events {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
            args + [max(1, min(300, limit)), max(0, offset)]).fetchall()
        return [dict(r) for r in rows], total


def event_type_counts() -> dict:
    with _lock:
        rows = _get().execute("SELECT type, COUNT(*) c FROM events GROUP BY type").fetchall()
        return {r["type"]: r["c"] for r in rows}


def clear_events(ev_type: str | None = None) -> None:
    with _lock:
        conn = _get()
        if ev_type:
            conn.execute("DELETE FROM events WHERE type = ?", (ev_type,))
        else:
            conn.execute("DELETE FROM events")
        conn.commit()


def prune_events(keep: int = 20000) -> None:
    with _lock:
        conn = _get()
        row = conn.execute("SELECT MAX(id) m FROM events").fetchone()
        if row and row["m"] and row["m"] > keep:
            conn.execute("DELETE FROM events WHERE id <= ?", (row["m"] - keep,))
            conn.commit()


def top_plates(limit: int = 10) -> list[dict]:
    with _lock:
        rows = _get().execute(
            "SELECT label, COUNT(*) c, MAX(ts) last FROM events "
            "WHERE type='plate' GROUP BY label ORDER BY c DESC, last DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]


def add_face(name: str, embedding, thumb: str | None) -> int:
    with _lock:
        cur = _get().execute(
            "INSERT INTO faces(name, embedding, thumb, created) VALUES(?,?,?,?)",
            (name, embedding.tobytes() if hasattr(embedding, "tobytes") else bytes(embedding),
             thumb, time.time()))
        _get().commit()
        return cur.lastrowid


def list_faces() -> list[dict]:
    with _lock:
        rows = _get().execute("SELECT * FROM faces ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def delete_face(face_id: int) -> bool:
    with _lock:
        cur = _get().execute("DELETE FROM faces WHERE id = ?", (face_id,))
        _get().commit()
        return cur.rowcount > 0


def iter_face_embeddings() -> list[tuple[int, str, "object"]]:
    import numpy as np
    out = []
    for f in list_faces():
        vec = np.frombuffer(f["embedding"], dtype=np.float32)
        out.append((f["id"], f["name"], vec))
    return out
