"""SQLite access. Keep /data on a Docker named volume (not a Windows bind mount) so WAL locking works."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plex_uuid   TEXT UNIQUE NOT NULL,
    username    TEXT NOT NULL,
    thumb       TEXT,
    is_admin    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    last_login  REAL
);

CREATE TABLE IF NOT EXISTS scans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    rating_key      TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    file_size       INTEGER,
    file_mtime      REAL,
    subtitle_source TEXT,          -- option id, e.g. sidecar:Movie.en.srt or stream:3
    subtitle_label  TEXT,
    status          TEXT NOT NULL, -- running | done | failed
    error           TEXT,
    cues_json       TEXT,          -- [{i,start,end,text}]
    hits_json       TEXT,          -- [{group,cue,start,end,cs,ce,text}]
    created_by      INTEGER,
    created_at      REAL NOT NULL,
    finished_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_scans_key ON scans(rating_key);

CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    rating_key       TEXT NOT NULL,
    title            TEXT NOT NULL,
    year             INTEGER,
    user_id          INTEGER,
    scan_id          INTEGER,
    status           TEXT NOT NULL,   -- queued|analyzing|needs_review|approved|rendering|publishing|done|failed|canceled
    stage            TEXT,
    progress         REAL NOT NULL DEFAULT 0,
    settings_json    TEXT NOT NULL,   -- {groups, custom_words, style}
    media_json       TEXT,            -- probe summary used by render/preview
    output_path      TEXT,
    error            TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_key ON jobs(rating_key);

CREATE TABLE IF NOT EXISTS job_hits (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    source      TEXT NOT NULL,   -- subtitle | audio | both
    group_id    TEXT NOT NULL,
    word        TEXT,            -- what was matched (audio text if heard, else subtitle text)
    cue_index   INTEGER,
    cue_start   REAL,
    cue_end     REAL,
    cue_cs      INTEGER,         -- char span inside the cue text (for subtitle censoring)
    cue_ce      INTEGER,
    line_text   TEXT,
    status      TEXT NOT NULL,   -- confirmed | unconfirmed | audio_only
    word_start  REAL,
    word_end    REAL,
    mute_start  REAL,
    mute_end    REAL,
    confidence  REAL,
    decision    TEXT NOT NULL,   -- mute | mute_line | skip | pending
    nudge_start REAL NOT NULL DEFAULT 0,  -- seconds added before the span (review tweak)
    nudge_end   REAL NOT NULL DEFAULT 0,  -- seconds added after the span
    next_word   REAL                      -- start of the following spoken word (limits echo cleanup)
);
CREATE INDEX IF NOT EXISTS idx_hits_job ON job_hits(job_id);

CREATE TABLE IF NOT EXISTS edits (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    start   REAL NOT NULL,
    "end"   REAL NOT NULL,
    kind    TEXT NOT NULL,       -- mute | bleep | cut | blur
    reason  TEXT,
    hit_id  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_edits_job ON edits(job_id);

CREATE TABLE IF NOT EXISTS window_cache (
    file_key     TEXT NOT NULL,
    window_start REAL NOT NULL,
    window_end   REAL NOT NULL,
    words_json   TEXT NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (file_key, window_start, window_end)
);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or settings.db_path, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(path: str | None = None) -> None:
    settings.ensure_dirs()
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
    finally:
        conn.close()


MIGRATIONS = {
    "job_hits": [
        ("nudge_start", "REAL NOT NULL DEFAULT 0"),
        ("nudge_end", "REAL NOT NULL DEFAULT 0"),
        ("next_word", "REAL"),
    ],
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created."""
    for table, cols in MIGRATIONS.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, decl in cols:
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


@contextmanager
def session(path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def get_conn() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def rows(conn: sqlite3.Connection, sql: str, args: tuple | list = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def row(conn: sqlite3.Connection, sql: str, args: tuple | list = ()) -> dict[str, Any] | None:
    r = conn.execute(sql, args).fetchone()
    return dict(r) if r else None


def execute(conn: sqlite3.Connection, sql: str, args: tuple | list = ()) -> int:
    cur = conn.execute(sql, args)
    return cur.lastrowid


def now() -> float:
    return time.time()


def update_job(conn: sqlite3.Connection, job_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", [*fields.values(), job_id])


def loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    return json.loads(text)


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))
