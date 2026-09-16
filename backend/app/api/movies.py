"""Library browsing, posters and subtitle scans."""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response

from .. import db, plex
from ..auth import current_user
from ..config import settings
from ..paths import file_key, plex_to_container
from ..pipeline import media, subtitles
from ..pipeline.profanity import WORD_GROUPS, group_label, public_groups

log = logging.getLogger("api.movies")
router = APIRouter(prefix="/api", tags=["movies"])

_scan_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan")
_probe_cache: dict[str, dict] = {}
_probe_lock = threading.Lock()
ACTIVE = ("queued", "analyzing", "needs_review", "approved", "rendering", "publishing")


def _plex_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except plex.PlexError as e:
        raise HTTPException(502, str(e))


def cached_probe(path: str) -> dict:
    key = file_key(path)
    with _probe_lock:
        if key in _probe_cache:
            return _probe_cache[key]
    info = media.probe(path)
    with _probe_lock:
        if len(_probe_cache) > 200:
            _probe_cache.clear()
        _probe_cache[key] = info
    return info


@router.get("/wordgroups")
def wordgroups(user: dict = Depends(current_user)):
    return public_groups()


@router.get("/movies")
def list_movies(q: str = "", refresh: bool = False, user: dict = Depends(current_user),
                conn: sqlite3.Connection = Depends(db.get_conn)):
    movies = _plex_call(plex.server().movies, force=refresh)
    if q:
        needle = q.lower().strip()
        movies = [m for m in movies if needle in (m["title"] or "").lower()]
    status_by_key: dict[str, str] = {}
    for r in db.rows(conn, "SELECT rating_key, status FROM jobs ORDER BY id"):
        if r["status"] == "done" or r["status"] in ACTIVE:
            status_by_key[r["rating_key"]] = r["status"]
    return [
        {k: m[k] for k in ("ratingKey", "title", "year", "thumb", "contentRating", "resolution")}
        | {"jobStatus": status_by_key.get(m["ratingKey"])}
        for m in movies
    ]


@router.get("/thumb")
def thumb(path: str = Query(...), w: int = 300, h: int = 450, user: dict = Depends(current_user)):
    if not path.startswith("/library/"):
        raise HTTPException(400, "bad thumb path")
    data, ctype = _plex_call(plex.server().photo, path, max(50, min(w, 1000)), max(50, min(h, 1500)))
    return Response(content=data, media_type=ctype, headers={"Cache-Control": "private, max-age=86400"})


def _movie_file(rating_key: str) -> tuple[dict, str]:
    meta = _plex_call(plex.server().metadata, rating_key)
    summary = plex.summarize(meta)
    if not summary["file"]:
        raise HTTPException(404, "Plex has no file for this movie")
    return summary, plex_to_container(summary["file"])


def scan_view(scan: dict | None, include_hits: bool = True) -> dict | None:
    if not scan:
        return None
    hits = db.loads(scan["hits_json"], []) or []
    cues = db.loads(scan["cues_json"], []) or []
    counts: dict[str, int] = {}
    variants: dict[str, dict[str, int]] = {}
    for h in hits:
        counts[h["group"]] = counts.get(h["group"], 0) + 1
        v = variants.setdefault(h["group"], {})
        word = h["text"].lower()
        v[word] = v.get(word, 0) + 1
    cue_text = {c["i"]: c["text"] for c in cues}
    view = {
        "id": scan["id"],
        "status": scan["status"],
        "error": scan["error"],
        "subtitleSource": scan["subtitle_source"],
        "subtitleLabel": scan["subtitle_label"],
        "cueCount": len(cues),
        "createdAt": scan["created_at"],
        "groups": [
            {"id": g["id"], "label": g["label"], "count": counts.get(g["id"], 0), "variants": variants.get(g["id"], {})}
            for g in WORD_GROUPS
        ],
        "total": len(hits),
    }
    if include_hits:
        view["hits"] = [
            {**h, "label": group_label(h["group"]), "line": cue_text.get(h["cue"], "")} for h in hits
        ]
    return view


@router.get("/movies/{rating_key}")
def movie_detail(rating_key: str, user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(db.get_conn)):
    summary, path = _movie_file(rating_key)
    exists = os.path.exists(path)
    detail = {**summary, "containerPath": path, "fileReachable": exists}
    if exists:
        try:
            info = cached_probe(path)
            detail["audio"] = media.audio_streams(info)
            detail["chosenAudio"] = media.choose_audio(info)["index"] if detail["audio"] else None
            detail["subtitleOptions"] = subtitles.list_options(path, info)
            detail["imageSubtitlesOnly"] = bool(detail["subtitleOptions"]) and not any(
                o["text"] for o in detail["subtitleOptions"])
        except media.MediaError as e:
            detail["probeError"] = str(e)
    scan = db.row(conn, "SELECT * FROM scans WHERE rating_key = ? ORDER BY id DESC LIMIT 1", (rating_key,))
    detail["scan"] = scan_view(scan)
    detail["jobs"] = db.rows(
        conn,
        "SELECT j.id, j.status, j.stage, j.progress, j.created_at, j.output_path, u.username FROM jobs j "
        "LEFT JOIN users u ON u.id = j.user_id WHERE j.rating_key = ? ORDER BY j.id DESC",
        (rating_key,),
    )
    try:
        detail["censoredFreeGB"] = round(shutil.disk_usage(settings.censored_dir).free / 1e9, 1)
    except OSError:
        detail["censoredFreeGB"] = None
    return detail


def _run_scan(scan_id: int, path: str, option_id: str | None) -> None:
    with db.session() as conn:
        tmp = os.path.join(settings.work_dir, f"scan-{scan_id}")
        try:
            info = cached_probe(path)
            options = subtitles.list_options(path, info)
            opt = next((o for o in options if o["id"] == option_id), None) if option_id else subtitles.best_option(options)
            if not opt:
                if any(not o["text"] for o in options):
                    raise media.MediaError(
                        "Only image-based (PGS) subtitles found. Add an .srt next to the movie (Bazarr works well) and scan again.")
                raise media.MediaError("No subtitles found. Add an .srt next to the movie (Bazarr works well) and scan again.")
            if not opt["text"]:
                raise media.MediaError("That subtitle track is image-based and can't be scanned yet.")
            conn.execute("UPDATE scans SET subtitle_source=?, subtitle_label=? WHERE id=?", (opt["id"], opt["label"], scan_id))
            cues = [c.as_dict() for c in subtitles.load_cues(path, opt["id"], tmp)]
            all_groups = [g["id"] for g in WORD_GROUPS]
            hits = subtitles.scan_cues(cues, all_groups)
            conn.execute(
                "UPDATE scans SET status='done', cues_json=?, hits_json=?, finished_at=? WHERE id=?",
                (db.dumps(cues), db.dumps(hits), db.now(), scan_id),
            )
            log.info("scan %s: %d cues, %d hits (%s)", scan_id, len(cues), len(hits), opt["label"])
        except Exception as e:
            log.warning("scan %s failed: %s", scan_id, e)
            conn.execute("UPDATE scans SET status='failed', error=?, finished_at=? WHERE id=?", (str(e)[:1000], db.now(), scan_id))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@router.post("/movies/{rating_key}/scan")
def start_scan(rating_key: str, body: dict = Body(default={}), user: dict = Depends(current_user),
               conn: sqlite3.Connection = Depends(db.get_conn)):
    option_id = body.get("option") or None
    force = bool(body.get("force"))
    _summary, path = _movie_file(rating_key)
    if not os.path.exists(path):
        raise HTTPException(400, f"The movie file isn't reachable inside the container ({path}). Check PATH_MAPS and volumes.")
    st = os.stat(path)
    running = db.row(conn, "SELECT * FROM scans WHERE rating_key=? AND status='running' ORDER BY id DESC LIMIT 1", (rating_key,))
    if running:
        return scan_view(running)
    if not force:
        prev = db.row(
            conn,
            "SELECT * FROM scans WHERE rating_key=? AND status='done' AND file_path=? AND file_size=? AND file_mtime=? "
            "ORDER BY id DESC LIMIT 1",
            (rating_key, path, st.st_size, int(st.st_mtime)),
        )
        if prev and (option_id is None or prev["subtitle_source"] == option_id):
            return scan_view(prev)
    scan_id = db.execute(
        conn,
        "INSERT INTO scans (rating_key, file_path, file_size, file_mtime, status, created_by, created_at) "
        "VALUES (?,?,?,?, 'running', ?, ?)",
        (rating_key, path, st.st_size, int(st.st_mtime), user["id"], db.now()),
    )
    _scan_pool.submit(_run_scan, scan_id, path, option_id)
    return scan_view(db.row(conn, "SELECT * FROM scans WHERE id=?", (scan_id,)))


@router.get("/scans/{scan_id}")
def get_scan(scan_id: int, user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(db.get_conn)):
    scan = db.row(conn, "SELECT * FROM scans WHERE id=?", (scan_id,))
    if not scan:
        raise HTTPException(404, "scan not found")
    return scan_view(scan)
