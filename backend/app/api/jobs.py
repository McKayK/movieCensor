"""Censor jobs: create, track, review, approve, cancel, preview."""
from __future__ import annotations

import os
import shutil
import sqlite3

from fastapi import APIRouter, Body, Depends, HTTPException, Response

from .. import db, plex
from ..auth import current_user
from ..config import settings
from ..jobs import job_edits, work_dir
from ..pipeline import media, render
from ..pipeline.profanity import WORD_GROUPS, group_label

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

ACTIVE = ("queued", "analyzing", "needs_review", "approved", "rendering", "publishing")
DECISIONS = ("mute", "mute_line", "skip")
VALID_GROUPS = {g["id"] for g in WORD_GROUPS}


def _job_or_404(conn: sqlite3.Connection, job_id: int) -> dict:
    job = db.row(conn, "SELECT j.*, u.username FROM jobs j LEFT JOIN users u ON u.id = j.user_id WHERE j.id = ?", (job_id,))
    if not job:
        raise HTTPException(404, "job not found")
    return job


def _can_modify(user: dict, job: dict) -> None:
    if not (user["is_admin"] or job["user_id"] == user["id"]):
        raise HTTPException(403, "Only the person who started this job (or an admin) can change it")


def job_view(job: dict) -> dict:
    opts = db.loads(job["settings_json"], {})
    return {
        "id": job["id"],
        "ratingKey": job["rating_key"],
        "title": job["title"],
        "year": job["year"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job["error"],
        "outputPath": job["output_path"],
        "username": job.get("username"),
        "userId": job["user_id"],
        "settings": {**opts, "groupLabels": [group_label(g) for g in opts.get("groups", [])]},
        "cancelRequested": bool(job["cancel_requested"]),
        "createdAt": job["created_at"],
        "updatedAt": job["updated_at"],
        "startedAt": job["started_at"],
        "finishedAt": job["finished_at"],
    }


@router.post("")
def create_job(body: dict = Body(...), user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(db.get_conn)):
    rating_key = str(body.get("ratingKey") or "")
    scan_id = body.get("scanId")
    groups = [g for g in body.get("groups", []) if g in VALID_GROUPS]
    custom = [w.strip()[:40] for w in body.get("customWords", []) if isinstance(w, str) and w.strip()][:50]
    style = body.get("style", "mute")
    if style not in ("mute", "bleep"):
        raise HTTPException(400, "style must be mute or bleep")
    if not groups and not custom:
        raise HTTPException(400, "Pick at least one word group or custom word")
    scan = db.row(conn, "SELECT * FROM scans WHERE id=? AND rating_key=?", (scan_id, rating_key))
    if not scan or scan["status"] != "done":
        raise HTTPException(400, "Run a subtitle scan first")
    if db.row(conn, f"SELECT id FROM jobs WHERE rating_key=? AND status IN {ACTIVE}", (rating_key,)):
        raise HTTPException(409, "This movie already has a job in progress")
    if not user["is_admin"]:
        active = db.row(conn, f"SELECT COUNT(*) AS n FROM jobs WHERE user_id=? AND status IN {ACTIVE}", (user["id"],))
        if active and active["n"] >= settings.max_active_jobs_per_user:
            raise HTTPException(429, f"You already have {active['n']} jobs in progress")
    try:
        meta = plex.summarize(plex.server().metadata(rating_key))
    except plex.PlexError as e:
        raise HTTPException(502, str(e))
    opts = {"groups": groups, "custom_words": custom, "style": style, "guids": meta["guids"]}
    job_id = db.execute(
        conn,
        "INSERT INTO jobs (rating_key, title, year, user_id, scan_id, status, stage, settings_json, created_at, updated_at) "
        "VALUES (?,?,?,?,?, 'queued', 'Waiting for the worker', ?, ?, ?)",
        (rating_key, meta["title"], meta["year"], user["id"], scan_id, db.dumps(opts), db.now(), db.now()),
    )
    return job_view(_job_or_404(conn, job_id))


@router.get("")
def list_jobs(mine: bool = False, user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(db.get_conn)):
    sql = "SELECT j.*, u.username FROM jobs j LEFT JOIN users u ON u.id = j.user_id"
    args: list = []
    if mine:
        sql += " WHERE j.user_id = ?"
        args.append(user["id"])
    sql += " ORDER BY j.id DESC LIMIT 200"
    return [job_view(j) for j in db.rows(conn, sql, args)]


@router.get("/{job_id}")
def get_job(job_id: int, user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(db.get_conn)):
    job = _job_or_404(conn, job_id)
    hits = db.rows(conn, "SELECT * FROM job_hits WHERE job_id=? ORDER BY COALESCE(word_start, cue_start), id", (job_id,))
    view = job_view(job)
    view["hits"] = [
        {
            "id": h["id"], "source": h["source"], "group": h["group_id"], "label": group_label(h["group_id"]),
            "word": h["word"], "line": h["line_text"], "cs": h["cue_cs"], "ce": h["cue_ce"], "cueStart": h["cue_start"], "cueEnd": h["cue_end"],
            "status": h["status"], "wordStart": h["word_start"], "wordEnd": h["word_end"],
            "muteStart": h["mute_start"], "muteEnd": h["mute_end"], "confidence": h["confidence"],
            "decision": h["decision"],
        }
        for h in hits
    ]
    view["counts"] = {
        s: sum(1 for h in hits if h["status"] == s) for s in ("confirmed", "unconfirmed", "audio_only")
    }
    view["pending"] = sum(1 for h in hits if h["decision"] == "pending")
    view["canModify"] = bool(user["is_admin"] or job["user_id"] == user["id"])
    view["editCount"] = len(job_edits(job, hits)) if hits else 0
    return view


def _set_decision(conn: sqlite3.Connection, hit: dict, decision: str) -> None:
    if decision not in DECISIONS:
        raise HTTPException(400, f"decision must be one of {DECISIONS}")
    if decision == "mute" and hit["mute_start"] is None:
        decision = "mute_line"  # nothing precise to mute for an unconfirmed hit
    if decision == "mute_line" and hit["cue_start"] is None:
        raise HTTPException(400, "This hit has no subtitle line to mute")
    conn.execute("UPDATE job_hits SET decision=? WHERE id=?", (decision, hit["id"]))


@router.post("/{job_id}/hits/{hit_id}")
def set_decision(job_id: int, hit_id: int, body: dict = Body(...), user: dict = Depends(current_user),
                 conn: sqlite3.Connection = Depends(db.get_conn)):
    job = _job_or_404(conn, job_id)
    _can_modify(user, job)
    if job["status"] != "needs_review":
        raise HTTPException(409, "Decisions can only change while the job is waiting for review")
    hit = db.row(conn, "SELECT * FROM job_hits WHERE id=? AND job_id=?", (hit_id, job_id))
    if not hit:
        raise HTTPException(404, "hit not found")
    _set_decision(conn, hit, body.get("decision", ""))
    return {"ok": True, "decision": db.row(conn, "SELECT decision FROM job_hits WHERE id=?", (hit_id,))["decision"]}


@router.post("/{job_id}/hits")
def bulk_decision(job_id: int, body: dict = Body(...), user: dict = Depends(current_user),
                  conn: sqlite3.Connection = Depends(db.get_conn)):
    job = _job_or_404(conn, job_id)
    _can_modify(user, job)
    if job["status"] != "needs_review":
        raise HTTPException(409, "Decisions can only change while the job is waiting for review")
    status = body.get("status")
    decision = body.get("decision", "")
    sql, args = "SELECT * FROM job_hits WHERE job_id=?", [job_id]
    if status:
        sql += " AND status=?"
        args.append(status)
    for hit in db.rows(conn, sql, args):
        _set_decision(conn, hit, decision)
    return {"ok": True}


@router.post("/{job_id}/approve")
def approve(job_id: int, user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(db.get_conn)):
    job = _job_or_404(conn, job_id)
    _can_modify(user, job)
    if job["status"] != "needs_review":
        raise HTTPException(409, "Job is not waiting for review")
    pending = db.row(conn, "SELECT COUNT(*) AS n FROM job_hits WHERE job_id=? AND decision='pending'", (job_id,))
    if pending and pending["n"]:
        raise HTTPException(400, f"{pending['n']} unconfirmed hits still need a decision")
    db.update_job(conn, job_id, status="approved", stage="Queued for render", progress=0.0)
    return job_view(_job_or_404(conn, job_id))


@router.post("/{job_id}/cancel")
def cancel(job_id: int, user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(db.get_conn)):
    job = _job_or_404(conn, job_id)
    _can_modify(user, job)
    if job["status"] in ("queued", "needs_review", "approved"):
        db.update_job(conn, job_id, status="canceled", stage="Canceled", finished_at=db.now())
        shutil.rmtree(work_dir(settings, job_id), ignore_errors=True)
    elif job["status"] in ("analyzing", "rendering", "publishing"):
        db.update_job(conn, job_id, cancel_requested=1, stage="Canceling…")
    else:
        raise HTTPException(409, f"Job is already {job['status']}")
    return job_view(_job_or_404(conn, job_id))


@router.post("/{job_id}/retry")
def retry(job_id: int, user: dict = Depends(current_user), conn: sqlite3.Connection = Depends(db.get_conn)):
    job = _job_or_404(conn, job_id)
    _can_modify(user, job)
    if job["status"] not in ("failed", "canceled"):
        raise HTTPException(409, "Only failed or canceled jobs can be retried")
    if db.row(conn, f"SELECT id FROM jobs WHERE rating_key=? AND status IN {ACTIVE} AND id != ?", (job["rating_key"], job_id)):
        raise HTTPException(409, "This movie already has another job in progress")
    # Re-analysis is cheap on retry: transcribed windows are cached per file.
    db.update_job(conn, job_id, status="queued", stage="Re-queued", progress=0.0, error=None, cancel_requested=0,
                  finished_at=None)
    return job_view(_job_or_404(conn, job_id))


@router.delete("/{job_id}")
def delete_job(job_id: int, delete_file: bool = False, user: dict = Depends(current_user),
               conn: sqlite3.Connection = Depends(db.get_conn)):
    job = _job_or_404(conn, job_id)
    if not user["is_admin"]:
        raise HTTPException(403, "Admin only")
    if job["status"] in ("analyzing", "rendering", "publishing"):
        raise HTTPException(409, "Cancel the job first")
    removed = None
    if delete_file and job["output_path"] and os.path.exists(job["output_path"]):
        folder = os.path.dirname(job["output_path"])
        if os.path.abspath(folder).startswith(os.path.abspath(settings.censored_dir) + os.sep):
            shutil.rmtree(folder, ignore_errors=True)
            removed = folder
    conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    shutil.rmtree(work_dir(settings, job_id), ignore_errors=True)
    return {"ok": True, "removed": removed}


@router.get("/{job_id}/hits/{hit_id}/preview")
def preview(job_id: int, hit_id: int, variant: str = "censored", user: dict = Depends(current_user),
            conn: sqlite3.Connection = Depends(db.get_conn)):
    job = _job_or_404(conn, job_id)
    hit = db.row(conn, "SELECT * FROM job_hits WHERE id=? AND job_id=?", (hit_id, job_id))
    if not hit:
        raise HTTPException(404, "hit not found")
    jm = db.loads(job["media_json"], {})
    if not jm.get("path") or not os.path.exists(jm["path"]):
        raise HTTPException(400, "Movie file not reachable")
    if hit["word_start"] is not None:
        center_lo, center_hi = hit["word_start"], hit["word_end"]
    else:
        center_lo, center_hi = hit["cue_start"] or 0.0, hit["cue_end"] or 0.0
    start = max(0.0, center_lo - 1.5)
    length = min(12.0, (center_hi - center_lo) + 3.0)
    edits = None
    if variant == "censored":
        hits = db.rows(conn, "SELECT * FROM job_hits WHERE job_id=?", (job_id,))
        edits = [e for e in job_edits(job, hits) if e["end"] >= start and e["start"] <= start + length]
    try:
        clip, rate = render.preview_clip(jm["path"], jm["streamIndex"], jm["channels"], start, length, edits,
                                         render.RenderConfig.from_settings(settings))
    except media.MediaError as e:
        raise HTTPException(500, str(e))
    return Response(content=render.wav_bytes(clip, rate), media_type="audio/wav", headers={"Cache-Control": "no-store"})
