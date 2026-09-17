"""Job orchestration: analyze (windows -> Whisper -> alignment -> matching) and render (audio, subs, remux, publish)."""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from typing import Any

from . import db
from .config import Settings, settings as default_settings
from .paths import file_key
from .pipeline import media, publish, render, subtitles
from .pipeline.boundaries import BoundaryConfig
from .pipeline.matching import analyze_windows, build_edits, reconcile
from .pipeline.transcribe import prompt_for
from .pipeline.windows import build_windows

log = logging.getLogger("jobs")

HIT_COLUMNS = [
    "source", "group_id", "word", "cue_index", "cue_start", "cue_end", "cue_cs", "cue_ce", "line_text",
    "status", "word_start", "word_end", "mute_start", "mute_end", "confidence", "decision", "next_word",
]


class Engine:
    """Creates the heavy models on demand. Tests swap this for fakes."""

    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg

    def transcriber(self):
        from .pipeline.transcribe import Transcriber

        return Transcriber(self.cfg.whisper_model, self.cfg.whisper_device, self.cfg.whisper_compute_type,
                           self.cfg.cpu_threads, self.cfg.whisper_beam_size, self.cfg.models_dir)

    def aligner(self):
        from .pipeline.align import Aligner

        return Aligner(self.cfg.align_model, self.cfg.cpu_threads, self.cfg.models_dir)

    def vocal_remover(self):
        from .pipeline.separate import DemucsRemover

        return DemucsRemover(self.cfg.demucs_model, self.cfg.cpu_threads, self.cfg.models_dir)

    def plex(self):
        from .plex import server

        return server()


class JobContext:
    def __init__(self, conn: sqlite3.Connection, job_id: int):
        self.conn = conn
        self.job_id = job_id
        self._last_write = 0.0
        self._last_cancel_check = 0.0
        self._canceled = False

    def stage(self, text: str, progress: float | None = None) -> None:
        fields: dict[str, Any] = {"stage": text}
        if progress is not None:
            fields["progress"] = round(progress, 4)
        db.update_job(self.conn, self.job_id, **fields)
        self._last_write = time.time()

    def progress(self, lo: float, hi: float):
        def cb(frac: float) -> None:
            if time.time() - self._last_write < 1.0:
                return
            db.update_job(self.conn, self.job_id, progress=round(lo + (hi - lo) * frac, 4))
            self._last_write = time.time()

        return cb

    def canceled(self) -> bool:
        if self._canceled:
            return True
        if time.time() - self._last_cancel_check < 1.5:
            return False
        self._last_cancel_check = time.time()
        r = db.row(self.conn, "SELECT cancel_requested FROM jobs WHERE id = ?", (self.job_id,))
        self._canceled = bool(r and r["cancel_requested"])
        return self._canceled

    def check_cancel(self) -> None:
        self._last_cancel_check = 0.0
        if self.canceled():
            raise media.Canceled()


def work_dir(cfg: Settings, job_id: int) -> str:
    return os.path.join(cfg.work_dir, f"job-{job_id}")


# ---------------------------------------------------------------------------------------------------------
def analyze(conn: sqlite3.Connection, job: dict, engine: Engine, cfg: Settings = default_settings) -> str:
    ctx = JobContext(conn, job["id"])
    scan = db.row(conn, "SELECT * FROM scans WHERE id = ?", (job["scan_id"],))
    if not scan or scan["status"] != "done":
        raise RuntimeError("Subtitle scan is missing or incomplete")
    path = scan["file_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Movie file not reachable from the container: {path}")
    opts = db.loads(job["settings_json"], {})
    groups, custom, style = opts.get("groups", []), opts.get("custom_words", []), opts.get("style", "mute")
    cues = db.loads(scan["cues_json"], [])
    cue_map = {c["i"]: c for c in cues}

    ctx.stage("Reading media info", 0.01)
    info = media.probe(path)
    audio = media.choose_audio(info)
    duration = media.duration_of(info)
    job_media = {
        "path": path,
        "size": os.path.getsize(path),
        "streamIndex": audio["index"],
        "channels": audio["channels"],
        "sampleRate": audio["sampleRate"],
        "codec": audio["codec"],
        "duration": duration,
        "guids": opts.get("guids", []),
    }
    db.update_job(conn, job["id"], media_json=db.dumps(job_media))

    sub_hits = subtitles.scan_cues(cues, groups, custom)
    windows = build_windows([(h["start"], h["end"]) for h in sub_hits], cfg.window_pad, cfg.window_merge_gap,
                            duration or None, cfg.window_max)
    window_words: list[list[dict]] = []
    failed_windows = 0
    reader = None

    if windows:
        wdir = work_dir(cfg, job["id"])
        os.makedirs(wdir, exist_ok=True)
        wav = os.path.join(wdir, "dialog.wav")
        ctx.stage("Extracting dialogue audio", 0.02)
        if not os.path.exists(wav) or os.path.getsize(wav) < 1024:
            media.extract_dialog_wav(path, audio["index"], audio["channels"], wav, duration,
                                     ctx.progress(0.02, 0.20), ctx.canceled)
        ctx.check_cancel()
        reader = media.WavReader(wav)

        fkey = f"{file_key(path)}|{audio['index']}|{cfg.whisper_model}|{cfg.align_model}|{cfg.whisper_prompt_mode}"
        pending: list[tuple[float, float]] = []
        for ws, we in windows:
            cached = db.row(conn, "SELECT words_json FROM window_cache WHERE file_key=? AND window_start=? AND window_end=?",
                            (fkey, ws, we))
            if cached:
                window_words.append(db.loads(cached["words_json"], []))
            else:
                pending.append((ws, we))

        if pending:
            total_audio = sum(we - ws for ws, we in pending)
            ctx.stage(f"Loading speech model ({cfg.whisper_model})", 0.20)
            transcriber = engine.transcriber()
            raw: list[tuple[float, float, list[dict]]] = []
            try:
                done = 0.0
                for k, (ws, we) in enumerate(pending):
                    ctx.check_cancel()
                    ctx.stage(f"Transcribing window {k + 1} of {len(pending)}", 0.20 + 0.55 * done / total_audio)
                    clip = reader.read(ws, we)
                    texts = [c["text"] for c in cues if c["end"] >= ws and c["start"] <= we]
                    try:
                        segs = transcriber.transcribe(clip, prompt_for(cfg.whisper_prompt_mode, texts))
                    except Exception as e:  # one bad window shouldn't sink the job; its hits go to review
                        log.warning("job %s: window %.1f-%.1f failed to transcribe: %s", job["id"], ws, we, e)
                        segs = None
                    if segs is not None:
                        raw.append((ws, we, segs))
                    else:
                        failed_windows += 1
                    done += we - ws
            finally:
                transcriber.close()

            ctx.stage("Loading alignment model", 0.75)
            aligner = engine.aligner()
            try:
                for k, (ws, we, segs) in enumerate(raw):
                    ctx.check_cancel()
                    ctx.stage(f"Aligning window {k + 1} of {len(raw)}", 0.75 + 0.2 * k / max(1, len(raw)))
                    words = aligner.align(reader.read(ws, we), segs) if segs else []
                    for w in words:
                        for key in ("start", "end", "whisper_start", "whisper_end"):
                            if key in w and w[key] is not None:
                                w[key] = round(w[key] + ws, 4)
                    conn.execute(
                        "INSERT OR REPLACE INTO window_cache (file_key, window_start, window_end, words_json, created_at) "
                        "VALUES (?,?,?,?,?)",
                        (fkey, ws, we, db.dumps(words), db.now()),
                    )
                    window_words.append(words)
            finally:
                aligner.close()

    ctx.stage("Matching words", 0.96)
    audio_hits = analyze_windows(window_words, groups, custom, reader.read if reader else None,
                                 BoundaryConfig.from_settings(cfg))
    hits = reconcile(sub_hits, audio_hits, cue_map, cfg.match_tolerance)
    save_hits(conn, job["id"], hits)

    needs_review = cfg.review_always or any(h["decision"] == "pending" for h in hits)
    status = "needs_review" if needs_review else "approved"
    confirmed = sum(1 for h in hits if h["status"] == "confirmed")
    unconfirmed = sum(1 for h in hits if h["status"] == "unconfirmed")
    extra = sum(1 for h in hits if h["status"] == "audio_only")
    summary = f"{confirmed} confirmed, {unconfirmed} unconfirmed, {extra} heard only in audio"
    if failed_windows:
        summary += f" ({failed_windows} windows could not be transcribed)"
    db.update_job(conn, job["id"], status=status, progress=1.0 if needs_review else 0.0,
                  stage=("Waiting for review: " if needs_review else "Queued for render: ") + summary)
    log.info("job %s analyzed: %s (style=%s)", job["id"], summary, style)
    return status


def save_hits(conn: sqlite3.Connection, job_id: int, hits: list[dict]) -> None:
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM job_hits WHERE job_id = ?", (job_id,))
        cols = ", ".join(HIT_COLUMNS)
        marks = ", ".join("?" for _ in HIT_COLUMNS)
        for h in hits:
            conn.execute(f"INSERT INTO job_hits (job_id, {cols}) VALUES (?, {marks})",
                         [job_id, *[h.get(c) for c in HIT_COLUMNS]])
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def job_edits(job: dict, hits: list[dict], cfg: Settings = default_settings) -> list[dict]:
    opts = db.loads(job["settings_json"], {})
    resolved = []
    for h in hits:
        h = dict(h)
        if h["decision"] == "pending":  # safety net: never ship an unreviewed hit unmuted
            h["decision"] = "mute_line"
        resolved.append(h)
    return build_edits(resolved, opts.get("style", "mute"), cfg.merge_gap, cfg.line_pad, cfg.echo_tail,
                       cfg.duck_fade + 0.03)


# ---------------------------------------------------------------------------------------------------------
def render_job(conn: sqlite3.Connection, job: dict, engine: Engine, cfg: Settings = default_settings) -> str:
    ctx = JobContext(conn, job["id"])
    jm = db.loads(job["media_json"], {})
    path = jm.get("path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Movie file not reachable: {path}")
    scan = db.row(conn, "SELECT * FROM scans WHERE id = ?", (job["scan_id"],))
    cues = db.loads(scan["cues_json"], []) if scan else []
    hits = db.rows(conn, "SELECT * FROM job_hits WHERE job_id = ? ORDER BY id", (job["id"],))

    edits = job_edits(job, hits, cfg)
    conn.execute("DELETE FROM edits WHERE job_id = ?", (job["id"],))
    for e in edits:
        conn.execute('INSERT INTO edits (job_id, start, "end", kind, reason, hit_id) VALUES (?,?,?,?,?,?)',
                     (job["id"], e["start"], e["end"], e["kind"], e["reason"], (e["hit_ids"] or [None])[0]))

    publish.check_space(cfg.censored_dir, int(jm.get("size", 0) * 1.1))
    wdir = work_dir(cfg, job["id"])
    os.makedirs(wdir, exist_ok=True)
    audio_out = os.path.join(wdir, "audio.mka")
    opts = db.loads(job["settings_json"], {})
    rcfg = render.RenderConfig.from_settings(cfg, opts.get("echo"))
    notes: list[str] = []

    remover = None
    if rcfg.echo_mode == "deep" and edits:
        ctx.stage(f"Loading voice separation model ({cfg.demucs_model})", 0.0)
        try:
            remover = engine.vocal_remover()
        except Exception as e:
            log.warning("job %s: Demucs unavailable, falling back to ducking: %s", job["id"], e)
            notes.append("echo cleanup fell back to ducking (Demucs failed to load)")
            rcfg.echo_mode = "duck"

    label = {"off": "", "duck": ", ducking echoes", "deep": ", removing voice echoes"}[rcfg.echo_mode]
    ctx.stage(f"Rendering audio ({len(edits)} edits{label})", 0.0)

    def region_done(done: int, total: int) -> None:
        ctx.stage(f"Rendering audio ({len(edits)} edits, cleaned {done} of {total} echo regions)")

    try:
        failed = render.render_audio(path, jm["streamIndex"], jm["channels"], jm["sampleRate"],
                                     jm.get("duration") or 0.0, edits, audio_out, rcfg, ctx.progress(0.0, 0.55),
                                     ctx.canceled, remover, region_done if remover else None)
    finally:
        if remover is not None:
            remover.close()
    if failed:
        notes.append(f"{failed} echo regions fell back to ducking")
    ctx.check_cancel()

    srt_path = None
    if cues:
        spans: dict[int, list[tuple[int, int]]] = {}
        for h in hits:
            if h["cue_cs"] is not None and h["decision"] != "skip":
                spans.setdefault(h["cue_index"], []).append((h["cue_cs"], h["cue_ce"]))
        srt_path = os.path.join(wdir, "censored.srt")
        subtitles.write_censored_srt(cues, spans, srt_path)

    tmp = publish.temp_path(cfg.censored_dir, job["id"])
    ctx.stage("Building the censored file", 0.55)
    try:
        render.mux(path, audio_out, srt_path, tmp, f"{job['title']} (Censored)", jm.get("duration"),
                   ctx.progress(0.55, 0.97), ctx.canceled)
        ctx.check_cancel()
        db.update_job(conn, job["id"], status="publishing", stage="Moving into the Censored library", progress=0.97)
        folder, filename = publish.output_names(job["title"], job["year"], jm.get("guids", []))
        dest = publish.place(tmp, cfg.censored_dir, folder, filename)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    warning = publish.plex_scan(engine.plex(), os.path.dirname(dest), cfg.censored_dir, cfg.censored_host_path)
    db.update_job(conn, job["id"], status="done", progress=1.0, output_path=dest, finished_at=db.now(),
                  stage="Done" + "".join(f" — {n}" for n in notes + ([warning] if warning else [])))
    if not cfg.keep_work_files:
        shutil.rmtree(wdir, ignore_errors=True)
    log.info("job %s done -> %s", job["id"], dest)
    return dest
