"""Background worker: one job at a time, lowest CPU priority, survives restarts."""
from __future__ import annotations

import logging
import os
import shutil
import signal
import time
import traceback

from . import db
from .config import settings
from .jobs import Engine, analyze, render_job, work_dir
from .pipeline.media import Canceled

log = logging.getLogger("worker")
_stop = False


def _handle_stop(*_):
    global _stop
    _stop = True


def recover(conn) -> None:
    """Jobs interrupted by a restart go back into the queue at the step they were on."""
    conn.execute("UPDATE jobs SET status='queued', stage='Re-queued after restart' WHERE status='analyzing'")
    conn.execute("UPDATE jobs SET status='approved', stage='Re-queued after restart' WHERE status IN ('rendering','publishing')")


def claim(conn) -> dict | None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        job = db.row(
            conn,
            "SELECT * FROM jobs WHERE status IN ('queued','approved') AND cancel_requested = 0 ORDER BY id LIMIT 1",
        )
        if job:
            new_status = "analyzing" if job["status"] == "queued" else "rendering"
            conn.execute(
                "UPDATE jobs SET status=?, started_at=COALESCE(started_at, ?), updated_at=?, error=NULL WHERE id=?",
                (new_status, db.now(), db.now(), job["id"]),
            )
            job["status"] = new_status
        conn.execute("COMMIT")
        return job
    except Exception:
        conn.execute("ROLLBACK")
        raise


def run_one(conn, job: dict, engine: Engine) -> None:
    try:
        if job["status"] == "analyzing":
            status = analyze(conn, job, engine, settings)
            if status == "approved":
                return  # the next loop claims it for rendering
        else:
            render_job(conn, job, engine, settings)
    except Canceled:
        log.info("job %s canceled", job["id"])
        db.update_job(conn, job["id"], status="canceled", stage="Canceled", finished_at=db.now())
        shutil.rmtree(work_dir(settings, job["id"]), ignore_errors=True)
    except Exception as e:
        log.error("job %s failed: %s\n%s", job["id"], e, traceback.format_exc())
        db.update_job(conn, job["id"], status="failed", error=str(e)[:2000], stage="Failed", finished_at=db.now())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.cpu_threads))
    os.environ.setdefault("HF_HOME", settings.models_dir)
    db.init_db()
    engine = Engine(settings)
    with db.session() as conn:
        recover(conn)
    log.info("worker started (model=%s, threads=%d)", settings.whisper_model, settings.cpu_threads)
    while not _stop:
        with db.session() as conn:
            job = claim(conn)
            if job:
                log.info("job %s: %s '%s'", job["id"], job["status"], job["title"])
                run_one(conn, job, engine)
                continue
        time.sleep(settings.worker_poll_seconds)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
