"""Censor one file from the command line, no Plex or web UI needed. Handy for tuning on the laptop.

    python -m app.tools.censor_file "Heat (1995).mkv" --groups f_word,s_word --out ./out
    python -m app.tools.censor_file movie.mkv --preset standard --subs movie.en.srt --unconfirmed mute_line

Uses the exact same analyze/render code as the worker, with a temporary database.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

from .. import db
from ..config import Settings
from ..jobs import Engine, analyze, render_job
from ..pipeline import media, subtitles
from ..pipeline.profanity import PRESETS, WORD_GROUPS


class _NoPlex:
    def censored_section_id(self):
        return None


class _LocalEngine(Engine):
    def plex(self):
        return _NoPlex()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--groups", default="", help="comma list: " + ",".join(g["id"] for g in WORD_GROUPS))
    ap.add_argument("--preset", choices=[p["id"] for p in PRESETS])
    ap.add_argument("--custom", default="", help="comma list of extra words (use * for wildcard)")
    ap.add_argument("--subs", help="subtitle file (default: best sidecar or embedded text track)")
    ap.add_argument("--style", choices=["mute", "bleep"], default="mute")
    ap.add_argument("--unconfirmed", choices=["mute_line", "skip"], default="mute_line")
    ap.add_argument("--out", default="censored-output")
    ap.add_argument("--title")
    ap.add_argument("--year", type=int)
    args = ap.parse_args()

    groups = [g for g in args.groups.split(",") if g]
    if args.preset:
        groups = next(p["groups"] for p in PRESETS if p["id"] == args.preset)
    custom = [w for w in args.custom.split(",") if w.strip()]
    if not groups and not custom:
        ap.error("pick --groups, --preset or --custom")

    movie = os.path.abspath(args.movie)
    cfg = Settings()
    tmp = tempfile.mkdtemp(prefix="censor-")
    cfg.data_dir = os.environ.get("DATA_DIR", tmp)
    cfg.censored_dir = os.path.abspath(args.out)
    cfg.ensure_dirs()
    dbp = os.path.join(tmp, "cli.db")
    db.init_db(dbp)
    conn = db.connect(dbp)

    if args.subs:
        cues = [c.as_dict() for c in subtitles.cues_from_subs(subtitles._load_file(args.subs))]
    else:
        info = media.probe(movie)
        opt = subtitles.best_option(subtitles.list_options(movie, info))
        if not opt:
            raise SystemExit("No text subtitles found; pass --subs")
        print(f"subtitles: {opt['label']}")
        cues = [c.as_dict() for c in subtitles.load_cues(movie, opt["id"], os.path.join(tmp, "subs"))]
    print(f"{len(cues)} subtitle lines, {len(subtitles.scan_cues(cues, groups, custom))} flagged")

    st = os.stat(movie)
    scan_id = db.execute(conn, "INSERT INTO scans (rating_key, file_path, file_size, file_mtime, status, cues_json, created_at) "
                               "VALUES ('cli', ?, ?, ?, 'done', ?, ?)", (movie, st.st_size, int(st.st_mtime), json.dumps(cues), db.now()))
    title = args.title or os.path.splitext(os.path.basename(movie))[0]
    job_id = db.execute(conn, "INSERT INTO jobs (rating_key, title, year, scan_id, status, settings_json, created_at, updated_at) "
                              "VALUES ('cli', ?, ?, ?, 'analyzing', ?, ?, ?)",
                        (title, args.year, scan_id, json.dumps({"groups": groups, "custom_words": custom, "style": args.style}),
                         db.now(), db.now()))
    engine = _LocalEngine(cfg)
    analyze(conn, db.row(conn, "SELECT * FROM jobs WHERE id=?", (job_id,)), engine, cfg)
    conn.execute("UPDATE job_hits SET decision=? WHERE job_id=? AND decision='pending'", (args.unconfirmed, job_id))
    for h in db.rows(conn, "SELECT * FROM job_hits WHERE job_id=? ORDER BY COALESCE(word_start, cue_start)", (job_id,)):
        t = h["word_start"] if h["word_start"] is not None else h["cue_start"]
        span = f"{h['mute_start']:.2f}-{h['mute_end']:.2f}" if h["mute_start"] is not None else "line"
        print(f"  {t:8.2f}s {h['status']:<12} {h['decision']:<9} {h['group_id']:<10} {h['word']!r} mute {span}")
    db.update_job(conn, job_id, status="rendering")
    dest = render_job(conn, db.row(conn, "SELECT * FROM jobs WHERE id=?", (job_id,)), engine, cfg)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
