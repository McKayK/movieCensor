"""End-to-end analyze + render on a synthetic 5.1 movie with a fake speech engine.

Center channel: tone bursts standing in for words. Other channels: constant "music" tones.
The fake transcriber/aligner report the bursts as words, so we can check that exactly those spans are muted.
"""
import json
import os
import subprocess

import numpy as np
import pytest

from app import db
from app.config import Settings
from app.jobs import analyze, render_job
from app.pipeline import media, subtitles

RATE = 48000
DUR = 12.0
# (word, start, end) in movie time; bursts on the center channel
WORDS = [("oh", 2.00, 2.25), ("shit", 2.40, 2.80), ("get", 3.00, 3.20), ("down", 3.30, 3.60),
         ("what", 7.00, 7.20), ("the", 7.30, 7.40), ("fuck", 7.55, 7.95), ("man", 8.10, 8.40)]
SRT = """1
00:00:01,800 --> 00:00:03,900
Oh, shit! Get down!

2
00:00:06,900 --> 00:00:08,600
What the fuck, man?

3
00:00:10,000 --> 00:00:11,000
[door slams]
"""


def make_movie(folder: str) -> str:
    n = int(DUR * RATE)
    t = np.arange(n) / RATE
    center = np.zeros(n, dtype=np.float32)
    for _w, s, e in WORDS:
        a, b = int(s * RATE), int(e * RATE)
        center[a:b] = 0.5 * np.sin(2 * np.pi * 300 * t[a:b])
    music = (0.2 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    lfe = (0.3 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
    frames = np.stack([music, music, center, lfe, music, music], axis=1).astype("<f4")
    raw = os.path.join(folder, "audio.f32")
    frames.tofile(raw)
    path = os.path.join(folder, "Test Movie (2020).mkv")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=24:duration={DUR}",
        "-f", "f32le", "-ar", str(RATE), "-ac", "6", "-i", raw,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "mpeg4", "-q:v", "10", "-c:a", "ac3", "-b:a", "448k",
        path,
    ], check=True)
    with open(os.path.join(folder, "Test Movie (2020).en.sdh.srt"), "w") as f:
        f.write(SRT)
    return path


class FakeTranscriber:
    def transcribe(self, audio, prompt=None):
        # Windows start where analyze() cut them; the fake engine is told the offset via FakeEngine.
        off = FakeEngine.current_offset
        dur = len(audio) / 16000
        words = []
        for w, s, e in WORDS:
            if off <= s and e <= off + dur:
                # Whisper-style sloppiness: start 150 ms late, end 100 ms early.
                words.append({"word": w, "start": s - off + 0.15, "end": e - off - 0.1, "prob": 0.9})
        return [{"start": words[0]["start"], "end": words[-1]["end"], "text": " ".join(x["word"] for x in words), "words": words}] if words else []

    def close(self):
        pass


class FakeAligner:
    def align(self, audio, segments):
        off = FakeEngine.current_offset
        truth = {w: (s - off, e - off) for w, s, e in WORDS}
        out = []
        for seg in segments:
            for w in seg["words"]:
                s, e = truth[w["word"]]
                out.append({**w, "start": s, "end": e, "aligned": True, "conf": 0.95})
        return out

    def close(self):
        pass


class FakePlex:
    def censored_section_id(self):
        return None


class FakeEngine:
    current_offset = 0.0

    def transcriber(self):
        return FakeTranscriber()

    def aligner(self):
        return FakeAligner()

    def plex(self):
        return FakePlex()


def decode(path: str) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-af", "aformat=channel_layouts=5.1",
                          "-ar", str(RATE), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<f4").reshape(-1, 6)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


@pytest.fixture()
def env(tmp_path, monkeypatch):
    cfg = Settings()
    cfg.data_dir = str(tmp_path / "data")
    cfg.censored_dir = str(tmp_path / "censored")
    cfg.window_pad = 1.0
    cfg.ensure_dirs()
    dbp = os.path.join(cfg.data_dir, "t.db")
    db.init_db(dbp)
    conn = db.connect(dbp)
    movies = tmp_path / "movies"
    movies.mkdir()
    return cfg, conn, make_movie(str(movies))


def test_analyze_and_render(env, monkeypatch):
    cfg, conn, path = env
    info = media.probe(path)
    opts = subtitles.list_options(path, info)
    assert opts[0]["id"].startswith("sidecar:") and opts[0]["sdh"]
    cues = [c.as_dict() for c in subtitles.load_cues(path, opts[0]["id"], cfg.work_dir)]
    assert [c["text"] for c in cues] == ["Oh, shit! Get down!", "What the fuck, man?"]  # SDH-only cue dropped
    st = os.stat(path)
    scan_id = db.execute(conn, "INSERT INTO scans (rating_key, file_path, file_size, file_mtime, status, cues_json, hits_json, created_at) "
                               "VALUES ('1', ?, ?, ?, 'done', ?, '[]', ?)", (path, st.st_size, int(st.st_mtime), json.dumps(cues), db.now()))
    job_id = db.execute(conn, "INSERT INTO jobs (rating_key, title, year, scan_id, status, settings_json, created_at, updated_at) "
                              "VALUES ('1', 'Test Movie', 2020, ?, 'analyzing', ?, ?, ?)",
                        (scan_id, json.dumps({"groups": ["f_word", "s_word"], "style": "mute", "guids": ["imdb://tt0000001"]}), db.now(), db.now()))

    # Track window offsets so the fakes can place words.
    from app.pipeline import media as media_mod
    orig_read = media_mod.WavReader.read

    def tracking_read(self, start, end):
        FakeEngine.current_offset = start
        return orig_read(self, start, end)

    monkeypatch.setattr(media_mod.WavReader, "read", tracking_read)

    job = db.row(conn, "SELECT * FROM jobs WHERE id=?", (job_id,))
    status = analyze(conn, job, FakeEngine(), cfg)
    assert status == "approved"
    hits = db.rows(conn, "SELECT * FROM job_hits WHERE job_id=? ORDER BY word_start", (job_id,))
    assert [(h["word"], h["status"]) for h in hits] == [("shit", "confirmed"), ("fuck", "confirmed")]
    for h, (_, s, e) in zip(hits, [WORDS[1], WORDS[6]]):
        assert h["mute_start"] <= s - 0.03 + 1e-6 and h["mute_end"] >= e + 0.03 - 1e-6   # full word + min pad
        assert h["mute_start"] >= s - 0.08 - 0.05 - 1e-6                                   # never more than pad + snap

    job = db.row(conn, "SELECT * FROM jobs WHERE id=?", (job_id,))
    db.update_job(conn, job_id, status="rendering")
    dest = render_job(conn, job, FakeEngine(), cfg)
    assert dest.endswith("Test Movie (2020) {imdb-tt0000001} {edition-Censored}.mkv")
    assert os.path.exists(dest) and not os.listdir(os.path.join(cfg.censored_dir, ".tmp"))

    out_info = media.probe(dest)
    kinds = [(s["codec_type"], s["codec_name"]) for s in out_info["streams"]]
    assert ("audio", "eac3") in kinds and ("subtitle", "subrip") in kinds and kinds[0][0] == "video"
    assert out_info["format"]["tags"].get("title") == "Test Movie (Censored)"

    y = decode(dest)
    src = decode(path)
    assert abs(len(y) - len(src)) < RATE * 0.1
    for h in hits:
        a, b = int(h["mute_start"] * RATE), int(h["mute_end"] * RATE)
        # Lossy codec: allow a little ringing, but the word must be gone.
        assert rms(y[a + 480:b - 480, 2]) < 0.02, h["word"]
        assert rms(src[a + 480:b - 480, 2]) > 0.2
        assert rms(y[a:b, 4]) > 0.1  # surround music keeps playing
    # Neighboring words survive intact.
    for w, s, e in (WORDS[0], WORDS[2], WORDS[5], WORDS[7]):
        a, b = int((s + 0.03) * RATE), int((e - 0.03) * RATE)
        assert rms(y[a:b, 2]) > 0.25, w

    subs = subprocess.run(["ffmpeg", "-v", "error", "-i", dest, "-map", "0:s:0", "-f", "srt", "-"],
                          capture_output=True, text=True, check=True).stdout
    assert "Oh, ****! Get down!" in subs and "What the ****, man?" in subs
