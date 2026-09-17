import numpy as np

from app.pipeline.matching import build_edits
from app.pipeline.render import RenderConfig, apply_edits, process_stream
from tests.test_pipeline_integration import FakeRemover

RATE = 8000


def scene(seconds=4.0):
    t = np.arange(int(seconds * RATE)) / RATE
    music = 0.2 * np.sin(2 * np.pi * 110 * t)
    voice = np.zeros_like(t)
    voice[(t >= 1.0) & (t < 1.4)] = 0.5                     # the swear
    echo = np.zeros_like(t)
    echo[(t >= 1.0) & (t < 1.8)] = 0.25                     # its reverb lingers 400 ms
    v = voice * np.sin(2 * np.pi * 300 * t)
    e = echo * np.sin(2 * np.pi * 300 * t)
    x = np.stack([music + e, music + e, v + e, music * 0, music + e, music + e], axis=1).astype(np.float32)
    return t, x


def run(x, cfg, edits, chunk):
    out = []
    chunks = (x[i:i + chunk].copy() for i in range(0, len(x), chunk))
    written, failed = process_stream(chunks, RATE, 6, "5.1", edits, cfg, out.append, FakeRemover())
    return np.concatenate(out), written, failed


def band(x, freq):
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / RATE)
    return spec[(freqs > freq - 20) & (freqs < freq + 20)].sum() / len(x)


def test_deep_clean_removes_echo_but_keeps_music_and_is_chunk_independent():
    t, x = scene()
    cfg = RenderConfig(echo_mode="deep", echo_tail=0.4, fade=0.025, duck_fade=0.06)
    edits = [{"start": 0.95, "end": 1.45, "kind": "mute", "tail": 0.4}]
    y1, written, failed = run(x, cfg, edits, chunk=RATE)       # 1 s chunks
    y2, _, _ = run(x, cfg, edits, chunk=777)                   # odd chunk size
    assert written == len(x) and failed == 0
    assert np.allclose(y1, y2, atol=1e-5)
    tail = slice(int(1.5 * RATE), int(1.8 * RATE))
    assert band(x[tail, 0], 300) > 10 * band(y1[tail, 0], 300)   # echo gone from front left
    assert band(y1[tail, 0], 110) > 0.8 * band(x[tail, 0], 110)  # music still there
    word = slice(int(1.0 * RATE), int(1.4 * RATE))
    assert np.max(np.abs(y1[word, 2])) < 1e-6                     # center fully silent during the word
    before = slice(0, int(0.5 * RATE))
    assert np.allclose(y1[before], x[before])                    # untouched away from the edit


def test_duck_mode_turns_other_speakers_down_through_tail():
    t, x = scene()
    cfg = RenderConfig(echo_mode="duck", echo_tail=0.4, surround_duck=0.35, front_duck=0.25)
    y = x.copy()
    apply_edits(y, 0, RATE, [{"start": 0.95, "end": 1.45, "kind": "mute", "tail": 0.4}], "5.1", cfg)
    mid_tail = int(1.7 * RATE)
    assert abs(y[mid_tail, 0] / x[mid_tail, 0] - 0.25) < 1e-3 if abs(x[mid_tail, 0]) > 1e-3 else True
    assert abs(y[mid_tail, 4] / x[mid_tail, 4] - 0.35) < 1e-3 if abs(x[mid_tail, 4]) > 1e-3 else True
    assert np.allclose(y[:, 3], x[:, 3])  # LFE untouched


def test_tail_stops_before_next_word_and_nudges_widen():
    hits = [{"decision": "mute", "mute_start": 5.0, "mute_end": 5.4, "next_word": 5.6, "group_id": "f_word",
             "nudge_start": 0.05, "nudge_end": 0.0}]
    e = build_edits(hits, "mute", 0.15, 0.15, echo_tail=0.35, tail_guard=0.09)[0]
    assert e["start"] == 4.95 and abs(e["tail"] - 0.11) < 1e-9
