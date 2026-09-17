import numpy as np

from app.paths import container_to_host, plex_to_container, safe_name
from app.pipeline.align import build_tokens, plausible, viterbi, word_spans
from app.pipeline.boundaries import BoundaryConfig, padded_span, quietest_point, refine
from app.pipeline.matching import build_edits, reconcile
from app.pipeline.publish import output_names
from app.pipeline.render import RenderConfig, apply_edits
from app.pipeline.windows import build_windows


def test_windows_merge_and_split():
    w = build_windows([(10, 12), (15, 16), (100, 101)], pad=6, merge_gap=2, duration=105)
    assert w == [(4.0, 22.0), (94.0, 105.0)]
    long = build_windows([(0, 300)], pad=0, merge_gap=0, max_len=120)
    assert len(long) == 3 and long[0][0] == 0 and long[-1][1] == 300


def test_padding_respects_neighbors_but_keeps_min():
    cfg = BoundaryConfig()
    # Plenty of room: full padding.
    lo, hi = padded_span(5.0, 5.4, 4.0, 6.5, cfg)
    assert abs(lo - 4.88) < 1e-9 and abs(hi - 5.45) < 1e-9
    # Tight neighbors: the minimum pads still apply (leaking the swear is worse than trimming "the")...
    lo, hi = padded_span(5.0, 5.4, 4.96, 5.44, cfg)
    assert abs(lo - 4.90) < 1e-9 and abs(hi - 5.44) < 1e-9
    # ...even when the words touch.
    lo, hi = padded_span(5.0, 5.4, 5.0, 5.4, cfg)
    assert abs(lo - 4.90) < 1e-9 and abs(hi - 5.44) < 1e-9


def test_snap_moves_to_silence_only_outward():
    rate = 16000
    audio = np.ones(rate * 2, dtype=np.float32) * 0.5
    audio[int(0.93 * rate):int(0.95 * rate)] = 0.0  # a quiet gap just before the word

    def read(a, b):
        return audio[int(a * rate):int(b * rate)]

    q = quietest_point(read, 0.90, 0.97)
    assert 0.925 <= q <= 0.955
    words = [{"start": 0.5, "end": 0.8}, {"start": 1.0, "end": 1.3}, {"start": 1.6, "end": 1.9}]
    lo, hi = refine(words, 1, 1, read, BoundaryConfig())
    assert 0.8 <= lo <= 0.9  # at least the minimum pre-pad, never back into the previous word
    assert hi >= 1.34


def test_ctc_viterbi_recovers_char_positions():
    vocab = {"<pad>": 0, "|": 1, "F": 2, "U": 3, "C": 4, "K": 5, "O": 6, "H": 7}
    tokens, ranges = build_tokens(["oh", "fuck"], vocab)
    assert ranges == [(1, 2), (4, 7)]
    T, V = 100, len(vocab)
    em = np.full((T, V), np.log(0.01))
    em[:, 0] = np.log(0.9)
    # Put a spike for each token at a known frame.
    spikes = {0: 2, 1: 10, 2: 14, 3: 30, 4: 50, 5: 55, 6: 60, 7: 70, 8: 90}
    for j, t in spikes.items():
        em[t, :] = np.log(0.01)
        em[t, tokens[j]] = np.log(0.95)
    em = em - np.logaddexp.reduce(em, axis=1, keepdims=True)
    first, last = viterbi(em, tokens, blank=0)
    assert list(first) == [spikes[j] for j in range(len(tokens))]
    spans = word_spans(em, tokens, ranges, blank=0, frame_sec=0.02)
    assert abs(spans[1]["start"] - 50 * 0.02) < 1e-9
    assert abs(spans[1]["end"] - 71 * 0.02) < 1e-9


def test_viterbi_impossible_when_too_short():
    assert viterbi(np.zeros((2, 4)), [1, 2, 3], 0) is None


def test_reconcile_statuses_and_edits():
    cues = {0: {"i": 0, "start": 10.0, "end": 12.0, "text": "oh shit"}, 1: {"i": 1, "start": 20.0, "end": 22.0, "text": "fuck"}}
    sub_hits = [
        {"group": "s_word", "cue": 0, "start": 10.0, "end": 12.0, "cs": 3, "ce": 7, "text": "shit"},
        {"group": "f_word", "cue": 1, "start": 20.0, "end": 22.0, "cs": 0, "ce": 4, "text": "fuck"},
    ]
    audio = [
        {"group": "s_word", "start": 10.5, "end": 10.8, "mute_start": 10.42, "mute_end": 10.84, "conf": 0.9, "text": "shit"},
        {"group": "b_word", "start": 30.0, "end": 30.3, "mute_start": 29.92, "mute_end": 30.34, "conf": 0.8, "text": "bitch"},
    ]
    hits = reconcile(sub_hits, audio, cues, tolerance=2.0)
    assert [h["status"] for h in hits] == ["confirmed", "unconfirmed", "audio_only"]
    assert hits[1]["decision"] == "pending"
    hits[1]["decision"] = "mute_line"
    edits = build_edits(hits, "mute", merge_gap=0.15, line_pad=0.15)
    assert all(e["tail"] == 0.35 for e in edits)
    assert [(e["start"], e["end"]) for e in edits] == [(10.42, 10.84), (19.85, 22.15), (29.92, 30.34)]


def test_envelope_is_silent_inside_span_and_fades_outside():
    rate = 1000
    x = np.ones((3000, 6), dtype=np.float32)
    edits = [{"start": 1.0, "end": 2.0, "kind": "mute"}]
    cfg = RenderConfig(fade=0.05, duck_fade=0.05, front_duck=0.25, echo_mode="off")
    # Process in two chunks to prove chunk boundaries don't matter.
    a, b = x[:1500].copy(), x[1500:].copy()
    apply_edits(a, 0, rate, edits, "5.1", cfg)
    apply_edits(b, 1500, rate, edits, "5.1", cfg)
    y = np.vstack([a, b])
    assert np.all(y[1000:2001, 2] == 0)                 # center silent for the whole span
    assert np.allclose(y[1000:2001, 0], 0.25)            # fronts ducked
    assert np.all(y[:, 3] == 1) and np.all(y[:, 4] == 1)  # LFE + surrounds untouched in "off" mode
    assert y[949, 2] == 1 and 0 < y[975, 2] < 1          # fade sits before the span
    assert y[2051, 2] == 1
    # Raised-cosine fade: smooth start (no slope jump) and halfway at the midpoint.
    assert abs(y[975, 2] - 0.5) < 0.05 and y[951, 2] > 0.99


def test_paths_and_names():
    maps = [("D:\\Media\\Movies", "/media/movies")]
    assert plex_to_container("D:\\Media\\Movies\\Heat (1995)\\Heat.mkv", maps) == "/media/movies/Heat (1995)/Heat.mkv"
    assert plex_to_container("d:/media/movies/X.mkv", maps) == "/media/movies/X.mkv"
    assert container_to_host("/media/censored/Heat (1995)", "/media/censored", "D:\\Media\\Censored") == "D:\\Media\\Censored\\Heat (1995)"
    assert safe_name('Mission: Impossible? "Rogue"') == "Mission - Impossible Rogue"
    folder, filename = output_names("Heat", 1995, ["tmdb://949", "imdb://tt0113277"])
    assert folder == "Heat (1995) {imdb-tt0113277} {edition-Censored}"
    assert filename == folder + ".mkv"


def test_alignment_sanity_check():
    assert plausible(10.02, 10.4, 10.1, 10.5)
    assert not plausible(9.0, 12.0, 10.1, 10.5)   # smeared over 3 s
    assert not plausible(12.0, 12.3, 10.1, 10.4)  # far from Whisper's guess
