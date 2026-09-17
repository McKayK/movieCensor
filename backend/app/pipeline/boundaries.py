"""Turn aligned word times into mute spans that cover the whole word.

Priority order: never leak the start of a swear > never leak the end > don't clip neighboring words.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

RATE = 16000


@dataclass
class BoundaryConfig:
    pre_pad: float = 0.12
    post_pad: float = 0.05
    min_pre_pad: float = 0.10
    min_post_pad: float = 0.04
    snap: float = 0.05
    onset_detect: bool = True
    onset_max: float = 0.20
    merge_gap: float = 0.15

    @classmethod
    def from_settings(cls, s) -> "BoundaryConfig":
        return cls(s.pre_pad, s.post_pad, s.min_pre_pad, s.min_post_pad, s.snap, s.onset_detect, s.onset_max,
                   s.merge_gap)


def padded_span(
    start: float,
    end: float,
    prev_end: float | None,
    next_start: float | None,
    cfg: BoundaryConfig,
) -> tuple[float, float]:
    """Padding weighted toward the start of the word.

    Extra padding stops at the midpoint to a neighboring word, but the minimum pads always apply,
    even when that trims the edge of the neighbor. A clipped "the" is better than an audible "F-".
    """
    lo = start - cfg.pre_pad
    if prev_end is not None:
        lo = max(lo, min((prev_end + start) / 2.0, start - cfg.min_pre_pad))
    hi = end + cfg.post_pad
    if next_start is not None:
        hi = min(hi, max((end + next_start) / 2.0, end + cfg.min_post_pad))
    return max(0.0, lo), hi


# ---- signal helpers -------------------------------------------------------------------------------------
def _frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        return np.zeros((0, frame), dtype=np.float32)
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def speech_energy(x: np.ndarray, rate: int = RATE) -> tuple[np.ndarray, float, int]:
    """Per-frame energy that also sees quiet hissy consonants.

    Plain loudness treats "f", "s", "sh" as near-silence. Pre-emphasis boosts high frequencies so those
    sounds register. Returns (energy per frame, hop seconds, frame length in samples).
    """
    frame, hop = int(0.010 * rate), int(0.005 * rate)
    if len(x) < 2:
        return np.zeros(0), hop / rate, frame
    emph = np.empty_like(x)
    emph[0] = x[0]
    emph[1:] = x[1:] - 0.95 * x[:-1]
    f = _frames(emph, frame, hop)
    return np.sqrt(np.mean(np.square(f), axis=1)) if len(f) else np.zeros(0), hop / rate, frame


def quietest_point(read: Callable[[float, float], np.ndarray], a: float, b: float, rate: int = RATE) -> float | None:
    """Center of the frame with the least speech energy (including hiss) inside [a, b]."""
    if b - a < 0.01:
        return None
    x = read(a, b)
    energy, hop_s, frame = speech_energy(x, rate)
    if len(energy) == 0:
        return None
    k = int(np.argmin(energy))
    return a + k * hop_s + frame / (2 * rate)


def spectral_flux(x: np.ndarray, rate: int = RATE, n_fft: int = 512, hop: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Onset strength per frame, weighted toward 1.5 kHz and up where consonants live.

    Returns (flux, frame start times in seconds relative to x).
    """
    if len(x) < n_fft + hop:
        return np.zeros(0), np.zeros(0)
    win = np.hanning(n_fft).astype(np.float32)
    f = _frames(x.astype(np.float32), n_fft, hop) * win
    mag = np.log1p(100.0 * np.abs(np.fft.rfft(f, axis=1)))
    diff = np.maximum(0.0, mag[1:] - mag[:-1])
    hf_bin = int(1500 * n_fft / rate)
    flux = diff[:, hf_bin:].sum(axis=1) + 0.3 * diff[:, :hf_bin].sum(axis=1)
    # Frame t+1 introduced the change; the new samples sit at the end of that window.
    times = (np.arange(1, len(mag)) * hop + n_fft - hop) / rate
    return flux, times


def detect_onset(
    read: Callable[[float, float], np.ndarray],
    start: float,
    prev_start: float | None,
    cfg: BoundaryConfig,
    rate: int = RATE,
) -> float | None:
    """Where the word really begins, if acoustics say earlier than the aligner.

    Aligners mark a consonant where it is most recognizable, which for "f"/"sh" is often 50-150 ms after the
    sound starts. We look for the nearest strong onset shortly before (or at) the aligned start.
    """
    look_a = start - cfg.onset_max
    if prev_start is not None:
        look_a = max(look_a, prev_start + 0.02)  # never walk into the previous word's own onset
    look_a = max(0.0, look_a)
    ctx_a = max(0.0, start - 0.8)
    ctx_b = start + 0.3
    x = read(ctx_a, ctx_b)
    flux, times = spectral_flux(x, rate)
    if len(flux) < 5:
        return None
    t = times + ctx_a
    in_range = (t >= look_a) & (t <= start + 0.02)
    if not in_range.any():
        return None
    candidates = np.where(in_range)[0]
    local_max = flux[candidates].max()
    floor = float(np.median(flux))
    threshold = max(2.0 * floor, 0.35 * local_max)
    if local_max <= 2.0 * floor or local_max <= 1e-6:
        return None  # nothing that stands out: trust the padding
    # Peaks: frames above threshold that are local maxima.
    peaks = [k for k in candidates
             if flux[k] >= threshold
             and flux[k] >= flux[max(0, k - 1)]
             and flux[k] >= flux[min(len(flux) - 1, k + 1)]]
    if not peaks:
        return None
    k = peaks[-1]  # nearest strong onset at or before the aligned start
    # Rewind to where the rise began so the very first bit of the consonant is inside the span.
    while k > 0 and t[k - 1] >= look_a and flux[k - 1] > floor * 1.2 and flux[k - 1] < flux[k]:
        k -= 1
    onset = float(t[k]) - 0.010
    return onset if onset < start else None


def snap_span(
    lo: float,
    hi: float,
    prev_end: float | None,
    next_start: float | None,
    read: Callable[[float, float], np.ndarray] | None,
    cfg: BoundaryConfig,
) -> tuple[float, float]:
    """Widen (never shrink) each edge to the quietest nearby point, without entering a neighbor word."""
    if read is None or cfg.snap <= 0:
        return lo, hi
    a = lo - cfg.snap
    if prev_end is not None and prev_end <= lo:
        a = max(a, prev_end)
    q = quietest_point(read, max(0.0, a), lo)
    if q is not None:
        lo = min(lo, q)
    b = hi + cfg.snap
    if next_start is not None and next_start >= hi:
        b = min(b, next_start)
    q = quietest_point(read, hi, b)
    if q is not None:
        hi = max(hi, q)
    return lo, hi


def refine(
    words: list[dict],
    first: int,
    last: int,
    read: Callable[[float, float], np.ndarray] | None,
    cfg: BoundaryConfig,
) -> tuple[float, float]:
    """Mute span for words[first..last] (inclusive) given the full, time-sorted word list."""
    start, end = words[first]["start"], words[last]["end"]
    prev_end = words[first - 1]["end"] if first > 0 else None
    prev_start = words[first - 1]["start"] if first > 0 else None
    next_start = words[last + 1]["start"] if last + 1 < len(words) else None
    lo, hi = padded_span(start, end, prev_end, next_start, cfg)
    if read is not None and cfg.onset_detect:
        onset = detect_onset(read, start, prev_start, cfg)
        if onset is not None:
            lo = min(lo, onset - 0.02)
    lo, hi = snap_span(lo, hi, prev_end, next_start, read, cfg)
    return max(0.0, lo), hi


def merge_spans(spans: list[tuple[float, float, dict]], gap: float) -> list[tuple[float, float, list[dict]]]:
    """Merge spans closer than `gap`. Each input carries a payload; merged spans keep all payloads."""
    out: list[tuple[float, float, list[dict]]] = []
    for s, e, payload in sorted(spans, key=lambda x: x[0]):
        if out and s <= out[-1][1] + gap:
            ps, pe, pl = out[-1]
            out[-1] = (ps, max(pe, e), pl + [payload])
        else:
            out.append((s, e, [payload]))
    return out
