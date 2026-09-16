"""Turn aligned word times into mute spans that cover the whole word without clipping neighbors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BoundaryConfig:
    pre_pad: float = 0.08
    post_pad: float = 0.04
    min_pad: float = 0.03
    snap: float = 0.05
    merge_gap: float = 0.15

    @classmethod
    def from_settings(cls, s) -> "BoundaryConfig":
        return cls(s.pre_pad, s.post_pad, s.min_pad, s.snap, s.merge_gap)


def padded_span(
    start: float,
    end: float,
    prev_end: float | None,
    next_start: float | None,
    cfg: BoundaryConfig,
) -> tuple[float, float]:
    """Front-weighted padding. Extra padding beyond min_pad never crosses the midpoint to a neighbor."""
    lo = start - cfg.pre_pad
    if prev_end is not None:
        lo = max(lo, min((prev_end + start) / 2.0, start - cfg.min_pad))
    hi = end + cfg.post_pad
    if next_start is not None:
        hi = min(hi, max((end + next_start) / 2.0, end + cfg.min_pad))
    return max(0.0, lo), hi


def quietest_point(read: Callable[[float, float], np.ndarray], a: float, b: float, rate: int = 16000) -> float | None:
    """Center of the lowest-RMS 10 ms frame (5 ms hop) inside [a, b]."""
    if b - a < 0.01:
        return None
    x = read(a, b)
    frame, hop = int(0.010 * rate), int(0.005 * rate)
    if len(x) < frame:
        return None
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(np.square(x[idx]), axis=1))
    k = int(np.argmin(rms))
    return a + (k * hop + frame / 2) / rate


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
    next_start = words[last + 1]["start"] if last + 1 < len(words) else None
    lo, hi = padded_span(start, end, prev_end, next_start, cfg)
    return snap_span(lo, hi, prev_end, next_start, read, cfg)


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
