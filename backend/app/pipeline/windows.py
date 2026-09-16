"""Turn flagged subtitle cues into merged audio windows to transcribe."""
from __future__ import annotations


def build_windows(
    spans: list[tuple[float, float]],
    pad: float,
    merge_gap: float,
    duration: float | None = None,
    max_len: float = 120.0,
) -> list[tuple[float, float]]:
    """Pad each (start, end), merge overlaps/near neighbors, and cap window length.

    Windows longer than max_len are split into equal parts so memory and per-window time stay bounded.
    """
    if not spans:
        return []
    padded = sorted((max(0.0, s - pad), e + pad) for s, e in spans)
    if duration:
        padded = [(s, min(e, duration)) for s, e in padded if s < duration]
    merged: list[list[float]] = []
    for s, e in padded:
        if merged and s <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: list[tuple[float, float]] = []
    for s, e in merged:
        length = e - s
        if length <= max_len:
            out.append((round(s, 3), round(e, 3)))
            continue
        parts = int(length // max_len) + 1
        step = length / parts
        for k in range(parts):
            # 1 s overlap so a word on a split boundary is fully inside one part.
            ps = s + k * step - (1.0 if k else 0.0)
            pe = s + (k + 1) * step
            out.append((round(max(s, ps), 3), round(min(e, pe), 3)))
    return out
