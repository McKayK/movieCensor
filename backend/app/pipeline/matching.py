"""Find swears in aligned audio words and reconcile them with subtitle hits."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .boundaries import BoundaryConfig, merge_spans, refine
from .profanity import find_matches

UNALIGNED_EXTRA = 0.12


def audio_matches(words: list[dict], groups: list[str], custom_words: list[str] = ()) -> list[dict]:
    """Run the matcher over a time-sorted word list; returns word-index ranges per match."""
    if not words:
        return []
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    for w in words:
        t = w["word"]
        offsets.append((pos, pos + len(t)))
        parts.append(t)
        pos += len(t) + 1
    text = " ".join(parts)
    out = []
    for m in find_matches(text, groups, custom_words):
        idx = [i for i, (a, b) in enumerate(offsets) if a < m.end and b > m.start]
        if not idx:
            continue
        first, last = idx[0], idx[-1]
        out.append(
            {
                "group": m.group_id,
                "first": first,
                "last": last,
                "text": " ".join(words[i]["word"] for i in range(first, last + 1)),
                "start": words[first]["start"],
                "end": words[last]["end"],
                "conf": float(np.mean([words[i].get("conf", words[i].get("prob", 0.0)) for i in range(first, last + 1)])),
                "aligned": all(words[i].get("aligned", False) for i in range(first, last + 1)),
                "estimated": any(words[i].get("est", False) for i in range(first, last + 1)),
            }
        )
    return out


def analyze_windows(
    window_words: list[list[dict]],
    groups: list[str],
    custom_words: list[str],
    read: Callable[[float, float], np.ndarray] | None,
    cfg: BoundaryConfig,
) -> list[dict]:
    """Audio matches across all windows (movie time), with refined mute spans, de-duplicated."""
    found: list[dict] = []
    for words in window_words:
        words = sorted(words, key=lambda w: w["start"])
        for am in audio_matches(words, groups, custom_words):
            if am["estimated"] and not am["aligned"]:
                continue  # no trustworthy timing: leave the subtitle hit unconfirmed for review
            lo, hi = refine(words, am["first"], am["last"], read, cfg)
            if not am["aligned"]:
                # Whisper-only timing drifts; be generous rather than leak half a word.
                lo, hi = max(0.0, lo - UNALIGNED_EXTRA), hi + UNALIGNED_EXTRA
            am["mute_start"], am["mute_end"] = lo, hi
            dup = next(
                (f for f in found if f["group"] == am["group"] and f["start"] < am["end"] and am["start"] < f["end"]),
                None,
            )
            if dup is None:
                found.append(am)
    found.sort(key=lambda a: a["start"])
    return found


def reconcile(sub_hits: list[dict], audio: list[dict], cues: dict[int, dict], tolerance: float) -> list[dict]:
    """Pair each subtitle hit with the nearest unused audio match of the same group near its cue."""
    used: set[int] = set()
    hits: list[dict] = []
    for sh in sorted(sub_hits, key=lambda h: (h["start"], h["cs"])):
        lo, hi = sh["start"] - tolerance, sh["end"] + tolerance
        center = (sh["start"] + sh["end"]) / 2
        best, best_d = None, None
        for k, am in enumerate(audio):
            if k in used or am["group"] != sh["group"]:
                continue
            if am["end"] < lo or am["start"] > hi:
                continue
            d = abs((am["start"] + am["end"]) / 2 - center)
            if best_d is None or d < best_d:
                best, best_d = k, d
        cue = cues.get(sh["cue"], {})
        base = {
            "group_id": sh["group"],
            "cue_index": sh["cue"],
            "cue_start": sh["start"],
            "cue_end": sh["end"],
            "cue_cs": sh["cs"],
            "cue_ce": sh["ce"],
            "line_text": cue.get("text"),
        }
        if best is not None:
            used.add(best)
            am = audio[best]
            hits.append({
                **base, "source": "both", "word": am["text"], "status": "confirmed",
                "word_start": am["start"], "word_end": am["end"],
                "mute_start": am["mute_start"], "mute_end": am["mute_end"],
                "confidence": am["conf"], "decision": "mute",
            })
        else:
            hits.append({
                **base, "source": "subtitle", "word": sh["text"], "status": "unconfirmed",
                "word_start": None, "word_end": None, "mute_start": None, "mute_end": None,
                "confidence": None, "decision": "pending",
            })
    for k, am in enumerate(audio):
        if k in used:
            continue
        cue = _cue_at(cues, am["start"])
        hits.append({
            "group_id": am["group"], "cue_index": cue["i"] if cue else None,
            "cue_start": cue["start"] if cue else None, "cue_end": cue["end"] if cue else None,
            "cue_cs": None, "cue_ce": None, "line_text": cue["text"] if cue else None,
            "source": "audio", "word": am["text"], "status": "audio_only",
            "word_start": am["start"], "word_end": am["end"],
            "mute_start": am["mute_start"], "mute_end": am["mute_end"],
            "confidence": am["conf"], "decision": "mute",
        })
    hits.sort(key=lambda h: (h["word_start"] if h["word_start"] is not None else h["cue_start"] or 0.0))
    return hits


def _cue_at(cues: dict[int, dict], t: float) -> dict | None:
    for c in cues.values():
        if c["start"] - 0.5 <= t <= c["end"] + 0.5:
            return c
    return None


def build_edits(hits: list[dict], style: str, merge_gap: float, line_pad: float) -> list[dict]:
    """The edit list the renderer consumes. Adjacent spans merge so fades don't stutter."""
    spans: list[tuple[float, float, dict]] = []
    for h in hits:
        d = h["decision"]
        if d == "mute" and h.get("mute_start") is not None:
            spans.append((h["mute_start"], h["mute_end"], h))
        elif d in ("mute", "mute_line") and h.get("cue_start") is not None:
            spans.append((max(0.0, h["cue_start"] - line_pad), h["cue_end"] + line_pad, h))
    kind = "bleep" if style == "bleep" else "mute"
    edits = []
    for s, e, payload in merge_spans(spans, merge_gap):
        edits.append({
            "start": round(s, 4), "end": round(e, 4), "kind": kind,
            "reason": ", ".join(sorted({p["group_id"] for p in payload})),
            "hit_ids": [p.get("id") for p in payload if p.get("id") is not None],
        })
    return edits
