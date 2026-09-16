"""Find, extract, parse and censor subtitles."""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

import pysubs2

from . import media
from .profanity import find_matches, mask_text

SUB_EXTS = (".srt", ".ass", ".ssa", ".vtt")
LANG_TOKENS = {"en", "eng", "english"}
SDH_TOKENS = {"sdh", "hi", "cc"}
FORCED_TOKENS = {"forced", "foreign"}


@dataclass
class Cue:
    i: int
    start: float
    end: float
    text: str  # cleaned plain text used for matching and display

    def as_dict(self) -> dict:
        return {"i": self.i, "start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


def list_options(video_path: str, info: dict) -> list[dict]:
    """All subtitle sources for a movie, best first. Only text sources are usable for scanning."""
    opts: list[dict] = []
    folder = os.path.dirname(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    try:
        names = os.listdir(folder)
    except OSError:
        names = []
    for name in names:
        low = name.lower()
        if not low.endswith(SUB_EXTS) or not low.startswith(stem.lower()):
            continue
        tokens = [t for t in re.split(r"[.\s_\-\[\]()]+", low[len(stem):].rsplit(".", 1)[0]) if t]
        lang = "en" if (not tokens or any(t in LANG_TOKENS for t in tokens)) else tokens[0]
        opts.append(
            {
                "id": f"sidecar:{name}",
                "kind": "sidecar",
                "label": f"External: {name}",
                "language": lang,
                "sdh": any(t in SDH_TOKENS for t in tokens),
                "forced": any(t in FORCED_TOKENS for t in tokens),
                "text": True,
            }
        )
    for s in media.subtitle_streams(info):
        label_bits = [s["language"].upper(), s["codec"]]
        if s["title"]:
            label_bits.append(s["title"])
        opts.append(
            {
                "id": f"stream:{s['index']}",
                "kind": "embedded",
                "label": "Embedded: " + " · ".join(label_bits),
                "language": s["language"],
                "sdh": s["sdh"],
                "forced": s["forced"],
                "text": s["text"],
            }
        )

    def score(o: dict) -> tuple:
        english = o["language"] in LANG_TOKENS or o["language"] in ("und", "en")
        return (o["text"], english, not o["forced"], o["sdh"], o["kind"] == "sidecar")

    opts.sort(key=score, reverse=True)
    return opts


def best_option(options: list[dict]) -> dict | None:
    for o in options:
        if o["text"] and not o["forced"]:
            return o
    return None


def load_cues(video_path: str, option_id: str, work_dir: str) -> list[Cue]:
    if option_id.startswith("sidecar:"):
        path = os.path.join(os.path.dirname(video_path), option_id.split(":", 1)[1])
        subs = _load_file(path)
    elif option_id.startswith("stream:"):
        index = int(option_id.split(":", 1)[1])
        os.makedirs(work_dir, exist_ok=True)
        out = os.path.join(work_dir, f"stream{index}.srt")
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", video_path, "-map", f"0:{index}", "-c:s", "srt", out]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise media.MediaError(f"Subtitle extraction failed: {proc.stderr.strip()[-500:]}")
        subs = _load_file(out)
    else:
        raise ValueError(f"Unknown subtitle option {option_id}")
    return cues_from_subs(subs)


def _load_file(path: str) -> pysubs2.SSAFile:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pysubs2.load(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise media.MediaError(f"Could not decode subtitles {path}")


_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")
_TAGS = re.compile(r"</?[a-zA-Z][^>]*>")


def clean_text(raw: str) -> str:
    text = _TAGS.sub("", raw)
    text = _BRACKETS.sub("", text)
    text = text.replace("\\N", "\n").replace("\\n", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def cues_from_subs(subs: pysubs2.SSAFile) -> list[Cue]:
    cues: list[Cue] = []
    for ev in subs.events:
        if ev.is_comment:
            continue
        text = clean_text(ev.plaintext)
        if not text:
            continue
        cues.append(Cue(i=len(cues), start=ev.start / 1000.0, end=ev.end / 1000.0, text=text))
    cues.sort(key=lambda c: c.start)
    for i, c in enumerate(cues):
        c.i = i
    return cues


def scan_cues(cues: list[dict], groups: list[str], custom_words: list[str] = ()) -> list[dict]:
    hits: list[dict] = []
    for c in cues:
        for m in find_matches(c["text"], groups, custom_words):
            hits.append(
                {
                    "group": m.group_id,
                    "cue": c["i"],
                    "start": c["start"],
                    "end": c["end"],
                    "cs": m.start,
                    "ce": m.end,
                    "text": m.text,
                }
            )
    return hits


def write_censored_srt(cues: list[dict], spans_by_cue: dict[int, list[tuple[int, int]]], out_path: str) -> None:
    subs = pysubs2.SSAFile()
    for c in cues:
        text = mask_text(c["text"], spans_by_cue.get(c["i"], []))
        ev = pysubs2.SSAEvent(start=int(round(c["start"] * 1000)), end=int(round(c["end"] * 1000)))
        ev.plaintext = text
        subs.events.append(ev)
    subs.save(out_path, format_="srt", encoding="utf-8")
