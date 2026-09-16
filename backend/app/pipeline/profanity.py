"""Word groups, presets and the matcher shared by subtitle scans and audio transcripts.

Patterns run on lower-cased text. Every match is expanded to whole words, checked against a
false-positive allowlist, and overlapping matches are resolved (longest wins, then group order).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# Order matters for overlap resolution: earlier groups win ties ("goddamn" before "damn").
WORD_GROUPS: list[dict] = [
    {
        "id": "f_word",
        "label": "F-word",
        "patterns": [
            r"\w*f+u+c+k+\w*",
            r"\bf[\*#@]+(?:k|ck|in|ing|in'|er|ers|ed)?\w*",
            r"\bmother\s*-?\s*f[\*#@\w]*",
            r"\bfu[\*#@]+\w*",
        ],
    },
    {
        "id": "slurs",
        "label": "Slurs",
        "patterns": [
            r"\bn+i+g+(?:g+)?(?:a+|a+h+|a+z+|a+s+|e+r+|e+r+s+|u+h+)\b",
            r"\bn[\*#@]+(?:a|er|ers|as|az)\b",
            r"\bfag+(?:s|ot|ots|got|gots|gy)?\b",
            r"\bretard(?:s|ed)?\b",
            r"\bspics?\b",
            r"\bchinks?\b",
            r"\bkikes?\b",
            r"\btrann(?:y|ies)\b",
        ],
    },
    {
        "id": "goddamn",
        "label": "God-damn",
        "patterns": [r"\bgod\s*-?\s*damn\w*", r"\bgoddam\w*", r"\bgod\s*-?\s*dammit\b", r"\bg[\*#@]+d\s*-?\s*damn\w*"],
    },
    {
        "id": "s_word",
        "label": "S-word",
        "patterns": [r"\w*sh+i+t+\w*", r"\bsh[\*#@]+t\w*", r"\bs[\*#@]{2,}t\w*"],
    },
    {
        "id": "b_word",
        "label": "B-word",
        "patterns": [r"\w*b+i+t+c+h+\w*", r"\bb[\*#@]+ch\w*"],
    },
    {
        "id": "a_word",
        "label": "A-word",
        "patterns": [
            r"\b(?:ass|asses|arse|arses)\b",
            r"\w*ass+hole\w*",
            r"\w*arsehole\w*",
            r"\b(?:jack|dumb|bad|smart|kick|lard|fat|dip|tight|half|hard|candy|kiss|big|sweet)[\s-]?ass(?:es|ed)?\b",
            r"\ba[\*#@]+hole\w*",
        ],
    },
    {
        "id": "sexual",
        "label": "Sexual terms",
        "patterns": [r"\bwhores?\b", r"\bsluts?\b", r"\bslutty\b", r"\bblow\s*jobs?\b", r"\bhand\s*jobs?\b", r"\bjerk(?:ing)?\s+off\b"],
    },
    {
        "id": "crude",
        "label": "Crude anatomy",
        "patterns": [
            r"\bdick(?:s|head|heads|wad|wads|ish)?\b",
            r"\bcock(?:s|sucker|suckers|sucking)?\b",
            r"\bpuss(?:y|ies)\b",
            r"\btits?\b",
            r"\btitties\b",
            r"\bpricks?\b",
            r"\bcunts?\b",
            r"\btwats?\b",
            r"\bdouche(?:bag|bags|s)?\b",
        ],
    },
    {"id": "bastard", "label": "Bastard", "patterns": [r"\bbastards?\b"]},
    {"id": "damn", "label": "Damn", "patterns": [r"\bdamn\w*", r"\bdammit\b", r"\bdamnit\b"]},
    {"id": "hell", "label": "Hell", "patterns": [r"\bhell\b", r"\bhellhole\b"]},
    {"id": "crap", "label": "Crap", "patterns": [r"\bcrap(?:s|py|pier|piest|ped)?\b"]},
    {"id": "piss", "label": "Piss", "patterns": [r"\bpiss(?:ed|es|ing|er|y)?\b"]},
    {
        "id": "lords_name",
        "label": "Lord's name",
        "patterns": [
            r"\boh\s*,?\s*(?:my\s+)?god\b",
            r"\bmy\s+god\b",
            r"\bjesus(?:\s+h\.?)?(?:\s+christ)?\b",
            r"\bchrist(?:\s+almighty)?\b",
            r"\boh\s*,?\s*(?:my\s+)?lord\b",
            r"\bfor\s+god'?s\s+sake\b",
            r"\bgod\s+almighty\b",
        ],
    },
]

PRESETS: list[dict] = [
    {"id": "essential", "label": "Essential", "groups": ["f_word", "slurs"]},
    {
        "id": "standard",
        "label": "Standard",
        "groups": ["f_word", "slurs", "goddamn", "s_word", "b_word", "a_word", "sexual", "crude", "bastard"],
    },
    {
        "id": "strict",
        "label": "Strict",
        "groups": [g["id"] for g in WORD_GROUPS],
    },
]

# Whole words that contain a pattern but are clean.
ALLOWLIST = {
    "shiitake", "shitake", "shih", "niger", "nigeria", "nigerian", "nigerians", "niggle", "niggles", "niggling",
    "cockpit", "cockpits", "cockatoo", "cockatoos", "cockroach", "cockroaches", "cocktail", "cocktails",
    "cockney", "cocky", "peacock", "hitchcock", "dickens", "dickensian", "dickinson", "scunthorpe",
    "assassin", "assassins", "assassinate", "assassinated", "assassination", "passage", "embassy",
    "classic", "compass", "harass", "bass", "mass", "grass", "brass", "glass", "class", "pass", "lass",
    "crappie", "hello", "shell", "shellfish", "prickly", "titter", "tittle", "title", "titan", "titanic",
    "whoreson", "damnation", "damned", "christmas", "christopher", "christian", "christina", "christine",
    "christie", "christy", "christs", "therapist", "matsushita", "yamashita", "kinoshita", "hiroshita",
    "fuchsia", "mukluk", "spice", "spicy", "chinky", "pissarro", "cumbersome",
}

# These words from the allowlist should still be flagged when a group explicitly wants them.
GROUP_ALLOW_EXCEPTIONS = {"damn": {"damned"}}


def custom_group_id(word: str) -> str:
    return "custom:" + word.strip().lower()


def _custom_pattern(word: str) -> str:
    w = word.strip().lower()
    wild = w.endswith("*")
    w = w.rstrip("*")
    parts = [re.escape(p) for p in w.split()]
    body = r"\s+".join(parts)
    return rf"\b{body}\w*" if wild else rf"\b{body}\b"


@dataclass(frozen=True)
class Match:
    group_id: str
    start: int  # char offset in the original text
    end: int
    text: str


@lru_cache(maxsize=64)
def _compiled(groups: tuple[str, ...], custom: tuple[str, ...]) -> list[tuple[str, list[re.Pattern]]]:
    wanted = set(groups)
    out: list[tuple[str, list[re.Pattern]]] = []
    for g in WORD_GROUPS:
        if g["id"] in wanted:
            out.append((g["id"], [re.compile(p, re.IGNORECASE) for p in g["patterns"]]))
    for word in custom:
        if word.strip().strip("*"):
            out.append((custom_group_id(word), [re.compile(_custom_pattern(word), re.IGNORECASE)]))
    return out


def normalize(text: str) -> str:
    """Lowercase + unify apostrophes while keeping every character at the same index."""
    out = []
    for c in text:
        low = c.lower()
        if len(low) != 1:
            low = c
        if low in "’‘`´":
            low = "'"
        out.append(low)
    return "".join(out)


_WORD_CHARS = re.compile(r"[\w'\*#@]")


def _expand(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and _WORD_CHARS.match(text[start - 1]) and text[start - 1] != "'":
        start -= 1
    while end < len(text) and _WORD_CHARS.match(text[end]):
        end += 1
    # Trailing apostrophe only belongs to the word for "fuckin'" style endings.
    return start, end


def find_matches(text: str, groups: list[str] | tuple[str, ...], custom_words: list[str] | tuple[str, ...] = ()) -> list[Match]:
    if not text:
        return []
    norm = normalize(text)
    candidates: list[tuple[int, int, int, str, int]] = []  # (start, -length, group order, group, end)
    for gid, patterns in _compiled(tuple(sorted(groups)), tuple(custom_words)):
        for pat in patterns:
            for m in pat.finditer(norm):
                s, e = _expand(norm, m.start(), m.end())
                # Strip a trailing apostrophe that is really a closing quote ('shit').
                while e > s and norm[e - 1] == "'" and not norm[s:e - 1].endswith("in"):
                    e -= 1
                token = norm[s:e].strip("'")
                if " " not in token and "-" not in token:
                    if token in ALLOWLIST and token not in GROUP_ALLOW_EXCEPTIONS.get(gid, set()):
                        continue
                candidates.append((s, -(e - s), _group_order(gid), gid, e))
    candidates.sort()
    result: list[Match] = []
    last_end = -1
    for s, _neg, _order, gid, e in candidates:
        if s < last_end:
            continue
        result.append(Match(gid, s, e, text[s:e]))
        last_end = e
    return result


def _group_order(gid: str) -> int:
    for i, g in enumerate(WORD_GROUPS):
        if g["id"] == gid:
            return i
    return len(WORD_GROUPS)


def group_label(gid: str) -> str:
    if gid.startswith("custom:"):
        return f"Custom: {gid[7:]}"
    for g in WORD_GROUPS:
        if g["id"] == gid:
            return g["label"]
    return gid


def mask_text(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace letters in spans with '*', keeping spaces/punctuation so line breaks survive."""
    chars = list(text)
    for s, e in spans:
        for i in range(max(0, s), min(len(chars), e)):
            if chars[i].isalnum() or chars[i] in "'*#@":
                chars[i] = "*"
    return "".join(chars)


def public_groups() -> dict:
    return {
        "groups": [{"id": g["id"], "label": g["label"]} for g in WORD_GROUPS],
        "presets": PRESETS,
    }
