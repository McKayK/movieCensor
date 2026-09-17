"""CTC forced alignment with wav2vec2 for tight word boundaries.

Whisper's word timestamps come from attention and drift 100-300 ms. Here we re-align Whisper's words to
the audio character by character. The Viterbi search is plain numpy so it can be unit tested without
downloading a model.
"""
from __future__ import annotations

import gc
import logging
import re

import numpy as np

log = logging.getLogger("align")

SAMPLE_RATE = 16000
SEGMENT_MARGIN = 0.3  # seconds of extra audio around each Whisper segment

STAY, ADVANCE, REPEAT = 0, 1, 2


def build_tokens(words: list[str], vocab: dict[str, int], sep: str = "|") -> tuple[list[int], list[tuple[int, int] | None]]:
    """Token ids for the transcript plus, per word, the (first, last) token index or None if unalignable."""
    sep_id = vocab.get(sep)
    tokens: list[int] = []
    ranges: list[tuple[int, int] | None] = []
    if sep_id is not None:
        tokens.append(sep_id)
    for w in words:
        chars = [c for c in w.upper() if c in vocab and c != sep]
        if not chars:
            ranges.append(None)
            continue
        first = len(tokens)
        tokens.extend(vocab[c] for c in chars)
        ranges.append((first, len(tokens) - 1))
        if sep_id is not None:
            tokens.append(sep_id)
    return tokens, ranges


def viterbi(emission: np.ndarray, tokens: list[int], blank: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Best CTC path. Returns (first_frame, last_frame) per token, or None if impossible.

    States: 0 = nothing emitted yet, k = token k-1 has been emitted. Each frame either emits blank
    (stay), repeats the current token (repeat) or emits the next token (advance).
    """
    T = emission.shape[0]
    J = len(tokens)
    if J == 0 or T < J:
        return None
    S = J + 1
    tok = np.asarray(tokens)
    score = np.full(S, -np.inf, dtype=np.float64)
    score[0] = 0.0
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(T):
        e = emission[t]
        stay = score + e[blank]
        adv = np.full(S, -np.inf)
        adv[1:] = score[:-1] + e[tok]
        rep = np.full(S, -np.inf)
        rep[1:] = score[1:] + e[tok]
        best = stay
        choice = np.zeros(S, dtype=np.int8)
        m = adv > best
        best = np.where(m, adv, best)
        choice[m] = ADVANCE
        m = rep > best
        best = np.where(m, rep, best)
        choice[m] = REPEAT
        # rep[0] stays -inf: nothing can be repeated before the first token. Doubled letters ("LL") are
        # allowed to advance on consecutive frames; strict CTC would need a blank, but for timing it
        # makes no practical difference.
        score = best
        back[t] = choice
    if not np.isfinite(score[J]):
        return None
    first = np.full(J, -1, dtype=np.int64)
    last = np.full(J, -1, dtype=np.int64)
    j = J
    for t in range(T - 1, -1, -1):
        c = back[t, j]
        if j > 0 and c in (ADVANCE, REPEAT):
            k = j - 1
            if last[k] < 0:
                last[k] = t
            first[k] = t
        if c == ADVANCE:
            j -= 1
    if j != 0 or (first < 0).any():
        return None
    return first, last


def word_spans(
    emission: np.ndarray,
    tokens: list[int],
    ranges: list[tuple[int, int] | None],
    blank: int,
    frame_sec: float,
) -> list[dict | None]:
    path = viterbi(emission, tokens, blank)
    if path is None:
        return [None] * len(ranges)
    first, last = path
    probs = np.exp(emission)
    out: list[dict | None] = []
    for r in ranges:
        if r is None:
            out.append(None)
            continue
        a, b = r
        start = first[a] * frame_sec
        end = (last[b] + 1) * frame_sec
        conf = float(np.mean([probs[first[k], tokens[k]] for k in range(a, b + 1)]))
        out.append({"start": float(start), "end": float(end), "conf": conf})
    return out


def plausible(start: float, end: float, whisper_start: float, whisper_end: float, estimated: bool = False) -> bool:
    """Reject alignments that clearly went wrong (noise, music, wrong transcript) so we fall back to Whisper.

    Estimated word times (segment-level fallback) are too rough to compare against, so only length is checked.
    """
    dur = end - start
    if estimated:
        return 0 < dur <= 1.5
    wdur = max(0.05, whisper_end - whisper_start)
    if dur <= 0 or dur > max(1.5, 3.0 * wdur):
        return False
    return abs((start + end) / 2 - (whisper_start + whisper_end) / 2) <= 1.0


def _fill_unaligned(words: list[dict]) -> None:
    for i, w in enumerate(words):
        if w.get("aligned"):
            continue
        prev_end = next((words[k]["end"] for k in range(i - 1, -1, -1) if words[k].get("aligned")), None)
        next_start = next((words[k]["start"] for k in range(i + 1, len(words)) if words[k].get("aligned")), None)
        if prev_end is not None and w["start"] < prev_end:
            w["start"] = prev_end
        if next_start is not None and w["end"] > next_start:
            w["end"] = next_start
        if w["end"] < w["start"]:
            w["end"] = w["start"]


class Aligner:
    def __init__(self, model_name: str, threads: int = 4, cache_dir: str | None = None):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        torch.set_num_threads(max(1, threads))
        log.info("loading alignment model %s", model_name)
        self._torch = torch
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            log.info("alignment model %s not cached yet, downloading", model_name)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()
        vocab = self.processor.tokenizer.get_vocab()
        self.vocab = {k: v for k, v in vocab.items() if len(k) == 1}
        # wav2vec2-base-960h uses upper-case letters; make lookups case-insensitive.
        for k, v in list(self.vocab.items()):
            self.vocab.setdefault(k.upper(), v)
        self.blank = self.processor.tokenizer.pad_token_id or 0

    def emissions(self, audio: np.ndarray) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(audio.astype(np.float32), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(inputs.input_values).logits[0]
            return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def align(self, audio: np.ndarray, segments: list[dict]) -> list[dict]:
        """Align Whisper segments (times relative to `audio`). Returns flat words relative to `audio`."""
        total = len(audio) / SAMPLE_RATE
        result: list[dict] = []
        for seg in segments:
            words = [dict(w) for w in seg["words"]]
            for w in words:
                w["aligned"] = False
                w["whisper_start"], w["whisper_end"] = w["start"], w["end"]
            s0 = max(0.0, min(seg["start"], words[0]["start"]) - SEGMENT_MARGIN)
            s1 = min(total, max(seg["end"], words[-1]["end"]) + SEGMENT_MARGIN)
            clip = audio[int(s0 * SAMPLE_RATE) : int(s1 * SAMPLE_RATE)]
            if len(clip) >= int(0.2 * SAMPLE_RATE):
                try:
                    em = self.emissions(clip)
                    frame_sec = (len(clip) / SAMPLE_RATE) / em.shape[0]
                    texts = [re.sub(r"[^\w']", "", w["word"]) for w in words]
                    tokens, ranges = build_tokens(texts, self.vocab)
                    spans = word_spans(em, tokens, ranges, self.blank, frame_sec)
                    for w, sp in zip(words, spans):
                        if sp is not None and plausible(s0 + sp["start"], s0 + sp["end"], w["start"], w["end"],
                                                                 bool(w.get("est"))):
                            w["start"] = s0 + sp["start"]
                            w["end"] = s0 + sp["end"]
                            w["conf"] = sp["conf"]
                            w["aligned"] = True
                except Exception as e:  # never fail a job on one bad segment
                    log.warning("alignment failed for segment %.2f-%.2f: %s", s0, s1, e)
            _fill_unaligned(words)
            result.extend(words)
        result.sort(key=lambda w: w["start"])
        return result

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
