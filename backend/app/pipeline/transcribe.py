"""faster-whisper wrapper (CPU int8 by default)."""
from __future__ import annotations

import gc
import logging

import numpy as np

log = logging.getLogger("transcribe")

GENERIC_PROMPT = (
    "Uncensored movie dialogue, transcribed verbatim with every swear word spelled out: "
    "fuck, fucking, shit, damn, goddamn, bitch, ass, hell."
)


class Transcriber:
    def __init__(self, model: str, device: str = "cpu", compute_type: str = "int8", threads: int = 4,
                 beam_size: int = 5, download_root: str | None = None):
        from faster_whisper import WhisperModel  # heavy import, keep lazy

        log.info("loading whisper model %s (%s/%s, %d threads)", model, device, compute_type, threads)
        self.model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=threads,
                                  download_root=download_root)
        self.beam_size = beam_size

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> list[dict]:
        """Whisper segments with word timestamps, times relative to the start of `audio`.

        faster-whisper's word-timestamp step can crash on some windows (IndexError "boolean index did not
        match" when cross-attention alignment comes back empty). We retry without the prompt, then fall
        back to segment-level output with estimated word times; wav2vec2 alignment fixes the timing.
        """
        try:
            return self._run(audio, prompt, word_timestamps=True)
        except (IndexError, ValueError) as e:
            log.warning("word timestamps failed (%s); retrying without prompt", e)
        try:
            return self._run(audio, None, word_timestamps=True)
        except (IndexError, ValueError) as e:
            log.warning("word timestamps failed again (%s); using segment-level timing", e)
        return self._run(audio, prompt, word_timestamps=False)

    def _run(self, audio: np.ndarray, prompt: str | None, word_timestamps: bool) -> list[dict]:
        segments, _info = self.model.transcribe(
            audio.astype(np.float32),
            language="en",
            task="transcribe",
            beam_size=self.beam_size,
            word_timestamps=word_timestamps,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            suppress_blank=True,
        )
        out: list[dict] = []
        for seg in segments:  # the generator does the work, so errors surface here
            if word_timestamps:
                words = [
                    {"word": w.word.strip(), "start": float(w.start), "end": float(w.end), "prob": float(w.probability)}
                    for w in (seg.words or [])
                    if w.word.strip()
                ]
            else:
                words = estimate_words(seg.text, float(seg.start), float(seg.end))
            if words:
                out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip(), "words": words})
        return out

    def close(self) -> None:
        del self.model
        gc.collect()


def estimate_words(text: str, start: float, end: float) -> list[dict]:
    """Spread a segment's words over its duration by character length. Marked est=True."""
    tokens = [t for t in text.split() if t.strip()]
    if not tokens or end <= start:
        return []
    total = sum(len(t) + 1 for t in tokens)
    pos, out = start, []
    for t in tokens:
        dur = (end - start) * (len(t) + 1) / total
        out.append({"word": t, "start": pos, "end": pos + dur, "prob": 0.0, "est": True})
        pos += dur
    return out


def prompt_for(mode: str, cue_texts: list[str]) -> str | None:
    if mode == "none":
        return None
    if mode == "subtitles" and cue_texts:
        return " ".join(t.replace("\n", " ") for t in cue_texts)[-800:]
    return GENERIC_PROMPT
