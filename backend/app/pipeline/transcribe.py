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
        """Return Whisper segments with word timestamps, times relative to the start of `audio`."""
        segments, _info = self.model.transcribe(
            audio.astype(np.float32),
            language="en",
            task="transcribe",
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            suppress_blank=True,
        )
        out: list[dict] = []
        for seg in segments:
            words = [
                {"word": w.word.strip(), "start": float(w.start), "end": float(w.end), "prob": float(w.probability)}
                for w in (seg.words or [])
                if w.word.strip()
            ]
            if words:
                out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip(), "words": words})
        return out

    def close(self) -> None:
        del self.model
        gc.collect()


def prompt_for(mode: str, cue_texts: list[str]) -> str | None:
    if mode == "none":
        return None
    if mode == "subtitles" and cue_texts:
        return " ".join(t.replace("\n", " ") for t in cue_texts)[-800:]
    return GENERIC_PROMPT
