"""Demucs vocal removal for short regions around each mute (the "deep clean" echo mode)."""
from __future__ import annotations

import gc
import logging
import os

import numpy as np

log = logging.getLogger("separate")


class DemucsRemover:
    """Loads a Demucs model once per render job and returns the music-and-effects part of stereo clips."""

    def __init__(self, model_name: str = "htdemucs", threads: int = 4, cache_dir: str | None = None, model=None):
        import torch

        if cache_dir:
            os.environ.setdefault("TORCH_HOME", os.path.join(cache_dir, "torch"))
        torch.set_num_threads(max(1, threads))
        self._torch = torch
        if model is None:
            from demucs.pretrained import get_model

            log.info("loading Demucs model %s", model_name)
            model = get_model(model_name)
        model.eval()
        self.model = model
        self.samplerate = int(getattr(model, "samplerate", 44100))
        self.vocals = list(model.sources).index("vocals")

    def remove_vocals(self, stereo: np.ndarray, rate: int) -> np.ndarray:
        """(frames, 2) float32 at `rate` -> same shape with the vocals subtracted."""
        import julius
        from demucs.apply import apply_model

        torch = self._torch
        n = stereo.shape[0]
        if n == 0:
            return stereo.copy()
        wav = torch.from_numpy(np.ascontiguousarray(stereo.T, dtype=np.float32))
        if rate != self.samplerate:
            wav = julius.resample_frac(wav, rate, self.samplerate)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std()
        if not torch.isfinite(std) or std < 1e-6:
            return stereo.copy()  # silence: nothing to remove
        with torch.inference_mode():
            sources = apply_model(self.model, ((wav - mean) / std)[None], shifts=0, split=True, overlap=0.25,
                                  progress=False, device="cpu", num_workers=0)[0]
        vocals = sources[self.vocals] * std
        clean = wav - vocals
        if rate != self.samplerate:
            clean = julius.resample_frac(clean, self.samplerate, rate)
        out = clean.T.numpy()
        if out.shape[0] < n:
            out = np.pad(out, ((0, n - out.shape[0]), (0, 0)))
        return out[:n].astype(np.float32)

    def close(self) -> None:
        del self.model
        gc.collect()
