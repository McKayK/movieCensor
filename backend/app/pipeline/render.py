"""Streaming audio render (sample-accurate envelopes, optional Demucs echo cleanup) and final remux."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from . import media

log = logging.getLogger("render")

EAC3_RATES = (48000, 44100, 32000)
ECHO_MODES = ("off", "duck", "deep")

# 5.1 channel order after normalization: FL FR FC LFE BL BR
FL, FR, FC, LFE, BL, BR = range(6)
OTHER_51 = (FL, FR, BL, BR)


class VocalRemover(Protocol):
    def remove_vocals(self, stereo: np.ndarray, rate: int) -> np.ndarray: ...


@dataclass
class RenderConfig:
    fade: float = 0.025
    duck_fade: float = 0.06
    echo_mode: str = "off"
    echo_tail: float = 0.35
    front_duck: float = 0.25
    surround_duck: float = 0.35
    center_tail_duck: float = 0.2
    deep_margin: float = 0.3
    deep_word_gain: float = 1.0
    bleep_freq: float = 1000.0
    bleep_level: float = 0.2
    bitrate_surround: str = "640k"
    bitrate_stereo: str = "224k"

    @classmethod
    def from_settings(cls, s, echo_mode: str | None = None) -> "RenderConfig":
        mode = echo_mode or s.echo_mode
        if mode not in ECHO_MODES:
            mode = "duck"
        return cls(
            fade=s.fade, duck_fade=s.duck_fade, echo_mode=mode, echo_tail=s.echo_tail,
            front_duck=s.front_duck, surround_duck=s.surround_duck, center_tail_duck=s.center_tail_duck,
            deep_margin=s.deep_margin, deep_word_gain=s.deep_word_gain, bleep_freq=s.bleep_freq,
            bleep_level=s.bleep_level, bitrate_surround=s.bitrate_surround, bitrate_stereo=s.bitrate_stereo,
        )


# ---- envelopes ------------------------------------------------------------------------------------------
def _outside(t: np.ndarray, s: float, e: float, f: float) -> np.ndarray:
    """0 inside [s, e], 1 once at least f samples outside; raised-cosine (click-free) in between."""
    c = np.clip(np.maximum(s - t, t - e) / f, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * c)


def envelopes(pos: int, n: int, rate: int, edits: list[dict], cfg: RenderConfig):
    """(word_gain, echo_weight, word_weight_slow, bleep) for samples [pos, pos+n), or None if nothing applies.

    word_gain:        1 -> 0 over `fade` before the word, 0 during it, back to 1 after (muted channels)
    echo_weight:      0 -> 1 over `duck_fade` before the word, 1 through word + echo tail, back to 0
    word_weight_slow: like echo_weight but without the tail (fronts in "off" mode)
    bleep:            0/1 weight for the tone
    """
    f_word = max(1.0, cfg.fade * rate)
    f_duck = max(1.0, cfg.duck_fade * rate)
    use_tail = cfg.echo_mode != "off"
    reach = max(f_word, f_duck) + (cfg.echo_tail * rate if use_tail else 0.0)
    t0, t1 = pos, pos + n
    active = [e for e in edits if e["start"] * rate - reach < t1 and e["end"] * rate + reach > t0]
    if not active:
        return None
    t = np.arange(t0, t1, dtype=np.float64)
    word_gain = np.ones(n)
    echo_w = np.zeros(n)
    slow_w = np.zeros(n)
    bleep = np.zeros(n)
    for e in active:
        s, en = e["start"] * rate, e["end"] * rate
        tail = min(e.get("tail", cfg.echo_tail), cfg.echo_tail) * rate if use_tail else 0.0
        g = _outside(t, s, en, f_word)
        np.minimum(word_gain, g, out=word_gain)
        np.maximum(echo_w, 1.0 - _outside(t, s, en + tail, f_duck), out=echo_w)
        np.maximum(slow_w, 1.0 - _outside(t, s, en, f_duck), out=slow_w)
        if e.get("kind") == "bleep":
            np.maximum(bleep, 1.0 - g, out=bleep)
    return word_gain, echo_w, slow_w, bleep


def apply_edits(
    x: np.ndarray,
    pos: int,
    rate: int,
    edits: list[dict],
    layout: str,
    cfg: RenderConfig,
    clean: np.ndarray | None = None,
    clean_mask: np.ndarray | None = None,
) -> None:
    """In-place: x is (frames, channels) float32 starting at absolute sample `pos`.

    `clean` is voice-removed audio aligned with x (deep mode); `clean_mask` marks which samples have it.
    Where deep cleanup isn't available the renderer falls back to ducking.
    """
    n = x.shape[0]
    if n == 0 or not edits:
        return
    env = envelopes(pos, n, rate, edits, cfg)
    if env is None:
        return
    word_gain, echo_w, slow_w, bleep = (a.astype(np.float32) for a in env)
    surround = layout == "5.1" and x.shape[1] >= 6
    mode = cfg.echo_mode

    if mode == "deep" and clean is not None and clean_mask is not None:
        have = clean_mask.astype(np.float32)
        deep_w = echo_w * have        # blend toward the clean version where we have it
        duck_w = echo_w * (1 - have)  # duck where we don't
    else:
        deep_w = np.zeros(n, dtype=np.float32)
        duck_w = echo_w if mode in ("duck", "deep") else np.zeros(n, dtype=np.float32)

    if surround:
        if mode == "off":
            fronts = 1 - slow_w * (1 - cfg.front_duck)
            x[:, FL] *= fronts
            x[:, FR] *= fronts
            x[:, FC] *= word_gain
        else:
            for ch, level in ((FL, cfg.front_duck), (FR, cfg.front_duck), (BL, cfg.surround_duck), (BR, cfg.surround_duck)):
                if clean is not None:
                    x[:, ch] = x[:, ch] * (1 - deep_w) + clean[:, ch] * deep_w
                x[:, ch] *= 1 - duck_w * (1 - level)
            if clean is not None:
                x[:, FC] = x[:, FC] * (1 - deep_w) + clean[:, FC] * deep_w
            x[:, FC] *= 1 - duck_w * (1 - cfg.center_tail_duck)
            # The dialogue channel is always fully silent during the word itself.
            x[:, FC] *= word_gain
    else:
        if mode == "deep" and clean is not None:
            blended = x * (1 - deep_w)[:, None] + clean * deep_w[:, None]
            # Where we have a clean version, the word span keeps the music at deep_word_gain;
            # elsewhere (no clean audio) the word is silenced as usual.
            word_level = word_gain + (1 - word_gain) * cfg.deep_word_gain * have
            x[:] = blended * word_level[:, None]
            x *= (1 - duck_w * (1 - cfg.front_duck))[:, None]
        else:
            x *= word_gain[:, None]
            if mode == "duck":
                x *= (1 - duck_w * (1 - cfg.front_duck))[:, None]

    if bleep.any():
        tone = (cfg.bleep_level * np.sin(2 * np.pi * cfg.bleep_freq * np.arange(pos, pos + n) / rate)).astype(np.float32) * bleep
        if surround:
            x[:, FC] += tone
        else:
            x += tone[:, None]


# ---- deep-clean regions -----------------------------------------------------------------------------------
def deep_regions(edits: list[dict], rate: int, cfg: RenderConfig) -> list[dict]:
    """Sample ranges that need voice-removed audio: each edit plus its echo tail, fades and a margin."""
    pad_before = cfg.duck_fade + cfg.deep_margin
    pad_after = cfg.echo_tail + cfg.duck_fade + cfg.deep_margin
    spans = sorted((max(0.0, e["start"] - pad_before), e["end"] + pad_after) for e in edits)
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [{"s": int(s * rate), "e": int(e * rate), "parts": [], "clean": None, "done": False} for s, e in merged]


def separate_block(audio: np.ndarray, layout: str, remover: VocalRemover, rate: int) -> np.ndarray:
    """Voice-removed copy of a (frames, channels) block, processed as stereo pairs."""
    clean = audio.copy()
    if layout == "5.1" and audio.shape[1] >= 6:
        for a, b in ((FL, FR), (BL, BR)):
            clean[:, [a, b]] = remover.remove_vocals(audio[:, [a, b]], rate)
        c = remover.remove_vocals(np.repeat(audio[:, [FC]], 2, axis=1), rate)
        clean[:, FC] = c.mean(axis=1)
    elif audio.shape[1] == 2:
        clean[:] = remover.remove_vocals(audio, rate)
    else:
        c = remover.remove_vocals(np.repeat(audio, 2, axis=1), rate)
        clean[:, 0] = c.mean(axis=1)
    return clean.astype(np.float32)


class DeepCleaner:
    """Collects region audio from the stream, separates it once complete, and hands out clean samples."""

    def __init__(self, regions: list[dict], layout: str, remover: VocalRemover | None, rate: int,
                 on_region: Callable[[int, int], None] | None = None):
        self.regions = regions
        self.layout = layout
        self.remover = remover
        self.rate = rate
        self.on_region = on_region
        self.failed = 0

    def feed(self, x: np.ndarray, pos: int) -> None:
        end = pos + len(x)
        for r in self.regions:
            if r["done"] or r["e"] <= pos or r["s"] >= end:
                continue
            a, b = max(r["s"], pos), min(r["e"], end)
            r["parts"].append(x[a - pos : b - pos].copy())
            if b >= r["e"]:
                self._finish(r)

    def finish_all(self) -> None:
        for r in self.regions:
            if not r["done"]:
                self._finish(r)

    def _finish(self, r: dict) -> None:
        r["done"] = True
        if not r["parts"] or self.remover is None:
            return
        block = np.concatenate(r["parts"], axis=0)
        r["parts"] = []
        try:
            r["clean"] = separate_block(block, self.layout, self.remover, self.rate)
        except Exception as e:  # fall back to ducking for this region
            self.failed += 1
            log.warning("vocal removal failed for region at %.1fs: %s", r["s"] / self.rate, e)
        if self.on_region:
            self.on_region(sum(1 for q in self.regions if q["done"]), len(self.regions))

    def ready_until(self) -> int:
        """Samples before this position don't depend on any unfinished region."""
        pending = [r["s"] for r in self.regions if not r["done"]]
        return min(pending) if pending else 1 << 62

    def clean_for(self, pos: int, n: int, ch: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        clean = None
        mask = None
        end = pos + n
        for r in self.regions:
            if r["clean"] is None or r["e"] <= pos or r["s"] >= end:
                continue
            if clean is None:
                clean = np.zeros((n, ch), dtype=np.float32)
                mask = np.zeros(n, dtype=bool)
            a, b = max(r["s"], pos), min(r["e"], end)
            src = r["clean"][a - r["s"] : b - r["s"]]
            clean[a - pos : a - pos + len(src)] = src
            mask[a - pos : a - pos + len(src)] = True
        return clean, mask


# ---- ffmpeg plumbing ----------------------------------------------------------------------------------
def _read_exact(stream, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def render_rate(src_rate: int) -> int:
    return src_rate if src_rate in EAC3_RATES else 48000


def decode_args(src: str, stream_index: int, layout: str, rate: int, start: float | None = None,
                length: float | None = None) -> list[str]:
    args = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error"]
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", src]
    if length is not None:
        args += ["-t", f"{length:.3f}"]
    af = f"aformat=sample_fmts=flt:channel_layouts={layout}"
    if start is None:
        af = "aresample=async=1:first_pts=0," + af
    args += ["-map", f"0:{stream_index}", "-vn", "-sn", "-dn", "-af", af, "-ar", str(rate), "-f", "f32le", "pipe:1"]
    return args


def process_stream(
    chunks,
    rate: int,
    ch: int,
    layout: str,
    edits: list[dict],
    cfg: RenderConfig,
    write: Callable[[np.ndarray], None],
    remover: VocalRemover | None = None,
    on_region: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Core render loop, independent of ffmpeg. `chunks` yields (frames, ch) float32 arrays in order.

    Output is delayed only as long as a deep-clean region is still being collected. Returns
    (samples written, regions whose separation failed).
    """
    edits = sorted(edits, key=lambda e: e["start"])
    cleaner = None
    if cfg.echo_mode == "deep" and remover is not None and edits:
        cleaner = DeepCleaner(deep_regions(edits, rate, cfg), layout, remover, rate, on_region)
    queue: deque[tuple[int, np.ndarray]] = deque()
    pos = 0
    written = 0

    def flush(limit: int) -> None:
        nonlocal written
        while queue and queue[0][0] + len(queue[0][1]) <= limit:
            p, blk = queue.popleft()
            clean, mask = cleaner.clean_for(p, len(blk), ch) if cleaner else (None, None)
            apply_edits(blk, p, rate, edits, layout, cfg, clean, mask)
            np.clip(blk, -1.0, 1.0, out=blk)
            write(blk)
            written += len(blk)

    for x in chunks:
        if len(x) == 0:
            continue
        if cleaner:
            cleaner.feed(x, pos)
        queue.append((pos, x))
        pos += len(x)
        flush(cleaner.ready_until() if cleaner else 1 << 62)
    if cleaner:
        cleaner.finish_all()
    flush(1 << 62)
    return written, (cleaner.failed if cleaner else 0)


def render_audio(
    src: str,
    stream_index: int,
    src_channels: int,
    src_rate: int,
    duration: float,
    edits: list[dict],
    out_path: str,
    cfg: RenderConfig,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    remover: VocalRemover | None = None,
    on_region: Callable[[int, int], None] | None = None,
) -> int:
    """Decode -> envelopes (+ Demucs regions) -> E-AC3. Returns the number of regions that fell back to ducking."""
    layout = media.output_layout(src_channels)
    ch = media.LAYOUT_CHANNELS[layout]
    rate = render_rate(src_rate)
    bitrate = cfg.bitrate_surround if ch >= 6 else cfg.bitrate_stereo
    enc_cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
        "-f", "f32le", "-ar", str(rate), "-ac", str(ch), "-i", "pipe:0",
        "-c:a", "eac3", "-b:a", bitrate, "-f", "matroska", out_path,
    ]
    with tempfile.TemporaryFile() as dec_err, tempfile.TemporaryFile() as enc_err:
        dec = subprocess.Popen(decode_args(src, stream_index, layout, rate), stdout=subprocess.PIPE, stderr=dec_err)
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=enc_err)
        assert dec.stdout is not None and enc.stdin is not None
        frame_bytes = 4 * ch
        read_pos = 0

        def chunks():
            nonlocal read_pos
            while True:
                if should_cancel and should_cancel():
                    raise media.Canceled()
                raw = _read_exact(dec.stdout, rate * frame_bytes)
                usable = len(raw) - len(raw) % frame_bytes
                if usable <= 0:
                    return
                x = np.frombuffer(raw[:usable], dtype="<f4").reshape(-1, ch).copy()
                read_pos += len(x)
                if on_progress and duration:
                    on_progress(min(1.0, read_pos / rate / duration))
                yield x

        try:
            written, failed = process_stream(chunks(), rate, ch, layout, edits, cfg,
                                             lambda blk: enc.stdin.write(blk.tobytes()), remover, on_region)
            enc.stdin.close()
            dec_rc, enc_rc = dec.wait(), enc.wait()
        except BaseException:
            for p in (dec, enc):
                if p.poll() is None:
                    p.kill()
            raise
        if dec_rc != 0:
            dec_err.seek(0)
            raise media.MediaError(f"audio decode failed: {dec_err.read().decode(errors='replace')[-1000:]}")
        if enc_rc != 0:
            enc_err.seek(0)
            raise media.MediaError(f"audio encode failed: {enc_err.read().decode(errors='replace')[-1000:]}")
        if written == 0:
            raise media.MediaError("audio decode produced no samples")
        return failed


def mux(
    src: str,
    audio_path: str,
    srt_path: str | None,
    out_path: str,
    title: str,
    duration: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """Video copied untouched (attached cover art skipped), censored audio, optional censored SRT, chapters kept."""
    args = ["-i", src, "-i", audio_path]
    if srt_path:
        args += ["-i", srt_path]
    args += ["-map", "0:V", "-map", "1:a:0"]
    if srt_path:
        args += ["-map", "2:s:0"]
    args += [
        "-map_metadata", "0", "-map_chapters", "0",
        "-c", "copy",
        "-metadata", f"title={title}",
        "-metadata:s:a:0", "language=eng",
        "-metadata:s:a:0", "title=Censored",
        "-disposition:a:0", "default",
    ]
    if srt_path:
        args += ["-c:s", "srt", "-metadata:s:s:0", "language=eng", "-metadata:s:s:0", "title=Censored", "-disposition:s:0", "0"]
    args += ["-max_muxing_queue_size", "9999", "-f", "matroska", out_path]
    media.run_ffmpeg(args, duration, on_progress, should_cancel)


def preview_clip(
    src: str,
    stream_index: int,
    src_channels: int,
    start: float,
    length: float,
    edits: list[dict] | None,
    cfg: RenderConfig,
) -> tuple[np.ndarray, int]:
    """Short stereo float32 clip around a hit, with the same envelopes the render uses.

    Deep-clean mode is previewed as ducking (running Demucs per click would be slow); once a job is done
    the preview reads the finished file instead, so what you hear is exact.
    """
    layout = media.output_layout(src_channels)
    ch = media.LAYOUT_CHANNELS[layout]
    rate = 48000
    proc = subprocess.run(decode_args(src, stream_index, layout, rate, start=max(0.0, start), length=length),
                          capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise media.MediaError(proc.stderr.decode(errors="replace")[-500:])
    raw = proc.stdout[: len(proc.stdout) - len(proc.stdout) % (4 * ch)]
    x = np.frombuffer(raw, dtype="<f4").reshape(-1, ch).copy()
    if edits:
        pcfg = cfg
        if cfg.echo_mode == "deep":
            from dataclasses import replace

            pcfg = replace(cfg, echo_mode="duck")
        apply_edits(x, int(round(max(0.0, start) * rate)), rate, edits, layout, pcfg)
    return to_stereo(x), rate


def to_stereo(x: np.ndarray) -> np.ndarray:
    ch = x.shape[1]
    if ch >= 6:
        L = x[:, FL] + 0.707 * x[:, FC] + 0.707 * x[:, BL]
        R = x[:, FR] + 0.707 * x[:, FC] + 0.707 * x[:, BR]
        stereo = np.stack([L, R], axis=1)
    elif ch == 1:
        stereo = np.repeat(x, 2, axis=1)
    else:
        stereo = x
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 1.0:
        stereo = stereo / peak
    return stereo


def wav_bytes(stereo: np.ndarray, rate: int) -> bytes:
    import io
    import wave

    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return bio.getvalue()
