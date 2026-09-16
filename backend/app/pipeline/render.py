"""Streaming audio render (sample-accurate gain envelope) and final remux."""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import media

EAC3_RATES = (48000, 44100, 32000)


@dataclass
class RenderConfig:
    fade: float = 0.008
    front_duck: float = 0.25
    bleep_freq: float = 1000.0
    bleep_level: float = 0.2
    bitrate_surround: str = "640k"
    bitrate_stereo: str = "224k"

    @classmethod
    def from_settings(cls, s) -> "RenderConfig":
        return cls(s.fade, s.front_duck, s.bleep_freq, s.bleep_level, s.bitrate_surround, s.bitrate_stereo)


def apply_edits(x: np.ndarray, pos: int, rate: int, edits: list[dict], layout: str, cfg: RenderConfig) -> None:
    """In-place: x is (frames, channels) float32 starting at absolute sample `pos`.

    Each span is fully silent between start and end; linear fades sit just outside it.
    5.1: center muted, front L/R ducked to `front_duck`, LFE and surrounds untouched.
    """
    n = x.shape[0]
    if n == 0 or not edits:
        return
    f = max(1.0, cfg.fade * rate)
    t0, t1 = pos, pos + n
    active = [e for e in edits if e["start"] * rate - f < t1 and e["end"] * rate + f > t0]
    if not active:
        return
    t = np.arange(t0, t1, dtype=np.float64)
    gain = np.ones(n, dtype=np.float64)
    bleep = np.zeros(n, dtype=np.float64)
    for e in active:
        s, en = e["start"] * rate, e["end"] * rate
        c = np.clip(np.maximum(s - t, t - en) / f, 0.0, 1.0)
        np.minimum(gain, c, out=gain)
        if e.get("kind") == "bleep":
            np.maximum(bleep, 1.0 - c, out=bleep)
    g = gain.astype(np.float32)
    if layout == "5.1" and x.shape[1] >= 6:
        duck = (cfg.front_duck + (1.0 - cfg.front_duck) * g).astype(np.float32)
        x[:, 0] *= duck
        x[:, 1] *= duck
        x[:, 2] *= g
    else:
        x *= g[:, None]
    if bleep.any():
        tone = (cfg.bleep_level * np.sin(2 * np.pi * cfg.bleep_freq * t / rate) * bleep).astype(np.float32)
        if layout == "5.1" and x.shape[1] >= 6:
            x[:, 2] += tone
        else:
            x += tone[:, None]


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
) -> None:
    layout = media.output_layout(src_channels)
    ch = media.LAYOUT_CHANNELS[layout]
    rate = render_rate(src_rate)
    bitrate = cfg.bitrate_surround if ch >= 6 else cfg.bitrate_stereo
    edits = sorted(edits, key=lambda e: e["start"])
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
        chunk_frames = rate  # 1 second
        pos = 0
        try:
            while True:
                if should_cancel and should_cancel():
                    raise media.Canceled()
                raw = _read_exact(dec.stdout, chunk_frames * frame_bytes)
                usable = len(raw) - len(raw) % frame_bytes
                if usable <= 0:
                    break
                x = np.frombuffer(raw[:usable], dtype="<f4").reshape(-1, ch).copy()
                apply_edits(x, pos, rate, edits, layout, cfg)
                np.clip(x, -1.0, 1.0, out=x)
                enc.stdin.write(x.tobytes())
                pos += x.shape[0]
                if on_progress and duration:
                    on_progress(min(1.0, pos / rate / duration))
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
        if pos == 0:
            raise media.MediaError("audio decode produced no samples")


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
    """Short stereo float32 clip around a hit, with the same envelope the render uses."""
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
        apply_edits(x, int(round(max(0.0, start) * rate)), rate, edits, layout, cfg)
    if ch == 6:
        L = x[:, 0] + 0.707 * x[:, 2] + 0.707 * x[:, 4]
        R = x[:, 1] + 0.707 * x[:, 2] + 0.707 * x[:, 5]
        stereo = np.stack([L, R], axis=1)
    elif ch == 1:
        stereo = np.repeat(x, 2, axis=1)
    else:
        stereo = x
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 1.0:
        stereo /= peak
    return stereo, rate


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
