"""ffprobe/ffmpeg helpers: probing, stream selection, dialogue extraction and WAV access."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
IMAGE_SUB_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}
ENGLISH = {"en", "eng", "english"}

ANALYSIS_RATE = 16000


class Canceled(Exception):
    pass


class MediaError(RuntimeError):
    pass


def probe(path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", "-show_chapters", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise MediaError(f"ffprobe failed: {e.stderr.strip()[:500]}") from e
    return json.loads(out)


def duration_of(info: dict) -> float:
    try:
        return float(info.get("format", {}).get("duration") or 0.0)
    except ValueError:
        return 0.0


def _lang(stream: dict) -> str:
    return (stream.get("tags", {}).get("language") or "und").lower()


def _title(stream: dict) -> str:
    return stream.get("tags", {}).get("title") or ""


def audio_streams(info: dict) -> list[dict]:
    out = []
    for s in info.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        out.append(
            {
                "index": s["index"],
                "codec": s.get("codec_name"),
                "profile": s.get("profile"),
                "channels": int(s.get("channels") or 0),
                "layout": s.get("channel_layout"),
                "sampleRate": int(s.get("sample_rate") or 48000),
                "language": _lang(s),
                "title": _title(s),
                "default": bool(s.get("disposition", {}).get("default")),
                "startTime": float(s.get("start_time") or 0.0),
            }
        )
    return out


def choose_audio(info: dict) -> dict:
    streams = audio_streams(info)
    if not streams:
        raise MediaError("No audio stream found")

    def score(s: dict) -> tuple:
        title = s["title"].lower()
        commentary = "comment" in title or "description" in title
        lang_ok = s["language"] in ENGLISH or s["language"] == "und"
        return (not commentary, lang_ok, s["default"], s["channels"], -s["index"])

    return max(streams, key=score)


def output_layout(channels: int) -> str:
    """Everything we render is normalized to mono, stereo or 5.1 (FL FR FC LFE BL BR)."""
    if channels <= 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return "5.1"


LAYOUT_CHANNELS = {"mono": 1, "stereo": 2, "5.1": 6}


def subtitle_streams(info: dict) -> list[dict]:
    out = []
    for s in info.get("streams", []):
        if s.get("codec_type") != "subtitle":
            continue
        disp = s.get("disposition", {})
        codec = s.get("codec_name") or ""
        out.append(
            {
                "index": s["index"],
                "codec": codec,
                "text": codec in TEXT_SUB_CODECS,
                "image": codec in IMAGE_SUB_CODECS,
                "language": _lang(s),
                "title": _title(s),
                "forced": bool(disp.get("forced")) or "forced" in _title(s).lower(),
                "sdh": bool(disp.get("hearing_impaired")) or any(k in _title(s).lower() for k in ("sdh", "cc", "hearing")),
            }
        )
    return out


def run_ffmpeg(
    args: list[str],
    duration: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """Run ffmpeg with -progress parsing and cooperative cancellation."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-progress", "pipe:1", *args]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if should_cancel and should_cancel():
                    proc.kill()
                    raise Canceled()
                if on_progress and duration and line.startswith("out_time_us="):
                    try:
                        us = int(line.split("=", 1)[1])
                        on_progress(max(0.0, min(1.0, us / 1e6 / duration)))
                    except ValueError:
                        pass
            rc = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
        if rc != 0:
            err.seek(0)
            raise MediaError(f"ffmpeg failed ({rc}): {err.read().decode(errors='replace')[-1500:]}")


def dialog_filter(channels: int) -> str:
    """Resample to container time 0 (pads leading silence), take the center channel if there is one."""
    chain = ["aresample=async=1:first_pts=0"]
    if channels >= 3:
        chain += ["aformat=channel_layouts=5.1", "pan=mono|c0=FC"]
    elif channels == 2:
        chain += ["pan=mono|c0=0.5*c0+0.5*c1"]
    chain += [f"aresample={ANALYSIS_RATE}"]
    return ",".join(chain)


def extract_dialog_wav(
    src: str,
    stream_index: int,
    channels: int,
    out_wav: str,
    duration: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    args = [
        "-i", src,
        "-map", f"0:{stream_index}",
        "-vn", "-sn", "-dn",
        "-af", dialog_filter(channels),
        "-ac", "1", "-ar", str(ANALYSIS_RATE), "-c:a", "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(args, duration, on_progress, should_cancel)


class WavReader:
    """Memory-mapped 16-bit PCM WAV so a 2-hour track never sits in RAM."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            header = f.read(12)
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise MediaError("Not a WAV file")
            channels = rate = bits = None
            data_offset = data_size = None
            while True:
                chunk = f.read(8)
                if len(chunk) < 8:
                    break
                cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
                if cid == b"fmt ":
                    fmt = f.read(size)
                    channels, rate = struct.unpack("<HI", fmt[2:8])
                    bits = struct.unpack("<H", fmt[14:16])[0]
                elif cid == b"data":
                    data_offset = f.tell()
                    data_size = size
                    break
                else:
                    f.seek(size + (size & 1), os.SEEK_CUR)
        if data_offset is None or bits != 16:
            raise MediaError("Unsupported WAV (need 16-bit PCM)")
        file_size = os.path.getsize(path)
        # ffmpeg writes a placeholder size when streaming; trust the file length.
        if not data_size or data_offset + data_size > file_size or data_size == 0xFFFFFFFF:
            data_size = file_size - data_offset
        self.channels = channels or 1
        self.rate = rate or ANALYSIS_RATE
        n = data_size // (2 * self.channels)
        self._mm = np.memmap(path, dtype="<i2", mode="r", offset=data_offset, shape=(n * self.channels,))
        self.frames = n
        self._lock = threading.Lock()

    @property
    def duration(self) -> float:
        return self.frames / self.rate

    def read(self, start: float, end: float) -> np.ndarray:
        """Float32 mono samples for [start, end) seconds, zero-padded outside the file."""
        s = int(round(start * self.rate))
        e = int(round(end * self.rate))
        out = np.zeros(max(0, e - s), dtype=np.float32)
        a, b = max(0, s), min(self.frames, e)
        if b > a:
            chunk = np.asarray(self._mm[a * self.channels : b * self.channels], dtype=np.float32)
            if self.channels > 1:
                chunk = chunk.reshape(-1, self.channels).mean(axis=1)
            out[a - s : a - s + (b - a)] = chunk / 32768.0
        return out
