"""Time transcription + alignment on real hardware.

    docker compose run --rm worker python -m app.tools.benchmark "/media/movies/Heat (1995)/Heat.mkv" --start 600 --seconds 60

Run it with a couple of models (--model large-v3-turbo / distil-large-v3 / large-v3) to pick WHISPER_MODEL.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time

from ..config import settings
from ..pipeline import media


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--start", type=float, default=600.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--model", default=settings.whisper_model)
    ap.add_argument("--threads", type=int, default=settings.cpu_threads)
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--demucs", action="store_true", help="also time Demucs voice removal on 10 s of the movie")
    args = ap.parse_args()

    info = media.probe(args.movie)
    audio = media.choose_audio(info)
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "clip.wav")
        t0 = time.time()
        media.run_ffmpeg([
            "-ss", str(args.start), "-t", str(args.seconds), "-i", args.movie, "-map", f"0:{audio['index']}",
            "-af", media.dialog_filter(audio["channels"]).replace("aresample=async=1:first_pts=0,", ""),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav,
        ])
        reader = media.WavReader(wav)
        clip = reader.read(0, reader.duration)
        print(f"extract: {time.time() - t0:.1f}s for {reader.duration:.0f}s of audio ({audio['channels']} ch {audio['codec']})")

        from ..pipeline.transcribe import GENERIC_PROMPT, Transcriber

        t0 = time.time()
        tr = Transcriber(args.model, settings.whisper_device, settings.whisper_compute_type, args.threads,
                         settings.whisper_beam_size, settings.models_dir)
        load = time.time() - t0
        t0 = time.time()
        segs = tr.transcribe(clip, GENERIC_PROMPT)
        took = time.time() - t0
        tr.close()
        rtf = reader.duration / took
        print(f"whisper {args.model}: load {load:.1f}s, transcribe {took:.1f}s -> {rtf:.2f}x realtime")
        for s in segs:
            print(f"  [{s['start']:6.2f}-{s['end']:6.2f}] {s['text']}")

        if not args.no_align and segs:
            from ..pipeline.align import Aligner

            t0 = time.time()
            al = Aligner(settings.align_model, args.threads, settings.models_dir)
            load = time.time() - t0
            t0 = time.time()
            words = al.align(clip, segs)
            took = time.time() - t0
            al.close()
            print(f"align: load {load:.1f}s, align {took:.1f}s")
            for w in words[:40]:
                ws, we = w.get("whisper_start", 0), w.get("whisper_end", 0)
                print(f"  {w['word']:<14} whisper {ws:6.2f}-{we:6.2f}  aligned {w['start']:6.2f}-{w['end']:6.2f}"
                      f"  shift {w['start'] - ws:+.3f}s")

        if args.demucs:
            import numpy as np

            from ..pipeline.render import decode_args, separate_block
            from ..pipeline.separate import DemucsRemover
            import subprocess

            layout = media.output_layout(audio["channels"])
            ch = media.LAYOUT_CHANNELS[layout]
            raw = subprocess.run(decode_args(args.movie, audio["index"], layout, 48000, start=args.start, length=10.0),
                                 capture_output=True, check=True).stdout
            block = np.frombuffer(raw[: len(raw) - len(raw) % (4 * ch)], dtype="<f4").reshape(-1, ch).copy()
            t0 = time.time()
            remover = DemucsRemover(settings.demucs_model, args.threads, settings.models_dir)
            load = time.time() - t0
            t0 = time.time()
            separate_block(block, layout, remover, 48000)
            took = time.time() - t0
            remover.close()
            per_hit = took / 10.0 * 1.5  # a typical cleaned region is ~1.5 s
            print(f"demucs {settings.demucs_model}: load {load:.1f}s, 10 s of {layout} took {took:.1f}s "
                  f"-> about {per_hit:.1f}s per censored word, ~{per_hit * 40 / 60:.0f} min for 40 words")

        est = 12 * 60  # ~12 minutes of windows for a typical R-rated movie
        print(f"\nEstimated analysis time for ~{est // 60} min of windows: ~{est / rtf / 60:.0f} min "
              "(transcription only; alignment adds a little)")


if __name__ == "__main__":
    main()
