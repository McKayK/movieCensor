# Movie Censor

Self-hosted, VidAngel-style profanity censoring for a Plex library. Family members sign in with Plex, pick a movie, see which swear words it has, choose what to remove, and get a muted copy in a separate **Censored** Plex library.

- **Subtitle-first:** subtitles find the swears instantly. The CPU only transcribes short windows around them (roughly 10–15 minutes of audio per movie instead of 2 hours).
- **No partial words:** Whisper finds the words, wav2vec2 forced alignment gets exact word boundaries, then padding weighted toward the start of the word, snapping to quiet spots, and 8 ms fades.
- **Surround-aware:** in 5.1/7.1 movies only the dialogue (center) channel is muted, so music and effects keep playing. Front L/R are ducked to catch bleed.
- **Review:** anything the subtitles flag but the audio didn't confirm waits for a person, with before/after audio previews.
- **Original files are never touched.** The video is copied bit-for-bit, and the audio is re-encoded to E-AC3.

The design doc lives in the Claude project. `Architecture` below is the short version.

---

## 1. One-time Plex setup

1. **Create the censored folder**, e.g. `D:\Media\Censored`.
2. **Add a Plex library** → *Movies* → name it `Censored Movies` → add the folder `D:\Media\Censored`.
3. **Share access.** For family members who should only see censored movies, share just that library with them (Plex → Settings → Manage Library Access).
4. **Get your Plex token.** Follow [Plex's guide](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/) and put it in `PLEX_TOKEN`.
5. **Subtitles.** Many remuxes only carry image-based (PGS) subtitles, which can't be scanned yet. If you run **Bazarr**, have it download English `.srt` files next to your movies. SDH/HI subtitles are preferred because they're the most literal.

Output files are named `Title (Year) {imdb-tt…} {edition-Censored}.mkv` inside their own folder. The IMDb tag makes Plex match the right movie. The edition tag shows "Censored" under the title (Plex's edition display may need Plex Pass; the separate library works either way).

## 2. Deploy on KleinmanServer

Requirements: Docker Desktop (WSL2 backend) and Git.

```powershell
git clone https://github.com/<you>/movieCensor.git
cd movieCensor
copy .env.example .env
notepad .env        # PLEX_TOKEN, MOVIES_HOST_DIR, CENSORED_HOST_DIR, PATH_MAPS, CENSORED_HOST_PATH
docker compose up -d --build
```

Open `http://kleinmanserver:8787` and sign in with Plex. The first censor job downloads the speech models (~1.7 GB) into the `censor-data` volume.

**Updating:**

```powershell
git pull
docker compose up -d --build
```

### Give WSL2 enough memory

Docker Desktop only gets about half your RAM by default (~6 GB on a 12 GB machine), and the worker is capped at 6 GB. Create `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=8GB
processors=6
swap=4GB
```

Then run `wsl --shutdown` and restart Docker Desktop.

### Nginx

Proxy a hostname to `http://127.0.0.1:8787`. Then set `PUBLIC_URL=https://censor.yourdomain.com` and `COOKIE_SECURE=true` in `.env`, and run `docker compose up -d`.

```nginx
location / {
    proxy_pass http://127.0.0.1:8787;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_read_timeout 300s;
}
```

### Path mapping (the most common setup mistake)

Plex reports Windows paths like `D:\Media\Movies\Heat (1995)\Heat.mkv`. The containers see `/media/movies/Heat (1995)/Heat.mkv`. `PATH_MAPS` connects the two:

```
PATH_MAPS='D:\Media\Movies=/media/movies'
```

If movies live on several drives, add a volume line per root in `docker-compose.yml` and a matching `;`-separated `PATH_MAPS` entry. The movie page shows a red warning when a file isn't reachable.

## 3. Pick the Whisper model (do this once)

```powershell
docker compose run --rm worker python -m app.tools.benchmark "/media/movies/Heat (1995)/Heat.mkv" --start 1200 --seconds 60
docker compose run --rm worker python -m app.tools.benchmark "/media/movies/Heat (1995)/Heat.mkv" --start 1200 --seconds 60 --model distil-large-v3
```

The benchmark prints realtime speed, the transcript, and how far alignment moved each word from Whisper's guess. Put the winner in `WHISPER_MODEL` and restart the worker.

## 4. Using it

1. **Library** → click a movie → **Scan for language**. You get counts per group and every line, masked by default.
2. Pick a preset (**Essential / Standard / Strict**) or individual groups, add extra words if needed, and choose Mute or Bleep.
3. **Create censored copy.** The worker extracts dialogue, transcribes and aligns the windows, and matches words.
4. If anything is **unconfirmed**, the job waits for review. Play the original and censored previews, then choose *Mute line* or *Leave in*. **Approve & render**.
5. The file lands in the Censored library, and Plex is asked to scan that folder.

Hit statuses:

| Status | Meaning | Default |
| --- | --- | --- |
| Confirmed | In the subtitles and heard in the audio, with exact word boundaries | Mute word |
| Unconfirmed | In the subtitles but not heard (subtitle mismatch, mumbling, loud scene) | Needs a decision |
| Heard in audio | Heard in a window but missing from the subtitles | Mute word |

## 5. Local development (laptop)

```bash
# backend
cd backend
python -m venv .venv && .venv\Scripts\activate    # or: source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements-dev.txt
pytest                                           # unit + synthetic end-to-end tests (needs ffmpeg on PATH)

# run against your real Plex (point PATH_MAPS at wherever the laptop sees the movies)
set DATA_DIR=./data & set CENSORED_DIR=./censored-output & set PLEX_URL=http://kleinmanserver:32400 & set PLEX_TOKEN=... & set AUTH_DISABLED=true
uvicorn app.main:app --reload --port 8000
python -m app.worker                             # second terminal

# frontend
cd frontend
npm install
npm run dev                                      # http://localhost:5173 (proxies /api to :8000)
```

**Censor a single file without Plex or the UI.** This is the fastest way to tune padding on real movies:

```bash
python -m app.tools.censor_file "Heat (1995).mkv" --preset standard --out ./censored-output
```

On the laptop's RTX 3060 you can set `WHISPER_DEVICE=cuda` and `WHISPER_COMPUTE_TYPE=float16` for fast experiments. The server stays on CPU.

## 6. Tuning partial-word leaks

All values are in `.env` and in seconds.

| Setting | Default | Effect |
| --- | --- | --- |
| `PRE_PAD` | 0.08 | Silence before the aligned word start. Raise if you still hear the first consonant. |
| `POST_PAD` | 0.04 | Silence after the word end. |
| `MIN_PAD` | 0.03 | Padding that always applies, even when words touch. |
| `SNAP` | 0.05 | How far an edge may move outward to find a quiet spot. |
| `MERGE_GAP` | 0.15 | Mutes closer than this merge into one. |
| `FADE` | 0.008 | Fade length (outside the muted span, so it never lets sound through). |
| `FRONT_DUCK` | 0.25 | 5.1 front L/R level during a mute (1.0 = leave music untouched). |

Transcriptions are cached per file and window, so re-running a job after tuning skips Whisper.

## Architecture

```
browser ── nginx ── web (FastAPI + React build) ─┐
                                                  ├── SQLite (/data volume)
                   worker (Whisper, wav2vec2, ffmpeg) ─┘
        both read /media/movies (ro); the worker writes /media/censored; both talk to Plex
```

```
backend/app
  main.py            FastAPI app, sessions, serves the SPA
  auth.py            Sign in with Plex (PIN flow); only users with access to this server
  plex.py            Plex server + plex.tv client
  jobs.py            analyze() and render_job() orchestration
  worker.py          single-job queue runner, restart recovery, cancel
  api/               movies/scans and jobs endpoints
  pipeline/
    profanity.py     word groups, presets, matcher, allowlist
    subtitles.py     find, extract, parse, censor subtitles
    media.py         ffprobe, stream choice, dialogue WAV, mmap reader
    windows.py       padded, merged transcription windows
    transcribe.py    faster-whisper
    align.py         wav2vec2 + CTC Viterbi forced alignment
    boundaries.py    padding, snapping, merging
    matching.py      audio matches, subtitle reconciliation, edit list
    render.py        streaming gain-envelope render, mux, previews
    publish.py       naming, atomic placement, Plex scan
  tools/             benchmark.py, censor_file.py
frontend/src         React + Tailwind UI
```

What happens to the output:

- **Audio:** one track, E-AC3 (640k for 5.1, 224k for stereo). 7.1 is folded to 5.1. Atmos is dropped.
- **Other tracks:** commentary and other audio tracks are dropped.
- **Subtitles:** text subtitles are censored with asterisks. Image subtitles are dropped because the swears would show on screen.
- **Chapters:** kept.

## Troubleshooting

- **"Only image-based (PGS) subtitles found":** add an `.srt` next to the movie (Bazarr), then rescan.
- **Movie file not reachable:** fix `PATH_MAPS` or the volume lines.
- **Sign-in says no access:** that Plex account isn't shared on this server.
- **Censored movie doesn't appear:** check that `CENSORED_HOST_PATH` exactly matches the folder in the Plex library (or set `PLEX_CENSORED_SECTION_ID`), then scan the library manually once.
- **Worker killed / out of memory:** raise the `.wslconfig` memory, or use `WHISPER_MODEL=distil-large-v3`.
- **Playback stutters while a job runs:** lower `WORKER_CPUS` and `CPU_THREADS`.
- **Logs:** `docker compose logs -f worker`
