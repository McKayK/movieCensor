"""Runtime configuration, read once from environment variables (.env via docker compose)."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path_maps(raw: str) -> list[tuple[str, str]]:
    """PATH_MAPS="D:\\Media\\Movies=/media/movies;E:\\More=/media/more" -> [(plex_prefix, container_prefix)]."""
    maps: list[tuple[str, str]] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        # Split on the LAST '=' so Windows paths never break it.
        plex_prefix, container_prefix = part.rsplit("=", 1)
        maps.append((plex_prefix.strip(), container_prefix.strip()))
    return maps


@dataclass
class Settings:
    # --- Plex -------------------------------------------------------------
    plex_url: str = field(default_factory=lambda: _str("PLEX_URL", "http://host.docker.internal:32400").rstrip("/"))
    plex_token: str = field(default_factory=lambda: _str("PLEX_TOKEN"))
    movie_section_ids: list[str] = field(
        default_factory=lambda: [s.strip() for s in _str("PLEX_MOVIE_SECTION_IDS").split(",") if s.strip()]
    )
    censored_section_id: str = field(default_factory=lambda: _str("PLEX_CENSORED_SECTION_ID"))

    # --- Paths ------------------------------------------------------------
    path_maps: list[tuple[str, str]] = field(default_factory=lambda: _path_maps(_str("PATH_MAPS")))
    censored_dir: str = field(default_factory=lambda: _str("CENSORED_DIR", "/media/censored"))
    censored_host_path: str = field(default_factory=lambda: _str("CENSORED_HOST_PATH"))
    data_dir: str = field(default_factory=lambda: _str("DATA_DIR", "/data"))
    static_dir: str = field(default_factory=lambda: _str("STATIC_DIR", str(Path(__file__).resolve().parent.parent / "static")))

    # --- Web / auth -------------------------------------------------------
    session_secret: str = field(default_factory=lambda: _str("SESSION_SECRET"))
    public_url: str = field(default_factory=lambda: _str("PUBLIC_URL").rstrip("/"))
    admin_usernames: list[str] = field(
        default_factory=lambda: [s.strip().lower() for s in _str("ADMIN_USERNAMES").split(",") if s.strip()]
    )
    auth_disabled: bool = field(default_factory=lambda: _bool("AUTH_DISABLED", False))
    cookie_secure: bool = field(default_factory=lambda: _bool("COOKIE_SECURE", False))
    max_active_jobs_per_user: int = field(default_factory=lambda: _int("MAX_ACTIVE_JOBS_PER_USER", 3))

    # --- Speech -----------------------------------------------------------
    whisper_model: str = field(default_factory=lambda: _str("WHISPER_MODEL", "large-v3-turbo"))
    whisper_compute_type: str = field(default_factory=lambda: _str("WHISPER_COMPUTE_TYPE", "int8"))
    whisper_device: str = field(default_factory=lambda: _str("WHISPER_DEVICE", "cpu"))
    whisper_beam_size: int = field(default_factory=lambda: _int("WHISPER_BEAM_SIZE", 5))
    whisper_prompt_mode: str = field(default_factory=lambda: _str("WHISPER_PROMPT_MODE", "generic"))  # generic|subtitles|none
    align_model: str = field(default_factory=lambda: _str("ALIGN_MODEL", "facebook/wav2vec2-base-960h"))
    cpu_threads: int = field(default_factory=lambda: _int("CPU_THREADS", 4))

    # --- Windows & matching ----------------------------------------------
    window_pad: float = field(default_factory=lambda: _float("WINDOW_PAD", 6.0))
    window_merge_gap: float = field(default_factory=lambda: _float("WINDOW_MERGE_GAP", 2.0))
    window_max: float = field(default_factory=lambda: _float("WINDOW_MAX", 120.0))
    match_tolerance: float = field(default_factory=lambda: _float("MATCH_TOLERANCE", 2.0))
    review_always: bool = field(default_factory=lambda: _bool("REVIEW_ALWAYS", False))

    # --- Boundaries (seconds) --------------------------------------------
    # Swears must never leak, so padding before a word wins over keeping the previous word intact.
    pre_pad: float = field(default_factory=lambda: _float("PRE_PAD", 0.08))
    post_pad: float = field(default_factory=lambda: _float("POST_PAD", 0.05))
    min_pre_pad: float = field(default_factory=lambda: _float("MIN_PRE_PAD", 0.06))
    min_post_pad: float = field(default_factory=lambda: _float("MIN_POST_PAD", 0.04))
    snap: float = field(default_factory=lambda: _float("SNAP", 0.03))
    onset_detect: bool = field(default_factory=lambda: _bool("ONSET_DETECT", True))
    onset_max: float = field(default_factory=lambda: _float("ONSET_MAX", 0.15))
    # Hard limit on how far before the aligned word a mute may start (nudges can still go further)
    max_lead: float = field(default_factory=lambda: _float("MAX_LEAD", 0.18))
    merge_gap: float = field(default_factory=lambda: _float("MERGE_GAP", 0.15))
    line_pad: float = field(default_factory=lambda: _float("LINE_PAD", 0.15))
    max_nudge: float = field(default_factory=lambda: _float("MAX_NUDGE", 0.5))

    # --- Render -----------------------------------------------------------
    fade: float = field(default_factory=lambda: _float("FADE", 0.025))            # center / muted channels
    duck_fade: float = field(default_factory=lambda: _float("DUCK_FADE", 0.06))   # speakers that are ducked or cleaned
    # off = mute center (5.1) or everything (stereo); duck = also turn down other speakers through the echo tail;
    # deep = Demucs removes the voice from every speaker through the echo tail, music keeps playing
    echo_mode: str = field(default_factory=lambda: _str("ECHO_MODE", "deep"))
    echo_tail: float = field(default_factory=lambda: _float("ECHO_TAIL", 0.35))
    front_duck: float = field(default_factory=lambda: _float("FRONT_DUCK", 0.25))
    surround_duck: float = field(default_factory=lambda: _float("SURROUND_DUCK", 0.35))
    center_tail_duck: float = field(default_factory=lambda: _float("CENTER_TAIL_DUCK", 0.2))
    demucs_model: str = field(default_factory=lambda: _str("DEMUCS_MODEL", "htdemucs"))
    deep_margin: float = field(default_factory=lambda: _float("DEEP_MARGIN", 0.3))
    # stereo/mono deep clean: level of the voice-removed audio during the word itself (0 = silence, 1 = music at full)
    deep_word_gain: float = field(default_factory=lambda: _float("DEEP_WORD_GAIN", 1.0))
    bleep_freq: float = field(default_factory=lambda: _float("BLEEP_FREQ", 1000.0))
    bleep_level: float = field(default_factory=lambda: _float("BLEEP_LEVEL", 0.2))
    bitrate_surround: str = field(default_factory=lambda: _str("AUDIO_BITRATE_SURROUND", "640k"))
    bitrate_stereo: str = field(default_factory=lambda: _str("AUDIO_BITRATE_STEREO", "224k"))
    keep_work_files: bool = field(default_factory=lambda: _bool("KEEP_WORK_FILES", False))
    worker_poll_seconds: float = field(default_factory=lambda: _float("WORKER_POLL_SECONDS", 2.0))

    # ---------------------------------------------------------------------
    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "movie-censor.db")

    @property
    def work_dir(self) -> str:
        return os.path.join(self.data_dir, "work")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.data_dir, "models")

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.work_dir, self.models_dir):
            os.makedirs(d, exist_ok=True)

    def get_session_secret(self) -> str:
        if self.session_secret:
            return self.session_secret
        # Persist a generated secret so sessions survive restarts.
        self.ensure_dirs()
        p = Path(self.data_dir) / "session_secret"
        if not p.exists():
            p.write_text(uuid.uuid4().hex + uuid.uuid4().hex)
        return p.read_text().strip()

    def get_client_id(self) -> str:
        """Stable X-Plex-Client-Identifier for this install."""
        self.ensure_dirs()
        p = Path(self.data_dir) / "plex_client_id"
        if not p.exists():
            p.write_text(f"movie-censor-{uuid.uuid4()}")
        return p.read_text().strip()


settings = Settings()
