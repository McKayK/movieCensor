"""Naming and placing the finished file in the Censored library."""
from __future__ import annotations

import logging
import os
import shutil

from ..paths import container_to_host, safe_name
from ..plex import guid_tag

log = logging.getLogger("publish")

EDITION = "{edition-Censored}"


def output_names(title: str, year: int | None, guids: list[str]) -> tuple[str, str]:
    """Folder and file name Plex will match to the right movie and label as the Censored edition."""
    base = safe_name(title)
    if year:
        base += f" ({year})"
    tag = guid_tag(guids)
    if tag:
        base += f" {tag}"
    name = f"{base} {EDITION}"
    return name, name + ".mkv"


def temp_path(censored_dir: str, job_id: int) -> str:
    tmp_dir = os.path.join(censored_dir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    ensure_plexignore(censored_dir)
    # Non-media extension so Plex never picks up a half-written file.
    return os.path.join(tmp_dir, f"job-{job_id}.mkv.partial")


def ensure_plexignore(censored_dir: str) -> None:
    p = os.path.join(censored_dir, ".plexignore")
    if not os.path.exists(p):
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(".tmp/*\n")
        except OSError:
            pass


def check_space(censored_dir: str, needed_bytes: int) -> None:
    os.makedirs(censored_dir, exist_ok=True)
    free = shutil.disk_usage(censored_dir).free
    if free < needed_bytes:
        raise OSError(
            f"Not enough free space in the Censored folder: need {needed_bytes / 1e9:.1f} GB, have {free / 1e9:.1f} GB"
        )


def place(tmp_file: str, censored_dir: str, folder: str, filename: str) -> str:
    dest_dir = os.path.join(censored_dir, folder)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    try:
        os.replace(tmp_file, dest)
    except OSError:
        # Different filesystem (unusual mount setup): fall back to copy + delete.
        shutil.move(tmp_file, dest)
    return dest


def plex_scan(plex_server, dest_dir: str, censored_dir: str, censored_host_path: str) -> str | None:
    """Ask Plex to scan just the new folder. Returns a warning message instead of raising."""
    try:
        section = plex_server.censored_section_id()
        if not section:
            return "Censored library not found in Plex (set PLEX_CENSORED_SECTION_ID or CENSORED_HOST_PATH)."
        host = container_to_host(dest_dir, censored_dir, censored_host_path)
        plex_server.refresh_path(section, host)
        log.info("plex scan requested for %s", host)
        return None
    except Exception as e:  # scanning is best-effort; Plex also scans on its own schedule
        return f"Plex scan request failed: {e}"
