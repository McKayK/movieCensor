"""Translate between the paths Plex reports (Windows host) and the paths inside the containers."""
from __future__ import annotations

import os
import re

from .config import settings


def _norm(p: str) -> str:
    return p.replace("\\", "/").rstrip("/")


def plex_to_container(plex_path: str, maps: list[tuple[str, str]] | None = None) -> str:
    """D:\\Media\\Movies\\X (2010)\\X.mkv -> /media/movies/X (2010)/X.mkv using PATH_MAPS.

    Matching is case-insensitive (Windows) and slash-agnostic. If no map matches the path is returned
    with forward slashes, which works when Plex itself runs in a Linux container with the same mounts.
    """
    maps = settings.path_maps if maps is None else maps
    src = _norm(plex_path)
    # Longest prefix wins so nested mappings behave.
    for plex_prefix, container_prefix in sorted(maps, key=lambda m: len(m[0]), reverse=True):
        pre = _norm(plex_prefix)
        if src.lower() == pre.lower() or src.lower().startswith(pre.lower() + "/"):
            rest = src[len(pre):].lstrip("/")
            return f"{container_prefix.rstrip('/')}/{rest}" if rest else container_prefix.rstrip("/")
    return src


def container_to_host(container_path: str, container_root: str | None = None, host_root: str | None = None) -> str:
    """/media/censored/Folder -> D:\\Media\\Censored\\Folder (for Plex partial scans)."""
    container_root = _norm(container_root or settings.censored_dir)
    host_root = host_root if host_root is not None else settings.censored_host_path
    if not host_root:
        return container_path
    rel = _norm(container_path)
    if rel.lower().startswith(container_root.lower()):
        rel = rel[len(container_root):].lstrip("/")
    windows = bool(re.match(r"^[A-Za-z]:", host_root)) or "\\" in host_root
    sep = "\\" if windows else "/"
    base = host_root.rstrip("\\/")
    return base + (sep + rel.replace("/", sep) if rel else "")


_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(name: str) -> str:
    """Make a Windows-safe file/folder name."""
    name = name.replace(":", " -")
    name = _ILLEGAL.sub("", name)
    name = re.sub(r"\s+", " ", name).strip().rstrip(". ")
    return name or "Untitled"


def file_key(path: str) -> str:
    """Identity for caches: path + size + mtime."""
    try:
        st = os.stat(path)
        return f"{path}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return path
