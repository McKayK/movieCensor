"""Thin Plex Media Server + plex.tv client (JSON API via httpx)."""
from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from .config import settings

PLEX_TV = "https://plex.tv/api/v2"
PRODUCT = "Movie Censor"


def _tv_headers(token: str | None = None) -> dict[str, str]:
    h = {
        "Accept": "application/json",
        "X-Plex-Product": PRODUCT,
        "X-Plex-Client-Identifier": settings.get_client_id(),
    }
    if token:
        h["X-Plex-Token"] = token
    return h


class PlexError(RuntimeError):
    pass


class PlexServer:
    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url or settings.plex_url).rstrip("/")
        self.token = token if token is not None else settings.plex_token
        self._client = httpx.Client(
            base_url=self.url,
            headers={
                "Accept": "application/json",
                "X-Plex-Token": self.token,
                "X-Plex-Product": PRODUCT,
                "X-Plex-Client-Identifier": settings.get_client_id(),
            },
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._lock = threading.Lock()
        self._movies_cache: tuple[float, list[dict[str, Any]]] | None = None
        self._machine_id: str | None = None
        self._sections_cache: tuple[float, list[dict[str, Any]]] | None = None

    # -- low level -----------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            r = self._client.get(path, params=params)
        except httpx.HTTPError as e:
            raise PlexError(f"Cannot reach Plex at {self.url}: {e}") from e
        if r.status_code == 401:
            raise PlexError("Plex rejected PLEX_TOKEN (401)")
        if r.status_code >= 400:
            raise PlexError(f"Plex {path} returned {r.status_code}")
        if not r.content:
            return {}
        return r.json().get("MediaContainer", {})

    # -- server --------------------------------------------------------------
    def machine_identifier(self) -> str:
        if not self._machine_id:
            self._machine_id = self._get("/identity").get("machineIdentifier", "")
        return self._machine_id

    def sections(self) -> list[dict[str, Any]]:
        now = time.time()
        if self._sections_cache and now - self._sections_cache[0] < 300:
            return self._sections_cache[1]
        dirs = self._get("/library/sections").get("Directory", [])
        out = [
            {
                "id": str(d.get("key")),
                "title": d.get("title"),
                "type": d.get("type"),
                "locations": [loc.get("path") for loc in d.get("Location", [])],
            }
            for d in dirs
        ]
        self._sections_cache = (now, out)
        return out

    def censored_section_id(self) -> str | None:
        if settings.censored_section_id:
            return settings.censored_section_id
        host = (settings.censored_host_path or "").replace("\\", "/").rstrip("/").lower()
        if not host:
            return None
        for s in self.sections():
            for loc in s["locations"]:
                if (loc or "").replace("\\", "/").rstrip("/").lower() == host:
                    return s["id"]
        return None

    def movie_section_ids(self) -> list[str]:
        if settings.movie_section_ids:
            return settings.movie_section_ids
        censored = self.censored_section_id()
        return [s["id"] for s in self.sections() if s["type"] == "movie" and s["id"] != censored]

    def movies(self, force: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            now = time.time()
            if not force and self._movies_cache and now - self._movies_cache[0] < 300:
                return self._movies_cache[1]
            items: list[dict[str, Any]] = []
            for sid in self.movie_section_ids():
                data = self._get(f"/library/sections/{sid}/all", {"type": 1, "includeGuids": 1})
                for m in data.get("Metadata", []):
                    items.append(summarize(m, sid))
            items.sort(key=lambda m: (m.get("titleSort") or m["title"] or "").lower())
            self._movies_cache = (now, items)
            return items

    def metadata(self, rating_key: str) -> dict[str, Any]:
        data = self._get(f"/library/metadata/{rating_key}", {"includeGuids": 1})
        items = data.get("Metadata", [])
        if not items:
            raise PlexError(f"Movie {rating_key} not found in Plex")
        return items[0]

    def photo(self, thumb: str, width: int = 300, height: int = 450) -> tuple[bytes, str]:
        params = {"width": width, "height": height, "minSize": 1, "upscale": 1, "url": thumb}
        try:
            r = self._client.get("/photo/:/transcode", params=params, headers={"Accept": "image/*"})
        except httpx.HTTPError as e:
            raise PlexError(str(e)) from e
        if r.status_code >= 400:
            raise PlexError(f"thumb {r.status_code}")
        return r.content, r.headers.get("content-type", "image/jpeg")

    def refresh_path(self, section_id: str, host_path: str) -> None:
        try:
            self._client.get(f"/library/sections/{section_id}/refresh", params={"path": host_path})
        except httpx.HTTPError as e:
            raise PlexError(str(e)) from e


def summarize(m: dict[str, Any], section_id: str | None = None) -> dict[str, Any]:
    media = (m.get("Media") or [{}])[0]
    part = (media.get("Part") or [{}])[0]
    return {
        "ratingKey": str(m.get("ratingKey")),
        "title": m.get("title") or "Untitled",
        "titleSort": m.get("titleSort"),
        "year": m.get("year"),
        "thumb": m.get("thumb"),
        "art": m.get("art"),
        "summary": m.get("summary"),
        "duration": m.get("duration"),
        "contentRating": m.get("contentRating"),
        "sectionId": section_id or str(m.get("librarySectionID") or ""),
        "file": part.get("file"),
        "size": part.get("size"),
        "resolution": media.get("videoResolution"),
        "audioChannels": media.get("audioChannels"),
        "guids": [g.get("id") for g in m.get("Guid", []) if g.get("id")],
    }


def guid_tag(guids: list[str]) -> str:
    """Plex naming hint, e.g. {imdb-tt1375666} or {tmdb-27205}."""
    for prefix, name in (("imdb://", "imdb"), ("tmdb://", "tmdb"), ("tvdb://", "tvdb")):
        for g in guids:
            if g.startswith(prefix):
                return "{" + f"{name}-{g[len(prefix):]}" + "}"
    return ""


# -- plex.tv sign-in (PIN flow) -----------------------------------------------
def create_pin() -> dict[str, Any]:
    r = httpx.post(f"{PLEX_TV}/pins", params={"strong": "true"}, headers=_tv_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def auth_url(code: str, forward_url: str) -> str:
    params = {
        "clientID": settings.get_client_id(),
        "code": code,
        "forwardUrl": forward_url,
        "context[device][product]": PRODUCT,
    }
    return "https://app.plex.tv/auth#?" + urlencode(params)


def check_pin(pin_id: int | str) -> str | None:
    r = httpx.get(f"{PLEX_TV}/pins/{pin_id}", headers=_tv_headers(), timeout=15)
    r.raise_for_status()
    return r.json().get("authToken")


def plex_user(token: str) -> dict[str, Any]:
    r = httpx.get(f"{PLEX_TV}/user", headers=_tv_headers(token), timeout=15)
    r.raise_for_status()
    return r.json()


def user_server_access(token: str, machine_id: str) -> dict[str, Any] | None:
    """Return the resource entry for this server if the signed-in user can access it."""
    r = httpx.get(
        f"{PLEX_TV}/resources", params={"includeHttps": 1, "includeRelay": 1}, headers=_tv_headers(token), timeout=20
    )
    r.raise_for_status()
    for res in r.json():
        if res.get("clientIdentifier") == machine_id and "server" in (res.get("provides") or ""):
            return res
    return None


_server: PlexServer | None = None


def server() -> PlexServer:
    global _server
    if _server is None:
        _server = PlexServer()
    return _server
