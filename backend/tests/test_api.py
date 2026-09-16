"""API smoke test with auth disabled and a fake Plex server."""
import os
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from tests.test_pipeline_integration import make_movie


class FakeServer:
    def __init__(self, path):
        self.path = path

    def movies(self, force=False):
        return [{"ratingKey": "42", "title": "Test Movie", "year": 2020, "thumb": "/library/metadata/42/thumb/1",
                 "contentRating": "R", "resolution": "240"}]

    def metadata(self, key):
        return {"ratingKey": "42", "title": "Test Movie", "year": 2020, "Guid": [{"id": "imdb://tt0000001"}],
                "Media": [{"Part": [{"file": self.path, "size": os.path.getsize(self.path)}]}]}

    def machine_identifier(self):
        return "fake"

    def censored_section_id(self):
        return None


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app import plex
    from app.config import settings

    movies = tmp_path / "movies"
    movies.mkdir()
    path = make_movie(str(movies))
    monkeypatch.setattr(settings, "path_maps", [])
    monkeypatch.setattr(settings, "censored_dir", str(tmp_path / "censored"))
    os.makedirs(settings.censored_dir, exist_ok=True)
    monkeypatch.setattr(plex, "_server", FakeServer(path))
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_browse_scan_and_create_job(client):
    me = client.get("/api/auth/me").json()
    assert me["isAdmin"] is True
    assert client.get("/api/movies?q=test").json()[0]["ratingKey"] == "42"
    detail = client.get("/api/movies/42").json()
    assert detail["fileReachable"] and detail["subtitleOptions"][0]["kind"] == "sidecar"

    scan = client.post("/api/movies/42/scan", json={}).json()
    for _ in range(50):
        if scan["status"] != "running":
            break
        time.sleep(0.1)
        scan = client.get(f"/api/scans/{scan['id']}").json()
    assert scan["status"] == "done", scan
    counts = {g["id"]: g["count"] for g in scan["groups"]}
    assert counts["f_word"] == 1 and counts["s_word"] == 1

    r = client.post("/api/jobs", json={"ratingKey": "42", "scanId": scan["id"], "groups": ["f_word"], "style": "bleep"})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] == "queued" and job["settings"]["guids"] == ["imdb://tt0000001"]
    dup = client.post("/api/jobs", json={"ratingKey": "42", "scanId": scan["id"], "groups": ["f_word"]})
    assert dup.status_code == 409
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "canceled"
    assert client.get("/api/wordgroups").json()["presets"][0]["id"] == "essential"
