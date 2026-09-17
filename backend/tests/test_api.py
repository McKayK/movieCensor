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


def test_review_tweaks_nudge_settings_and_rerender(client):
    from app import db
    from app.config import settings

    scan = client.post("/api/movies/42/scan", json={}).json()
    for _ in range(50):
        if scan["status"] != "running":
            break
        time.sleep(0.1)
        scan = client.get(f"/api/scans/{scan['id']}").json()
    job = client.post("/api/jobs", json={"ratingKey": "42", "scanId": scan["id"], "groups": ["f_word"], "echo": "duck"}).json()
    assert job["settings"]["echo"] == "duck"
    with db.session() as conn:
        conn.execute("INSERT INTO job_hits (job_id, source, group_id, word, cue_start, cue_end, status, word_start, word_end, "
                     "mute_start, mute_end, decision) VALUES (?, 'both', 'f_word', 'fuck', 6.9, 8.6, 'confirmed', 7.55, 7.95, 7.43, 8.0, 'mute')",
                     (job["id"],))
        db.update_job(conn, job["id"], status="needs_review", media_json='{"path": "x"}')
    hit = client.get(f"/api/jobs/{job['id']}").json()["hits"][0]
    r = client.post(f"/api/jobs/{job['id']}/hits/{hit['id']}/nudge", json={"start": 0.05}).json()
    r = client.post(f"/api/jobs/{job['id']}/hits/{hit['id']}/nudge", json={"start": 0.05, "end": 0.05}).json()
    assert (r["nudgeStart"], r["nudgeEnd"]) == (0.1, 0.05)
    big = client.post(f"/api/jobs/{job['id']}/hits/{hit['id']}/nudge", json={"start": 5}).json()
    assert big["nudgeStart"] == settings.max_nudge
    assert client.post(f"/api/jobs/{job['id']}/settings", json={"echo": "deep", "style": "bleep"}).json()["settings"]["echo"] == "deep"
    assert client.post(f"/api/jobs/{job['id']}/settings", json={"echo": "nope"}).status_code == 400
    with db.session() as conn:
        db.update_job(conn, job["id"], status="done")
    reopened = client.post(f"/api/jobs/{job['id']}/rerender").json()
    assert reopened["status"] == "needs_review"
    assert client.get(f"/api/jobs/{job['id']}").json()["hits"][0]["nudgeStart"] == settings.max_nudge
