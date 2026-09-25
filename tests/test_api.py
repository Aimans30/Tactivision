"""Lightweight FastAPI smoke tests against processed runs when present."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tactivision.api.app import app

RUN = Path("data/processed/1_720p")
SNMOT = Path("data/processed/snmot101_e2e")
HAS_DEMO = (RUN / "match_bundle.json").exists()
HAS_SNMOT = (SNMOT / "match_bundle.json").exists()


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["processed_root"] == "data/processed"
    assert "C:" not in body["processed_root"]
    assert "uploads_enabled" in body
    assert "bundle_runs" in body
    assert isinstance(body["bundle_runs"], int)

def test_list_matches(client: TestClient):
    response = client.get("/api/matches")
    assert response.status_code == 200
    matches = response.json()["matches"]
    assert isinstance(matches, list)
    assert all(not m["match_id"].startswith("_") for m in matches)
    if HAS_DEMO:
        ids = {m["match_id"] for m in matches}
        assert "1_720p" in ids
        demo = next(m for m in matches if m["match_id"] == "1_720p")
        assert demo["has_bundle"] is True
        assert "available_media" in demo
        assert "source" in demo
        assert "fps" in demo


@pytest.mark.skipif(not HAS_DEMO, reason="1_720p demo not present")
def test_match_summary_and_timeline(client: TestClient):
    match = client.get("/api/matches/1_720p")
    assert match.status_code == 200
    body = match.json()
    assert body["match_id"] == "1_720p"
    assert "summaries" in body

    poss = client.get("/api/matches/1_720p/summary/possession")
    assert poss.status_code == 200
    assert poss.json().get("ground_truth") is False

    timeline = client.get("/api/matches/1_720p/possession?limit=10")
    assert timeline.status_code == 200
    payload = timeline.json()
    assert payload["count"] >= 1
    assert len(payload["rows"]) <= 10
    row = payload["rows"][0]
    assert "possession_status" in row
    assert "timestamp" in row


@pytest.mark.skipif(not HAS_DEMO, reason="1_720p demo not present")
def test_media_and_ask(client: TestClient):
    media = client.get("/api/matches/1_720p/media/possession_timeline")
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("image/")

    ask = client.post("/api/matches/1_720p/ask", json={"question": "How did possession change?"})
    assert ask.status_code == 200
    body = ask.json()
    assert body.get("ground_truth") is False
    assert body.get("facts")
    assert "narrative" in body


@pytest.mark.skipif(not HAS_SNMOT, reason="snmot101_e2e run not present")
def test_second_run_discovered_and_ask_isolated(client: TestClient):
    matches = client.get("/api/matches").json()["matches"]
    ids = {m["match_id"] for m in matches}
    assert "snmot101_e2e" in ids
    entry = next(m for m in matches if m["match_id"] == "snmot101_e2e")
    assert entry["has_bundle"] is True
    assert entry.get("width") == 1920
    assert "tracking" in entry.get("available_media", [])

    bundle = client.get("/api/matches/snmot101_e2e")
    assert bundle.status_code == 200
    assert bundle.json()["match_id"] == "snmot101_e2e"

    tracking = client.get("/api/matches/snmot101_e2e/media/tracking")
    assert tracking.status_code == 200

    ask = client.post(
        "/api/matches/snmot101_e2e/ask",
        json={"question": "What is estimated possession coverage?"},
    )
    assert ask.status_code == 200
    body = ask.json()
    assert body.get("ground_truth") is False
    assert "narrative" in body


def test_static_dashboard_assets(client: TestClient):
    index = client.get("/")
    assert index.status_code == 200
    html = index.text
    assert "matchSelect" in html
    assert "1_720p" not in html
    js = client.get("/app.js")
    assert js.status_code == 200
    assert "loadMatch" in js.text
    assert 'match_id === "1_720p"' not in js.text
