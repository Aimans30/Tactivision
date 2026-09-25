"""Tests for local browser upload helpers and /api/runs validation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tactivision.api.app import app
from tactivision.api.uploads import (
    ensure_run_slot_free,
    extension_allowed,
    new_upload_run_id,
    parse_max_seconds,
    public_error_message,
    sanitize_upload_filename,
    upload_destination,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_sanitize_strips_path_traversal():
    assert sanitize_upload_filename("../etc/passwd.mp4") == "passwd.mp4"
    assert sanitize_upload_filename(r"..\\..\\secret.mkv") == "secret.mkv"
    with pytest.raises(ValueError):
        sanitize_upload_filename("")
    with pytest.raises(ValueError):
        sanitize_upload_filename("...")


def test_extension_validation():
    assert extension_allowed("clip.mp4")
    assert extension_allowed("clip.MKV")
    assert not extension_allowed("notes.txt")
    assert not extension_allowed("clip.exe")


def test_new_run_id_unique(tmp_path: Path):
    a = new_upload_run_id(processed_root=tmp_path)
    (tmp_path / a).mkdir()
    b = new_upload_run_id(processed_root=tmp_path)
    assert a != b
    assert a.startswith("upload_")
    assert b.startswith("upload_")


def test_upload_destination_isolated(tmp_path: Path):
    dest = upload_destination("upload_test_abc", "match.mp4", upload_root=tmp_path)
    assert dest.parent.name == "upload_test_abc"
    assert dest.name == "source.mp4"
    assert ".." not in str(dest)


def test_ensure_run_slot_free(tmp_path: Path):
    ensure_run_slot_free("fresh_run", processed_root=tmp_path)
    (tmp_path / "taken").mkdir()
    with pytest.raises(ValueError):
        ensure_run_slot_free("taken", processed_root=tmp_path)


def test_parse_max_seconds():
    assert parse_max_seconds("30") == 30.0
    assert parse_max_seconds("full") == 300.0
    with pytest.raises(ValueError):
        parse_max_seconds("nope")


def test_public_error_strips_paths():
    msg = public_error_message(ValueError(r"failed C:\Users\aiman\secret\video.mp4"))
    assert "aiman" not in msg.lower() or "<path>" in msg
    assert "traceback" not in msg.lower()


def test_upload_disabled_returns_503(client: TestClient):
    with patch("tactivision.api.app.uploads_disabled_reason", return_value="uploads off"):
        res = client.post(
            "/api/runs",
            files={"video": ("clip.mp4", b"bytes", "video/mp4")},
            data={"max_seconds": "5"},
        )
    assert res.status_code == 503
    assert "uploads off" in res.json()["detail"]


def test_upload_missing_file(client: TestClient):
    res = client.post("/api/runs", data={"max_seconds": "5"})
    assert res.status_code in {400, 422}


def test_upload_unsupported_extension(client: TestClient):
    res = client.post(
        "/api/runs",
        files={"video": ("notes.txt", b"hello", "text/plain")},
        data={"max_seconds": "5"},
    )
    assert res.status_code == 400
    assert "Unsupported" in res.json()["detail"]
    assert "C:" not in res.json()["detail"]
    assert "Users" not in res.json()["detail"]


def test_upload_empty_file(client: TestClient):
    with patch("tactivision.api.app.uploads_disabled_reason", return_value=None):
        res = client.post(
            "/api/runs",
            files={"video": ("empty.mp4", b"", "video/mp4")},
            data={"max_seconds": "5"},
        )
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert "empty" in detail.lower() or "unreadable" in detail.lower() or "missing" in detail.lower()


def test_upload_success_mocked(client: TestClient, tmp_path: Path):
    # Minimal fake mp4 bytes — VideoReader is mocked before open.
    fake_meta = MagicMock(width=64, height=48, fps=10.0)

    class FakeReader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def metadata(self):
            return fake_meta

    with (
        patch("tactivision.api.app.uploads_disabled_reason", return_value=None),
        patch("tactivision.api.app.VideoReader", return_value=FakeReader()),
        patch(
            "tactivision.api.app.run_match_pipeline",
            return_value={
                "runtime_seconds": 1.2,
                "match_bundle": "data/processed/upload_x/match_bundle.json",
            },
        ) as mocked,
        patch("tactivision.api.app.new_upload_run_id", return_value="upload_test_fixture"),
        patch("tactivision.api.app.PROCESSED", tmp_path / "processed"),
        patch("tactivision.api.uploads.UPLOAD_ROOT", tmp_path / "uploads"),
        patch("tactivision.api.app.upload_destination") as dest_fn,
    ):
        (tmp_path / "processed").mkdir()
        dest = tmp_path / "uploads" / "upload_test_fixture" / "source.mp4"
        dest.parent.mkdir(parents=True)
        dest_fn.return_value = dest

        res = client.post(
            "/api/runs",
            files={"video": ("clip.mp4", b"not-really-mp4-but-mocked", "video/mp4")},
            data={"max_seconds": "8"},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["status"] == "completed"
        assert body["run_id"] == "upload_test_fixture"
        assert body["source_filename"] == "clip.mp4"
        assert body["max_seconds"] == 8.0
        assert "C:" not in str(body)
        assert "Users" not in str(body)
        mocked.assert_called_once()
        assert mocked.call_args.args[1] == "upload_test_fixture"
        assert mocked.call_args.kwargs.get("max_seconds") == 8.0


def test_upload_pipeline_failure(client: TestClient, tmp_path: Path):
    fake_meta = MagicMock(width=64, height=48, fps=10.0)

    class FakeReader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def metadata(self):
            return fake_meta

    with (
        patch("tactivision.api.app.uploads_disabled_reason", return_value=None),
        patch("tactivision.api.app.VideoReader", return_value=FakeReader()),
        patch(
            "tactivision.api.app.run_match_pipeline",
            side_effect=RuntimeError(r"boom at C:\Users\aiman\secret"),
        ),
        patch("tactivision.api.app.new_upload_run_id", return_value="upload_fail_fixture"),
        patch("tactivision.api.app.PROCESSED", tmp_path / "processed"),
        patch("tactivision.api.app.upload_destination") as dest_fn,
    ):
        (tmp_path / "processed").mkdir()
        dest = tmp_path / "uploads" / "upload_fail_fixture" / "source.mp4"
        dest.parent.mkdir(parents=True)
        dest_fn.return_value = dest
        res = client.post(
            "/api/runs",
            files={"video": ("clip.mp4", b"bytes", "video/mp4")},
            data={"max_seconds": "5"},
        )
        assert res.status_code == 500
        detail = res.json()["detail"]
        assert "Processing failed" in detail
        assert "aiman" not in detail
        assert "C:" not in detail
