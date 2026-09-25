"""Tests for the generic end-to-end match pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tactivision.analytics.bundle import build_match_bundle, write_match_bundle
from tactivision.pipeline.paths import make_run_paths, project_relative, validate_run_id
from tactivision.pipeline.runner import run_ingestion


def test_validate_run_id():
    assert validate_run_id("snmot_clip") == "snmot_clip"
    assert validate_run_id("1_720p") == "1_720p"
    with pytest.raises(ValueError):
        validate_run_id("../escape")
    with pytest.raises(ValueError):
        validate_run_id("has spaces")


def test_make_run_paths_isolated(tmp_path: Path):
    paths = make_run_paths(tmp_path, "run_a")
    assert paths.root == tmp_path / "run_a"
    assert paths.tracks == tmp_path / "run_a" / "tracks.jsonl"
    assert "1_720p" not in str(paths.tracks)


def test_project_relative_no_absolute(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    target = root / "data" / "raw" / "clip.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    rel = project_relative(target, root)
    assert rel == "data/raw/clip.mp4"
    assert ":\\" not in rel
    assert not rel.startswith("/Users/")


def test_ingest_arbitrary_run_id(tmp_path: Path):
    video = tmp_path / "fixture.mp4"
    _write_tiny_video(video, frames=5, fps=5.0)
    out = tmp_path / "processed"
    run_dir = run_ingestion(video, out, run_id="fixture_run")
    assert run_dir == out / "fixture_run"
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "fixture_run"
    assert meta["source_filename"] == "fixture.mp4"
    assert meta["width"] > 0
    assert meta["height"] > 0
    assert meta["fps"] > 0


def test_ingest_invalid_video(tmp_path: Path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError):
        run_ingestion(missing, tmp_path / "processed", run_id="bad")


def test_bundle_empty_run_no_crash(tmp_path: Path):
    run = tmp_path / "empty_run"
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps({"video": "x.mp4", "fps": 25.0, "width": 64, "height": 64}) + "\n",
        encoding="utf-8",
    )
    bundle = write_match_bundle(run, clip_seconds=1.0, video_source="clips/x.mp4")
    assert bundle["match_id"] == "empty_run"
    assert bundle["video"]["source"] == "clips/x.mp4"
    assert (run / "match_bundle.json").is_file()
    text = (run / "match_bundle.json").read_text(encoding="utf-8")
    assert "C:\\Users" not in text
    assert "/home/" not in text


def test_bundle_strips_absolute_source(tmp_path: Path):
    run = tmp_path / "abs_run"
    run.mkdir()
    computed = build_match_bundle(
        run,
        clip_seconds=1.0,
        video_source=r"C:\Users\someone\Videos\match.mp4",
    )
    assert computed["bundle"]["video"]["source"] == "match.mp4"


def _write_tiny_video(path: Path, *, frames: int, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (40, 120 + (i % 20), 40)
        writer.write(frame)
    writer.release()
