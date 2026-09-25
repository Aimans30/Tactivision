import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tactivision.pipeline.runner import run_ingestion
from tactivision.video.reader import VideoReader


def _write_clip(path: Path, *, frames: int = 10, fps: float = 10.0, size: tuple[int, int] = (64, 48)) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    if not writer.isOpened():
        pytest.skip("OpenCV could not open an MJPG writer on this machine")
    width, height = size
    for index in range(frames):
        frame = np.full((height, width, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_reader_reports_container_metadata(tmp_path: Path) -> None:
    video = tmp_path / "match.avi"
    _write_clip(video)

    with VideoReader(video) as reader:
        meta = reader.metadata()

    assert meta.video == "match.avi"
    assert meta.width == 64
    assert meta.height == 48
    assert meta.fps == pytest.approx(10.0, abs=0.1)
    assert meta.frame_count == 10
    assert meta.duration_seconds == pytest.approx(1.0, abs=0.05)
    assert meta.coordinate_system == "statsbomb"
    assert meta.pitch_length == 120
    assert meta.pitch_width == 80


def test_ingestion_writes_metadata_without_frames(tmp_path: Path) -> None:
    video = tmp_path / "match.avi"
    _write_clip(video)
    output = tmp_path / "processed"

    run_dir = run_ingestion(video, output, extract=False)

    metadata_path = run_dir / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["video"] == "match.avi"
    assert payload["frame_count"] == 10
    assert not (run_dir / "frames").exists()


def test_ingestion_samples_frames(tmp_path: Path) -> None:
    video = tmp_path / "match.avi"
    _write_clip(video, frames=10, fps=10.0)
    output = tmp_path / "processed"

    run_dir = run_ingestion(video, output, extract=True, sample_fps=5.0)

    frames = sorted((run_dir / "frames").glob("frame_*.jpg"))
    # 10 fps source, 5 fps sample => every 2nd frame: 0, 2, 4, 6, 8
    assert [path.name for path in frames] == [
        "frame_000000.jpg",
        "frame_000002.jpg",
        "frame_000004.jpg",
        "frame_000006.jpg",
        "frame_000008.jpg",
    ]


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "missing.mp4")
