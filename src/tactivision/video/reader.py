"""Open a football video and expose container metadata plus sampled frames."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from tactivision.video.metadata import VideoMetadata


class VideoReader:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {self.path}")

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def metadata(self) -> VideoMetadata:
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps < 0:
            fps = 0.0
        if frame_count < 0:
            frame_count = 0
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return VideoMetadata(
            video=self.path.name,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=round(duration, 3),
            frame_count=frame_count,
        )

    def iter_frames(self, sample_fps: float | None = None) -> Iterator[tuple[int, float, object]]:
        """Yield ``(frame_id, timestamp_seconds, bgr_frame)``.

        ``sample_fps`` keeps one frame per interval. ``None`` yields every frame.
        """
        info = self.metadata()
        if info.fps <= 0:
            raise ValueError(f"Video has no usable fps: {self.path}")
        step = 1
        if sample_fps is not None:
            if sample_fps <= 0:
                raise ValueError("sample_fps must be positive")
            step = max(1, round(info.fps / sample_fps))

        frame_id = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            if frame_id % step == 0:
                yield frame_id, frame_id / info.fps, frame
            frame_id += 1
