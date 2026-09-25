"""Write sampled frames to disk. Off by default for full matches."""

from __future__ import annotations

from pathlib import Path

import cv2

from tactivision.video.reader import VideoReader


def extract_frames(
    reader: VideoReader,
    output_dir: Path,
    sample_fps: float,
    image_ext: str = "jpg",
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    ext = image_ext.lstrip(".").lower()
    for frame_id, _, frame in reader.iter_frames(sample_fps=sample_fps):
        path = output_dir / f"frame_{frame_id:06d}.{ext}"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Failed to write frame: {path}")
        saved += 1
    return saved
