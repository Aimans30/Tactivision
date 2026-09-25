"""Video metadata written before any detection runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    video: str
    fps: float
    width: int
    height: int
    duration_seconds: float
    frame_count: int
    coordinate_system: str = "statsbomb"
    pitch_length: float = 120.0
    pitch_width: float = 80.0

    def to_dict(self) -> dict:
        return asdict(self)


def write_metadata(path: Path, metadata: VideoMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2) + "\n", encoding="utf-8")
