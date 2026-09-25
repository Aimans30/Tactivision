"""Tracked players with persistent IDs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    track_id: int | None
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        }


@dataclass(frozen=True)
class FrameTracks:
    frame_id: int
    timestamp: float
    tracks: tuple[Track, ...]
    shot_id: int = 0

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(float(self.timestamp), 3),
            "shot_id": self.shot_id,
            "tracks": [track.to_dict() for track in self.tracks],
        }
