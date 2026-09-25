"""Ball detection and tracking schemas. Image-space only for this baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BallDetection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        cx, cy = self.center
        return {
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center": [round(cx, 1), round(cy, 1)],
        }


@dataclass(frozen=True)
class FrameBallDetections:
    frame_id: int
    timestamp: float
    detections: tuple[BallDetection, ...]

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(float(self.timestamp), 3),
            "detections": [det.to_dict() for det in self.detections],
        }


@dataclass(frozen=True)
class BallObservation:
    """One frame in the ball trajectory. Missing detections stay explicit."""

    frame_id: int
    timestamp: float
    observed: bool
    track_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        row: dict = {
            "frame_id": self.frame_id,
            "timestamp": round(float(self.timestamp), 3),
            "observed": self.observed,
        }
        if not self.observed:
            row["track_id"] = None
            row["bbox"] = None
            row["center"] = None
            row["confidence"] = None
            row["class"] = None
            return row
        assert self.bbox is not None and self.center is not None
        x1, y1, x2, y2 = self.bbox
        cx, cy = self.center
        row.update(
            {
                "track_id": self.track_id,
                "class": self.class_name or "ball",
                "confidence": None if self.confidence is None else round(float(self.confidence), 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "center": [round(cx, 1), round(cy, 1)],
            }
        )
        return row
