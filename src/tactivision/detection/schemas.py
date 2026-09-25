"""Detection records written before tracking starts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        }


@dataclass(frozen=True)
class FrameDetections:
    frame_id: int
    timestamp: float
    detections: tuple[Detection, ...]

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(float(self.timestamp), 3),
            "detections": [detection.to_dict() for detection in self.detections],
        }
