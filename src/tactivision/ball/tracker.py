"""Temporal ball tracker. Image-space association only; no pitch projection."""

from __future__ import annotations

import math

from tactivision.ball.schemas import BallDetection, BallObservation


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class BallTracker:
    """Greedy single-hypothesis tracker for the football.

    Takes the highest-confidence ball detection each frame (when any exist).
    Associates to the active track by IoU or center distance. Does not invent
    boxes on missed frames; those become ``observed=False`` observations.
    A new track segment ID starts after ``max_gap_frames`` consecutive misses
    or when association fails.
    """

    def __init__(
        self,
        *,
        min_iou: float = 0.05,
        max_center_distance_px: float = 120.0,
        max_gap_frames: int = 15,
    ) -> None:
        self.min_iou = min_iou
        self.max_center_distance_px = max_center_distance_px
        self.max_gap_frames = max_gap_frames
        self._next_id = 1
        self._active_id: int | None = None
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._last_center: tuple[float, float] | None = None
        self._gap = 0

    def update(
        self,
        frame_id: int,
        timestamp: float,
        detections: tuple[BallDetection, ...],
    ) -> BallObservation:
        if not detections:
            self._gap += 1
            if self._gap > self.max_gap_frames:
                self._active_id = None
                self._last_bbox = None
                self._last_center = None
            return BallObservation(
                frame_id=frame_id,
                timestamp=timestamp,
                observed=False,
            )

        best = detections[0]  # detector already sorts by confidence desc
        associated = False
        if self._active_id is not None and self._last_bbox is not None and self._last_center is not None:
            iou = _iou(best.bbox, self._last_bbox)
            dist = math.hypot(best.center[0] - self._last_center[0], best.center[1] - self._last_center[1])
            if iou >= self.min_iou or dist <= self.max_center_distance_px:
                associated = True

        if not associated or self._active_id is None:
            self._active_id = self._next_id
            self._next_id += 1

        self._last_bbox = best.bbox
        self._last_center = best.center
        self._gap = 0
        return BallObservation(
            frame_id=frame_id,
            timestamp=timestamp,
            observed=True,
            track_id=self._active_id,
            class_name=best.class_name,
            confidence=best.confidence,
            bbox=best.bbox,
            center=best.center,
        )
