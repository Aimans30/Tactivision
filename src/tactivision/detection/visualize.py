"""Draw player boxes on a frame for a visual sanity check."""

from __future__ import annotations

import cv2

from tactivision.detection.schemas import Detection
from tactivision.tracking.schemas import Track


def draw_detections(frame: object, detections: tuple[Detection, ...], timestamp: float) -> object:
    canvas = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 220, 80), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 220, 80),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"t={timestamp:.2f}s  n={len(detections)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def color_for_track(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (180, 180, 180)
    return ((track_id * 47) % 200 + 40, (track_id * 91) % 200 + 40, (track_id * 13) % 200 + 40)


def draw_tracks(frame: object, tracks: tuple[Track, ...], timestamp: float) -> object:
    canvas = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        color = color_for_track(track.track_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = "?" if track.track_id is None else str(track.track_id)
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"t={timestamp:.2f}s  tracks={len(tracks)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas
