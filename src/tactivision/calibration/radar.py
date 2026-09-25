"""Draw a StatsBomb pitch and the players that passed calibration."""

from __future__ import annotations

import cv2
import numpy as np

from tactivision.calibration.pitch import PITCH_EDGES, PITCH_LENGTH, PITCH_WIDTH, pitch_vertices
from tactivision.detection.visualize import color_for_track


def draw_pitch(
    players: list[tuple[int, float, float]],
    *,
    title: str,
    width: int = 900,
    height: int = 620,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (24, 90, 40), dtype=np.uint8)
    margin = 40
    scale_x = (width - 2 * margin) / PITCH_LENGTH
    scale_y = (height - 2 * margin) / PITCH_WIDTH

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(margin + x * scale_x), int(margin + y * scale_y)

    vertices = pitch_vertices()
    for start, end in PITCH_EDGES:
        a = to_px(*vertices[start - 1])
        b = to_px(*vertices[end - 1])
        cv2.line(canvas, a, b, (230, 230, 230), 1, cv2.LINE_AA)
    center = to_px(PITCH_LENGTH / 2, PITCH_WIDTH / 2)
    radius_px = int((915.0 / 7000.0 * PITCH_WIDTH) * scale_y)
    cv2.circle(canvas, center, max(radius_px, 1), (230, 230, 230), 1, cv2.LINE_AA)

    for track_id, x, y in players:
        cv2.circle(canvas, to_px(x, y), 6, color_for_track(track_id), -1)
        cv2.putText(
            canvas,
            str(track_id),
            to_px(x, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
