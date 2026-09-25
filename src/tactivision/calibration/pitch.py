"""Pitch template used by the 32-keypoint model, scaled to the StatsBomb frame."""

from __future__ import annotations

import numpy as np

# StatsBomb open-data frame. The keypoint template is a 120 m by 70 m schematic,
# scaled onto 120 by 80 so later event data uses the same axes.
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

_LENGTH_CM = 12000.0
_WIDTH_CM = 7000.0
_PENALTY_BOX_WIDTH_CM = 4100.0
_PENALTY_BOX_LENGTH_CM = 2015.0
_GOAL_BOX_WIDTH_CM = 1832.0
_GOAL_BOX_LENGTH_CM = 550.0
_CENTRE_CIRCLE_RADIUS_CM = 915.0
_PENALTY_SPOT_DISTANCE_CM = 1100.0


def _xy(x_cm: float, y_cm: float) -> tuple[float, float]:
    return (x_cm / _LENGTH_CM * PITCH_LENGTH, y_cm / _WIDTH_CM * PITCH_WIDTH)


def pitch_vertices() -> np.ndarray:
    """32 landmarks in keypoint-index order. Shape ``(32, 2)``."""
    w = _WIDTH_CM
    length = _LENGTH_CM
    pb_w = _PENALTY_BOX_WIDTH_CM
    pb_l = _PENALTY_BOX_LENGTH_CM
    gb_w = _GOAL_BOX_WIDTH_CM
    gb_l = _GOAL_BOX_LENGTH_CM
    radius = _CENTRE_CIRCLE_RADIUS_CM
    spot = _PENALTY_SPOT_DISTANCE_CM
    raw = [
        (0, 0),
        (0, (w - pb_w) / 2),
        (0, (w - gb_w) / 2),
        (0, (w + gb_w) / 2),
        (0, (w + pb_w) / 2),
        (0, w),
        (gb_l, (w - gb_w) / 2),
        (gb_l, (w + gb_w) / 2),
        (spot, w / 2),
        (pb_l, (w - pb_w) / 2),
        (pb_l, (w - gb_w) / 2),
        (pb_l, (w + gb_w) / 2),
        (pb_l, (w + pb_w) / 2),
        (length / 2, 0),
        (length / 2, w / 2 - radius),
        (length / 2, w / 2 + radius),
        (length / 2, w),
        (length - pb_l, (w - pb_w) / 2),
        (length - pb_l, (w - gb_w) / 2),
        (length - pb_l, (w + gb_w) / 2),
        (length - pb_l, (w + pb_w) / 2),
        (length - spot, w / 2),
        (length - gb_l, (w - gb_w) / 2),
        (length - gb_l, (w + gb_w) / 2),
        (length, 0),
        (length, (w - pb_w) / 2),
        (length, (w - gb_w) / 2),
        (length, (w + gb_w) / 2),
        (length, (w + pb_w) / 2),
        (length, w),
        (length / 2 - radius, w / 2),
        (length / 2 + radius, w / 2),
    ]
    return np.array([_xy(x, y) for x, y in raw], dtype=np.float32)


# 1-indexed edges of the pitch template.
PITCH_EDGES: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (7, 8),
    (10, 11),
    (11, 12),
    (12, 13),
    (14, 15),
    (15, 16),
    (16, 17),
    (18, 19),
    (19, 20),
    (20, 21),
    (23, 24),
    (25, 26),
    (26, 27),
    (27, 28),
    (28, 29),
    (29, 30),
    (1, 14),
    (2, 10),
    (3, 7),
    (4, 8),
    (5, 13),
    (6, 17),
    (14, 25),
    (18, 26),
    (23, 27),
    (24, 28),
    (21, 29),
    (17, 30),
)
