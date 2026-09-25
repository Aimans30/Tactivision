"""Pitch-coordinate heatmaps. These are locations, not distance or team shape."""

from __future__ import annotations

import cv2
import numpy as np

from tactivision.calibration.pitch import PITCH_EDGES, PITCH_LENGTH, PITCH_WIDTH, pitch_vertices

SPARSE_SAMPLE_THRESHOLD = 5
# One yard per bin, then a small blur so a sample is a patch rather than a pixel.
BIN_YARDS = 1.0
BLUR_YARDS = 2.0


def smoothed_points(rows: list[dict]) -> list[tuple[int, float, float]]:
    """Keep calibrated smoothed locations. Track ids are not merged."""
    points = []
    for row in rows:
        if not row.get("calibration_valid", False):
            continue
        points.append((int(row["track_id"]), float(row["x_smooth"]), float(row["y_smooth"])))
    return points


def render_heatmap(
    points: list[tuple[float, float]],
    *,
    title: str,
    width: int = 900,
    height: int = 620,
) -> np.ndarray:
    margin = 40
    canvas = np.full((height, width, 3), (24, 90, 40), dtype=np.uint8)
    scale_x = (width - 2 * margin) / PITCH_LENGTH
    scale_y = (height - 2 * margin) / PITCH_WIDTH
    pitch_w = width - 2 * margin
    pitch_h = height - 2 * margin

    density = _density(points)
    if density.max() > 0:
        colored = cv2.applyColorMap(_norm_u8(density), cv2.COLORMAP_INFERNO)
        colored = cv2.resize(colored, (pitch_w, pitch_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(_norm_u8(density), (pitch_w, pitch_h), interpolation=cv2.INTER_LINEAR) > 8
        canvas[margin : margin + pitch_h, margin : margin + pitch_w][mask] = colored[mask]

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(margin + x * scale_x), int(margin + y * scale_y)

    vertices = pitch_vertices()
    for start, end in PITCH_EDGES:
        cv2.line(canvas, to_px(*vertices[start - 1]), to_px(*vertices[end - 1]), (240, 240, 240), 1, cv2.LINE_AA)
    center = to_px(PITCH_LENGTH / 2, PITCH_WIDTH / 2)
    radius_px = int((915.0 / 7000.0 * PITCH_WIDTH) * scale_y)
    cv2.circle(canvas, center, max(radius_px, 1), (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _density(points: list[tuple[float, float]]) -> np.ndarray:
    bins_x = int(PITCH_LENGTH / BIN_YARDS)
    bins_y = int(PITCH_WIDTH / BIN_YARDS)
    heat = np.zeros((bins_y, bins_x), dtype=np.float32)
    for x, y in points:
        ix = int(np.clip(x / BIN_YARDS, 0, bins_x - 1))
        iy = int(np.clip(y / BIN_YARDS, 0, bins_y - 1))
        heat[iy, ix] += 1.0
    sigma = BLUR_YARDS / BIN_YARDS
    return cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _norm_u8(density: np.ndarray) -> np.ndarray:
    peak = float(density.max())
    if peak <= 0:
        return np.zeros(density.shape, dtype=np.uint8)
    return np.clip(density / peak * 255.0, 0, 255).astype(np.uint8)
