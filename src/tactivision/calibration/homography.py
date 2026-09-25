"""Image pixels to StatsBomb pitch coordinates."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_homography(
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    *,
    ransac_px: float = 8.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Map image pixels onto the pitch. Returns ``(H, inlier_mask)`` or nulls."""
    if len(image_points) < 4 or len(image_points) != len(pitch_points):
        return None, None
    matrix, mask = cv2.findHomography(
        image_points.astype(np.float32),
        pitch_points.astype(np.float32),
        cv2.RANSAC,
        ransac_px,
    )
    if matrix is None or mask is None:
        return None, None
    return matrix, mask.ravel().astype(bool)


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    shaped = points.astype(np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(shaped, homography)
    return projected.reshape(-1, 2)


def mean_reprojection_error(
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    homography: np.ndarray,
    inliers: np.ndarray | None = None,
) -> float:
    projected = transform_points(image_points, homography)
    error = np.linalg.norm(projected - pitch_points, axis=1)
    if inliers is not None and inliers.any():
        error = error[inliers]
    if len(error) == 0:
        return float("inf")
    return float(error.mean())


def on_pitch(points: np.ndarray, *, margin: float = 3.0) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    return (
        (x >= -margin)
        & (x <= 120.0 + margin)
        & (y >= -margin)
        & (y <= 80.0 + margin)
    )
