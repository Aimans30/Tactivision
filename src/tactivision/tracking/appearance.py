"""Lightweight appearance descriptors for experimental identity remapping.

HSV torso histograms only — no pretrained ReID model. Experimental; not the
default tracker path.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AppearanceParams:
    """Centralized thresholds for appearance-assisted ID remapping."""

    hist_h_bins: int = 16
    hist_s_bins: int = 12
    min_crop_side: int = 16
    min_crop_area: int = 400
    # Use middle band of the box to reduce grass/head contamination.
    torso_y0: float = 0.2
    torso_y1: float = 0.75
    torso_x0: float = 0.15
    torso_x1: float = 0.85
    ema_alpha: float = 0.35
    # Remap a newly appeared ByteTrack ID onto a recently lost ID when all gates pass.
    appearance_threshold: float = 0.75
    max_gap_frames: int = 15
    max_center_distance_px: float = 120.0


# Named presets for the SNMOT sweep (baseline = remapper off).
PRESETS: dict[str, AppearanceParams] = {
    "weak": AppearanceParams(appearance_threshold=0.88, max_gap_frames=8, max_center_distance_px=80.0),
    "medium": AppearanceParams(appearance_threshold=0.75, max_gap_frames=15, max_center_distance_px=120.0),
    "strong": AppearanceParams(appearance_threshold=0.62, max_gap_frames=30, max_center_distance_px=200.0),
}


def clip_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    ix1 = max(0, int(np.floor(x1)))
    iy1 = max(0, int(np.floor(y1)))
    ix2 = min(width, int(np.ceil(x2)))
    iy2 = min(height, int(np.ceil(y2)))
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return None
    return ix1, iy1, ix2, iy2


def crop_quality_ok(clipped: tuple[int, int, int, int], params: AppearanceParams) -> bool:
    x1, y1, x2, y2 = clipped
    w, h = x2 - x1, y2 - y1
    return w >= params.min_crop_side and h >= params.min_crop_side and (w * h) >= params.min_crop_area


def extract_torso_bgr(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    params: AppearanceParams,
) -> np.ndarray | None:
    """Return a torso crop or None when the box/crop is unusable."""
    h, w = frame_bgr.shape[:2]
    clipped = clip_bbox(bbox, w, h)
    if clipped is None or not crop_quality_ok(clipped, params):
        return None
    x1, y1, x2, y2 = clipped
    bw, bh = x2 - x1, y2 - y1
    tx1 = x1 + int(bw * params.torso_x0)
    tx2 = x1 + int(bw * params.torso_x1)
    ty1 = y1 + int(bh * params.torso_y0)
    ty2 = y1 + int(bh * params.torso_y1)
    if tx2 - tx1 < 4 or ty2 - ty1 < 4:
        return None
    crop = frame_bgr[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None
    return crop


def appearance_vector(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    params: AppearanceParams | None = None,
) -> np.ndarray | None:
    """L2-normalized concatenated H+S histograms from the torso crop."""
    params = params or AppearanceParams()
    crop = extract_torso_bgr(frame_bgr, bbox, params)
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [params.hist_h_bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [params.hist_s_bins], [0, 256])
    vec = np.concatenate([hist_h.ravel(), hist_s.ravel()]).astype(np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return vec / norm


def appearance_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] for non-negative histograms (dot product)."""
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
