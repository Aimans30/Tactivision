"""Drop obvious non-players before any spatial metric is computed."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


@dataclass(frozen=True)
class TrackFilter:
    track_id: int
    kept: bool
    reason: str
    team: int | None
    inside_fraction: float
    observation_count: int
    median_x: float | None
    median_y: float | None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "kept": self.kept,
            "reason": self.reason,
            "team": self.team,
            "inside_fraction": round(self.inside_fraction, 3),
            "observation_count": self.observation_count,
            "median_x": None if self.median_x is None else round(self.median_x, 2),
            "median_y": None if self.median_y is None else round(self.median_y, 2),
        }


def inside_pitch(x: float, y: float) -> bool:
    """Playable pitch. The calibration margin that kept touchline staff is not used."""
    return 0.0 <= x <= PITCH_LENGTH and 0.0 <= y <= PITCH_WIDTH


def pitch_keep(
    track_id: int,
    points: list[tuple[float, float]],
    *,
    min_inside: int = 2,
    min_inside_fraction: float = 0.5,
) -> TrackFilter:
    if not points:
        return TrackFilter(track_id, False, "no_pitch_position", None, 0.0, 0, None, None)
    array = np.array(points, dtype=np.float64)
    inside = np.array([inside_pitch(float(x), float(y)) for x, y in points])
    fraction = float(inside.mean())
    median_x = float(np.median(array[:, 0]))
    median_y = float(np.median(array[:, 1]))
    kept = int(inside.sum()) >= min_inside and fraction >= min_inside_fraction
    reason = "on_pitch" if kept else "off_pitch"
    return TrackFilter(track_id, kept, reason, None, fraction, len(points), median_x, median_y)


def cluster_teams(
    features: np.ndarray,
    *,
    outlier_gap_fraction: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-means on appearance features.

    Returns ``(labels, keep_mask, centroids)``. A detection far from both kit
    colors is marked not-kept. That is the referee / other reject, not a third team.
    """
    count = len(features)
    if count == 0:
        empty = np.empty((0,), dtype=int)
        return empty, np.empty((0,), dtype=bool), np.empty((0, features.shape[1] if features.ndim == 2 else 0))
    if count == 1:
        return np.array([0]), np.array([True]), features.copy()

    labels, centroids = _kmeans2(features)
    nearest = np.linalg.norm(features - centroids[labels], axis=1)
    gap = float(np.linalg.norm(centroids[0] - centroids[1]))
    median = float(np.median(nearest))
    mad = float(np.median(np.abs(nearest - median)))
    limit = max(median + 2.5 * mad, outlier_gap_fraction * gap)
    keep = nearest <= limit
    # A one-point "team" is not a kit cluster. Leave it for the caller to drop.
    for label in (0, 1):
        if int((labels == label).sum()) < 2:
            keep[labels == label] = False
    return labels, keep, centroids


def assign_teams(
    on_pitch: list[TrackFilter],
    features_by_id: dict[int, np.ndarray],
) -> list[TrackFilter]:
    ordered = [item for item in on_pitch if item.track_id in features_by_id]
    missing = [item for item in on_pitch if item.track_id not in features_by_id]
    dropped = [
        TrackFilter(
            item.track_id,
            False,
            "no_appearance",
            None,
            item.inside_fraction,
            item.observation_count,
            item.median_x,
            item.median_y,
        )
        for item in missing
    ]
    if not ordered:
        return dropped
    features = np.stack([features_by_id[item.track_id] for item in ordered])
    labels, keep, _ = cluster_teams(features)
    assigned: list[TrackFilter] = []
    for item, label, kept in zip(ordered, labels, keep, strict=True):
        if not kept:
            assigned.append(
                TrackFilter(
                    item.track_id,
                    False,
                    "not_a_team_kit",
                    None,
                    item.inside_fraction,
                    item.observation_count,
                    item.median_x,
                    item.median_y,
                )
            )
            continue
        assigned.append(
            TrackFilter(
                item.track_id,
                True,
                "team",
                int(label),
                item.inside_fraction,
                item.observation_count,
                item.median_x,
                item.median_y,
            )
        )
    return assigned + dropped


def torso_color(frame: object, bbox: list[float]) -> np.ndarray | None:
    """Median jersey color, skipping the head, the legs, and green grass."""
    x1, y1, x2, y2 = (float(value) for value in bbox)
    height, width = frame.shape[:2]
    box_h = y2 - y1
    box_w = x2 - x1
    if box_h < 8 or box_w < 4:
        return None
    top = int(np.clip(y1 + 0.18 * box_h, 0, height - 1))
    bottom = int(np.clip(y1 + 0.55 * box_h, 0, height))
    left = int(np.clip(x1 + 0.25 * box_w, 0, width - 1))
    right = int(np.clip(x2 - 0.25 * box_w, 0, width))
    if bottom - top < 2 or right - left < 2:
        return None
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    pixels = crop.reshape(-1, 3).astype(np.float64)
    blue, green, red = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    grass = (green > red + 10) & (green > blue + 10)
    pixels = pixels[~grass]
    if len(pixels) < 4:
        return None
    return np.median(pixels, axis=0)


def kit_feature(bgr: np.ndarray) -> np.ndarray:
    """Hue as a circle, scaled by saturation, so white does not get a fake hue."""
    pixel = np.uint8([[bgr.astype(np.uint8)]])
    hue, saturation, _value = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    angle = float(hue) / 180.0 * 2.0 * np.pi
    sat = float(saturation) / 255.0
    return np.array([sat * np.cos(angle), sat * np.sin(angle), sat], dtype=np.float64)


def _kmeans2(features: np.ndarray, steps: int = 15) -> tuple[np.ndarray, np.ndarray]:
    first = features[0]
    farthest = int(np.argmax(np.linalg.norm(features - first, axis=1)))
    centroids = np.stack([first, features[farthest]]).astype(np.float64)
    labels = np.zeros(len(features), dtype=int)
    for _ in range(steps):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        if len(set(labels.tolist())) < 2:
            break
        for label in (0, 1):
            members = features[labels == label]
            if len(members):
                centroids[label] = members.mean(axis=0)
    return labels, centroids
