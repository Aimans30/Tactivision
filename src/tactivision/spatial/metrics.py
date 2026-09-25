"""Estimated distance and speed from smoothed pitch tracks.

These numbers are model-derived. They are not measured tracking devices.
"""

from __future__ import annotations

import numpy as np

MAX_GAP_SECONDS = 0.75
ESTIMATE_LABEL = "estimated_model_derived"


def measure_track(samples: list[dict]) -> dict | None:
    """Sum movement only across consecutive samples inside one stretch."""
    ordered = sorted(samples, key=lambda row: (row["timestamp"], row["frame"]))
    used: set[int] = set()
    duration = 0.0
    smooth_distance = 0.0
    raw_distance = 0.0
    smooth_speeds: list[float] = []
    raw_speeds: list[float] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        dt = float(right["timestamp"]) - float(left["timestamp"])
        if dt <= 0 or dt > MAX_GAP_SECONDS:
            continue
        if not left.get("calibration_valid", False) or not right.get("calibration_valid", False):
            continue
        smooth = _step(left, right, "x_smooth", "y_smooth")
        raw = _step(left, right, "x_raw", "y_raw")
        duration += dt
        smooth_distance += smooth
        raw_distance += raw
        smooth_speeds.append(smooth / dt)
        raw_speeds.append(raw / dt)
        used.add(int(left["frame"]))
        used.add(int(right["frame"]))
    if not used:
        return None
    return {
        "track_id": int(ordered[0]["track_id"]),
        "duration_seconds": round(duration, 3),
        "distance_yards": round(smooth_distance, 2),
        "mean_speed_yards_per_second": round(smooth_distance / duration, 3),
        "max_speed_yards_per_second": round(max(smooth_speeds), 3),
        "valid_samples": len(used),
        "distance_yards_raw": round(raw_distance, 2),
        "mean_speed_yards_per_second_raw": round(raw_distance / duration, 3),
        "max_speed_yards_per_second_raw": round(max(raw_speeds), 3),
        "metric_type": ESTIMATE_LABEL,
    }


def summarize(rows: list[dict]) -> dict:
    distances = np.array([row["distance_yards"] for row in rows], dtype=np.float64)
    speeds = np.array([row["mean_speed_yards_per_second"] for row in rows], dtype=np.float64)
    return {
        "metric_type": ESTIMATE_LABEL,
        "coordinate_system": "statsbomb_120x80_yards",
        "coordinates": "smoothed",
        "max_gap_seconds": MAX_GAP_SECONDS,
        "tracks_measured": len(rows),
        "distance_yards": _range(distances),
        "mean_speed_yards_per_second": _range(speeds),
    }


def _step(left: dict, right: dict, x_key: str, y_key: str) -> float:
    return float(np.hypot(right[x_key] - left[x_key], right[y_key] - left[y_key]))


def _range(values: np.ndarray) -> dict | None:
    if values.size == 0:
        return None
    return {
        "minimum": round(float(values.min()), 3),
        "median": round(float(np.median(values)), 3),
        "maximum": round(float(values.max()), 3),
    }
