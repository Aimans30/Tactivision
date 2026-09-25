"""Detect broadcast cuts from SoccerNet labels or frame appearance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Shot:
    shot_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    label: str | None = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "label": self.label,
        }


@dataclass(frozen=True)
class ShotBoundary:
    """A time where a new shot starts."""

    timestamp: float
    label: str | None = None


class ShotDetector:
    """Histogram-distance cut detector used when camera labels are absent."""

    def __init__(self, *, min_hist_distance: float = 0.45, min_cut_gap_frames: int = 12) -> None:
        self.min_hist_distance = min_hist_distance
        self.min_cut_gap_frames = min_cut_gap_frames
        self._previous: np.ndarray | None = None
        self._last_cut_frame: int | None = None

    def reset(self) -> None:
        self._previous = None
        self._last_cut_frame = None

    def is_cut(self, frame: object, frame_id: int) -> bool:
        hist = _histogram(frame)
        if self._previous is None:
            self._previous = hist
            return False
        distance = float(cv2.compareHist(self._previous, hist, cv2.HISTCMP_BHATTACHARYYA))
        self._previous = hist
        if distance < self.min_hist_distance:
            return False
        if self._last_cut_frame is not None and frame_id - self._last_cut_frame < self.min_cut_gap_frames:
            return False
        self._last_cut_frame = frame_id
        return True


def load_soccernet_boundaries(
    labels_path: Path | str,
    *,
    half: int = 1,
    max_seconds: float | None = None,
) -> list[ShotBoundary]:
    """Read SoccerNet ``Labels-cameras.json`` shot starts for one half."""
    payload = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    prefix = f"{half} -"
    starts = [ShotBoundary(0.0, label="clip_start")]
    for annotation in payload.get("annotations", []):
        game_time = str(annotation.get("gameTime", ""))
        if not game_time.startswith(prefix):
            continue
        timestamp = int(annotation["position"]) / 1000.0
        if max_seconds is not None and timestamp >= max_seconds:
            continue
        if timestamp <= 0:
            continue
        starts.append(ShotBoundary(timestamp, label=str(annotation.get("label"))))
    starts.sort(key=lambda item: item.timestamp)
    # Deduplicate near-identical timestamps.
    unique: list[ShotBoundary] = []
    for boundary in starts:
        if unique and abs(boundary.timestamp - unique[-1].timestamp) < 0.04:
            continue
        unique.append(boundary)
    return unique


def find_camera_labels(video_path: Path | str) -> Path | None:
    path = Path(video_path)
    candidate = path.parent / "Labels-cameras.json"
    return candidate if candidate.is_file() else None


def finalize_shots(
    boundaries: list[tuple[int, float, str | None]],
    last_frame: int,
    last_time: float,
) -> list[Shot]:
    """``boundaries`` are (frame_id, timestamp, label) where each shot starts."""
    if not boundaries:
        return []
    starts = sorted(boundaries, key=lambda item: item[0])
    shots: list[Shot] = []
    for index, (start_frame, start_time, label) in enumerate(starts):
        if index + 1 < len(starts):
            end_frame = starts[index + 1][0] - 1
            end_time = starts[index + 1][1]
        else:
            end_frame = last_frame
            end_time = last_time
        if end_frame < start_frame:
            continue
        shots.append(
            Shot(
                shot_id=index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
                label=label,
            )
        )
    return shots


def _histogram(frame: object) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist
