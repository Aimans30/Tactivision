"""Image-space trajectories from persistent tracks.

These are pixel coordinates, not pitch coordinates. Homography comes later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrajectoryPoint:
    frame_id: int
    timestamp: float
    x: float
    y: float

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "x": round(self.x, 1),
            "y": round(self.y, 1),
        }


@dataclass(frozen=True)
class Trajectory:
    track_id: int
    segment: int
    points: tuple[TrajectoryPoint, ...]

    @property
    def duration_seconds(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].timestamp - self.points[0].timestamp

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "segment": self.segment,
            "start": round(self.points[0].timestamp, 3),
            "end": round(self.points[-1].timestamp, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "points": [point.to_dict() for point in self.points],
        }


def foot_point(bbox: list[float]) -> tuple[float, float]:
    """Bottom-center of the box. This is the contact point until pitch mapping exists."""
    x1, _, x2, y2 = bbox
    return (float(x1) + float(x2)) / 2.0, float(y2)


def build_trajectories(
    rows: list[dict],
    *,
    min_seconds: float = 2.0,
    max_jump_px: float = 120.0,
    max_gap_seconds: float = 0.2,
) -> list[Trajectory]:
    """Turn track rows into path segments.

    A large jump or a time gap starts a new segment, so a broadcast cut does not
    draw a line across the frame. Segments shorter than ``min_seconds`` are dropped.
    """
    by_id: dict[int, list[TrajectoryPoint]] = {}
    for row in rows:
        frame_id = int(row["frame_id"])
        timestamp = float(row["timestamp"])
        for track in row["tracks"]:
            if track.get("track_id") is None:
                continue
            x, y = foot_point(track["bbox"])
            by_id.setdefault(int(track["track_id"]), []).append(
                TrajectoryPoint(frame_id, timestamp, x, y)
            )

    kept: list[Trajectory] = []
    for track_id, points in by_id.items():
        points.sort(key=lambda point: point.frame_id)
        segment_index = 0
        current: list[TrajectoryPoint] = []
        for point in points:
            if current and _breaks(current[-1], point, max_jump_px, max_gap_seconds):
                kept.extend(_emit(track_id, segment_index, current, min_seconds))
                segment_index += 1
                current = []
            current.append(point)
        kept.extend(_emit(track_id, segment_index, current, min_seconds))
    kept.sort(key=lambda item: (item.points[0].timestamp, item.track_id, item.segment))
    return kept


def write_trajectories(path: Path, trajectories: list[Trajectory], *, min_seconds: float) -> None:
    payload = {
        "coordinate_space": "image",
        "min_seconds": min_seconds,
        "count": len(trajectories),
        "trajectories": [item.to_dict() for item in trajectories],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _breaks(
    previous: TrajectoryPoint,
    point: TrajectoryPoint,
    max_jump_px: float,
    max_gap_seconds: float,
) -> bool:
    if point.timestamp - previous.timestamp > max_gap_seconds:
        return True
    jump = ((point.x - previous.x) ** 2 + (point.y - previous.y) ** 2) ** 0.5
    return jump > max_jump_px


def _emit(
    track_id: int,
    segment: int,
    points: list[TrajectoryPoint],
    min_seconds: float,
) -> list[Trajectory]:
    if len(points) < 2:
        return []
    trajectory = Trajectory(track_id, segment, tuple(points))
    if trajectory.duration_seconds < min_seconds:
        return []
    return [trajectory]
