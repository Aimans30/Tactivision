"""Ball movement analytics from per-frame pitch observations.

Does not interpolate missing ball positions.
"""

from __future__ import annotations

import math

from tactivision.analytics.schema import COORDINATE_SYSTEM, ESTIMATE, HEURISTIC


def _valid_pitch(row: dict) -> bool:
    return bool(row.get("pitch_valid")) and row.get("pitch_x") is not None and row.get("pitch_y") is not None


def ball_segments(rows: list[dict]) -> list[dict]:
    """Continuous pitch-valid runs (gaps break segments; no fill)."""
    segments: list[dict] = []
    current: dict | None = None
    prev_frame: int | None = None
    for row in sorted(rows, key=lambda r: (r.get("frame", 0), r.get("timestamp", 0))):
        frame = int(row["frame"])
        if not _valid_pitch(row):
            if current is not None:
                segments.append(current)
                current = None
            prev_frame = frame
            continue
        contiguous = prev_frame is not None and frame == prev_frame + 1 and current is not None
        if not contiguous:
            if current is not None:
                segments.append(current)
            current = {
                "start_frame": frame,
                "end_frame": frame,
                "start_time": float(row["timestamp"]),
                "end_time": float(row["timestamp"]),
                "frames": 1,
            }
        else:
            assert current is not None
            current["end_frame"] = frame
            current["end_time"] = float(row["timestamp"])
            current["frames"] += 1
        prev_frame = frame
    if current is not None:
        segments.append(current)
    return segments


def missing_intervals(rows: list[dict]) -> list[dict]:
    gaps: list[dict] = []
    current: dict | None = None
    for row in sorted(rows, key=lambda r: (r.get("frame", 0), r.get("timestamp", 0))):
        frame = int(row["frame"])
        ts = float(row["timestamp"])
        missing = not _valid_pitch(row)
        if missing:
            if current is None:
                current = {
                    "start_frame": frame,
                    "end_frame": frame,
                    "start_time": ts,
                    "end_time": ts,
                    "frames": 1,
                }
            else:
                current["end_frame"] = frame
                current["end_time"] = ts
                current["frames"] += 1
        elif current is not None:
            gaps.append(current)
            current = None
    if current is not None:
        gaps.append(current)
    return gaps


def ball_speed_samples(rows: list[dict], *, max_gap_seconds: float = 0.12) -> list[dict]:
    """Instantaneous speeds between consecutive pitch-valid observations."""
    ordered = [r for r in sorted(rows, key=lambda r: (r["frame"], r["timestamp"])) if _valid_pitch(r)]
    samples: list[dict] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        dt = float(right["timestamp"]) - float(left["timestamp"])
        if dt <= 0 or dt > max_gap_seconds:
            continue
        dx = float(right["pitch_x"]) - float(left["pitch_x"])
        dy = float(right["pitch_y"]) - float(left["pitch_y"])
        dist = math.hypot(dx, dy)
        samples.append(
            {
                "frame": int(right["frame"]),
                "timestamp": round(float(right["timestamp"]), 3),
                "speed_yards_per_second": round(dist / dt, 3),
                "displacement_yards": round(dist, 3),
                "dt_seconds": round(dt, 4),
                "provenance": HEURISTIC,
                "metric_type": ESTIMATE,
                "coordinate_system": COORDINATE_SYSTEM,
            }
        )
    return samples


def summarize_ball(rows: list[dict], *, fps: float = 25.0) -> dict:
    valid = [r for r in rows if _valid_pitch(r)]
    segments = ball_segments(rows)
    gaps = missing_intervals(rows)
    speeds = ball_speed_samples(rows)
    speed_vals = [s["speed_yards_per_second"] for s in speeds]
    longest = max(segments, key=lambda s: s["frames"], default=None)
    return {
        "metric_type": ESTIMATE,
        "provenance": ESTIMATE,
        "ground_truth": False,
        "coordinate_system": COORDINATE_SYSTEM,
        "total_frames": len(rows),
        "frames_with_pitch_ball": len(valid),
        "pitch_coverage_percent": round(100.0 * len(valid) / len(rows), 2) if rows else 0.0,
        "segments": len(segments),
        "longest_segment_frames": None if longest is None else longest["frames"],
        "longest_segment_seconds": (
            None if longest is None else round(longest["frames"] / fps, 3)
        ),
        "missing_intervals": len(gaps),
        "missing_frames": sum(g["frames"] for g in gaps),
        "speed_samples": len(speeds),
        "median_speed_yards_per_second": (
            None
            if not speed_vals
            else round(sorted(speed_vals)[len(speed_vals) // 2], 3)
        ),
        "max_speed_yards_per_second": None if not speed_vals else round(max(speed_vals), 3),
        "note": (
            "Speeds use consecutive pitch-valid frames only. "
            "Missing detections are not interpolated."
        ),
        "segment_table": segments,
        "gap_table": gaps,
    }
