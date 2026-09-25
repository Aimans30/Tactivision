"""Team-level spatial shape from calibrated pitch positions.

Model-derived estimates only. Not ground-truth formation labels.
Track IDs are fragmented detections, not persistent player identities.
"""

from __future__ import annotations

import math
from collections import defaultdict

# Minimum players required for a valid team-shape sample at one timestamp.
# Width/length need at least two points; three reduces single-pair noise.
MIN_PLAYERS = 3
ESTIMATE_LABEL = "estimated_model_derived"
COORDINATE_SYSTEM = "statsbomb_120x80_yards"


def team_shape_at_timestamp(
    points: list[tuple[float, float]],
    *,
    min_players: int = MIN_PLAYERS,
) -> dict | None:
    """Compute centroid, width, length, compactness for one team at one time.

    ``points`` are ``(x, y)`` in StatsBomb yards. Returns ``None`` when fewer
    than ``min_players`` points are provided (no invented values).
    """
    n = len(points)
    if n < min_players:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) / n
    cy = sum(ys) / n
    width = max(xs) - min(xs)
    length = max(ys) - min(ys)
    compactness = sum(math.hypot(x - cx, y - cy) for x, y in points) / n
    return {
        "n_players": n,
        "centroid_x": round(cx, 3),
        "centroid_y": round(cy, 3),
        "width": round(width, 3),
        "length": round(length, 3),
        "compactness": round(compactness, 3),
    }


def build_team_shape_rows(
    trajectory_rows: list[dict],
    team_by_track: dict[int, int],
    *,
    min_players: int = MIN_PLAYERS,
) -> list[dict]:
    """One row per timestamp with per-team shape fields.

    Uses only samples with ``calibration_valid`` and smoothed coordinates.
    Does not interpolate across missing tracks or timestamps.
    """
    by_time: dict[tuple[int, float], dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in trajectory_rows:
        if not row.get("calibration_valid", False):
            continue
        track_id = int(row["track_id"])
        team = team_by_track.get(track_id)
        if team is None:
            continue
        x = row.get("x_smooth")
        y = row.get("y_smooth")
        if x is None or y is None:
            continue
        key = (int(row["frame"]), float(row["timestamp"]))
        by_time[key][int(team)].append((float(x), float(y)))

    rows: list[dict] = []
    for frame, timestamp in sorted(by_time.keys(), key=lambda item: (item[1], item[0])):
        teams = by_time[(frame, timestamp)]
        out: dict = {
            "frame": frame,
            "timestamp": round(timestamp, 3),
            "metric_type": ESTIMATE_LABEL,
            "coordinate_system": COORDINATE_SYSTEM,
            "coordinates": "smoothed",
            "min_players_required": min_players,
        }
        for team_id in (0, 1):
            prefix = f"team{team_id}_"
            points = teams.get(team_id, [])
            out[f"{prefix}n_available"] = len(points)
            shape = team_shape_at_timestamp(points, min_players=min_players)
            if shape is None:
                out[f"{prefix}valid"] = False
                out[f"{prefix}n_players"] = len(points)
                out[f"{prefix}centroid_x"] = ""
                out[f"{prefix}centroid_y"] = ""
                out[f"{prefix}width"] = ""
                out[f"{prefix}length"] = ""
                out[f"{prefix}compactness"] = ""
            else:
                out[f"{prefix}valid"] = True
                out[f"{prefix}n_players"] = shape["n_players"]
                out[f"{prefix}centroid_x"] = shape["centroid_x"]
                out[f"{prefix}centroid_y"] = shape["centroid_y"]
                out[f"{prefix}width"] = shape["width"]
                out[f"{prefix}length"] = shape["length"]
                out[f"{prefix}compactness"] = shape["compactness"]
        rows.append(out)
    return rows


def summarize_team_shape(rows: list[dict], team_by_track: dict[int, int]) -> dict:
    """Aggregate counts and ranges for valid team-shape samples."""
    team_ids = sorted(set(team_by_track.values()))
    per_team: dict[str, dict] = {}
    for team_id in (0, 1):
        prefix = f"team{team_id}_"
        valid_rows = [r for r in rows if r.get(f"{prefix}valid") is True]
        def _vals(key: str) -> list[float]:
            return [float(r[f"{prefix}{key}"]) for r in valid_rows]

        per_team[str(team_id)] = {
            "valid_samples": len(valid_rows),
            "invalid_or_insufficient_samples": sum(
                1 for r in rows if r.get(f"{prefix}valid") is not True
            ),
            "n_players": _stats(_vals("n_players")) if valid_rows else None,
            "centroid_x": _stats(_vals("centroid_x")) if valid_rows else None,
            "centroid_y": _stats(_vals("centroid_y")) if valid_rows else None,
            "width": _stats(_vals("width")) if valid_rows else None,
            "length": _stats(_vals("length")) if valid_rows else None,
            "compactness": _stats(_vals("compactness")) if valid_rows else None,
        }

    both_valid = sum(1 for r in rows if r.get("team0_valid") and r.get("team1_valid"))
    return {
        "metric_type": ESTIMATE_LABEL,
        "coordinate_system": COORDINATE_SYSTEM,
        "coordinates": "smoothed",
        "min_players_required": MIN_PLAYERS,
        "formulas": {
            "centroid": "mean(x), mean(y) over valid players on that team at the timestamp",
            "width": "max(x) - min(x)",
            "length": "max(y) - min(y)",
            "compactness": "mean Euclidean distance of players from the team centroid",
        },
        "axis_note": (
            "StatsBomb frame: x in [0, 120] along pitch length, y in [0, 80] along pitch width. "
            "This module follows the requested formulas (width on x, length on y)."
        ),
        "timestamps_total": len(rows),
        "timestamps_both_teams_valid": both_valid,
        "tracks_with_team_label": len(team_by_track),
        "team_label_counts": {
            str(t): sum(1 for v in team_by_track.values() if v == t) for t in team_ids
        },
        "teams": per_team,
        "ground_truth": False,
        "identity_note": (
            "Team labels come from kit-color clustering on filtered tracks. "
            "Track IDs are not persistent player identities; counts are not guaranteed to be 11."
        ),
    }


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "minimum": round(float(ordered[0]), 3),
        "median": round(float(median), 3),
        "maximum": round(float(ordered[-1]), 3),
        "mean": round(float(sum(ordered) / len(ordered)), 3),
    }
