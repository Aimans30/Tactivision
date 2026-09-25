"""Proximity-based possession heuristic on StatsBomb pitch coordinates.

Model-derived estimate only. Not ground-truth match possession.
"""

from __future__ import annotations

import math
from collections import defaultdict

# Control-radius heuristic in StatsBomb yards (~4.6 m).
# Larger than a single stride (~2 yd) so brief on-ball contact can register;
# below the median nearest-player distance on this clip's joinable frames
# (~6.9 yd) so distant "nearest" players are not auto-assigned possession.
POSSESSION_RADIUS_YARDS = 5.0

# Debounce: a team switch is accepted only after this many consecutive
# raw assignments to the new team (evaluated frames only). Single-frame
# flips are held as the previous assigned team when still within radius
# for the old team is NOT re-checked; instead the flicker frame is marked
# unknown until the new team repeats. See apply_debounce().
DEBOUNCE_FRAMES = 2

ESTIMATE_LABEL = "estimated_model_derived"
COORDINATE_SYSTEM = "statsbomb_120x80_yards"


def nearest_player(
    ball_xy: tuple[float, float],
    players: list[tuple[int, int, float, float]],
) -> tuple[int, int, float] | None:
    """Return (track_id, team, distance) for the nearest player, or None."""
    if not players:
        return None
    bx, by = ball_xy
    best: tuple[int, int, float] | None = None
    for track_id, team, x, y in players:
        dist = math.hypot(bx - x, by - y)
        if best is None or dist < best[2]:
            best = (track_id, team, dist)
    return best


def raw_possession(
    *,
    ball_xy: tuple[float, float] | None,
    players: list[tuple[int, int, float, float]],
    radius: float = POSSESSION_RADIUS_YARDS,
) -> dict:
    """Assign possession for one frame without temporal smoothing."""
    if ball_xy is None:
        return {
            "ball_x": None,
            "ball_y": None,
            "nearest_player_track_id": None,
            "nearest_player_team": None,
            "nearest_player_distance": None,
            "possession_team": None,
            "possession_status": "unavailable",
            "unavailable_reason": "no_ball_pitch",
            "raw_possession_team": None,
        }
    nearest = nearest_player(ball_xy, players)
    if nearest is None:
        return {
            "ball_x": round(ball_xy[0], 3),
            "ball_y": round(ball_xy[1], 3),
            "nearest_player_track_id": None,
            "nearest_player_team": None,
            "nearest_player_distance": None,
            "possession_team": None,
            "possession_status": "unavailable",
            "unavailable_reason": "no_players",
            "raw_possession_team": None,
        }
    track_id, team, dist = nearest
    if dist <= radius:
        status = "assigned"
        poss_team: int | None = team
    else:
        status = "unknown"
        poss_team = None
    return {
        "ball_x": round(ball_xy[0], 3),
        "ball_y": round(ball_xy[1], 3),
        "nearest_player_track_id": track_id,
        "nearest_player_team": team,
        "nearest_player_distance": round(dist, 3),
        "possession_team": poss_team,
        "possession_status": status,
        "unavailable_reason": None,
        "raw_possession_team": poss_team,
    }


def apply_debounce(
    rows: list[dict],
    *,
    debounce_frames: int = DEBOUNCE_FRAMES,
) -> list[dict]:
    """Reduce single-frame team flicker among assigned frames.

    Rule (evaluated in chronological order):
    - ``unavailable`` frames are left unchanged (no possession claim).
    - ``unknown`` raw frames (nearest player beyond radius) stay ``unknown``.
    - When raw status is ``assigned`` to team T:
      - if T equals the current held team, keep assigned to T and reset the
        opposing streak;
      - if T differs, require ``debounce_frames`` consecutive raw assignments
        to T before switching; meanwhile emit ``unknown`` (do not invent a
        hold of the old team across a clear distance-based unknown).
    """
    held: int | None = None
    streak_team: int | None = None
    streak = 0
    out: list[dict] = []
    for row in rows:
        row = dict(row)
        raw_team = row.get("raw_possession_team")
        status = row["possession_status"]

        if status == "unavailable":
            out.append(row)
            continue

        if status == "unknown" or raw_team is None:
            streak_team = None
            streak = 0
            row["possession_team"] = None
            row["possession_status"] = "unknown"
            row["debounce_note"] = "beyond_radius_or_cleared"
            out.append(row)
            continue

        # assigned raw
        if held is None:
            held = int(raw_team)
            streak_team = held
            streak = 1
            row["possession_team"] = held
            row["possession_status"] = "assigned"
            row["debounce_note"] = "initial_assign"
            out.append(row)
            continue

        if int(raw_team) == held:
            streak_team = held
            streak = 1
            row["possession_team"] = held
            row["possession_status"] = "assigned"
            row["debounce_note"] = "hold"
            out.append(row)
            continue

        # candidate switch
        if streak_team == int(raw_team):
            streak += 1
        else:
            streak_team = int(raw_team)
            streak = 1

        if streak >= debounce_frames:
            held = int(raw_team)
            row["possession_team"] = held
            row["possession_status"] = "assigned"
            row["debounce_note"] = "switch_accepted"
        else:
            row["possession_team"] = None
            row["possession_status"] = "unknown"
            row["debounce_note"] = "switch_pending"
        out.append(row)
    return out


def build_possession_rows(
    *,
    ball_rows: list[dict],
    players_by_frame: dict[int, list[tuple[int, int, float, float]]],
    radius: float = POSSESSION_RADIUS_YARDS,
    debounce_frames: int = DEBOUNCE_FRAMES,
) -> list[dict]:
    """One row per ball timeline frame."""
    raw_rows: list[dict] = []
    for ball in sorted(ball_rows, key=lambda r: (r.get("timestamp", 0), r.get("frame", 0))):
        frame = int(ball["frame"])
        timestamp = float(ball["timestamp"])
        ball_xy = None
        if ball.get("pitch_valid") and ball.get("pitch_x") is not None and ball.get("pitch_y") is not None:
            ball_xy = (float(ball["pitch_x"]), float(ball["pitch_y"]))
        players = players_by_frame.get(frame, [])
        base = raw_possession(ball_xy=ball_xy, players=players, radius=radius)
        raw_rows.append(
            {
                "frame": frame,
                "timestamp": round(timestamp, 3),
                "metric_type": ESTIMATE_LABEL,
                "coordinate_system": COORDINATE_SYSTEM,
                "possession_radius_yards": radius,
                **base,
            }
        )
    return apply_debounce(raw_rows, debounce_frames=debounce_frames)


def summarize_possession(rows: list[dict], *, radius: float, debounce_frames: int) -> dict:
    total = len(rows)
    assigned = [r for r in rows if r["possession_status"] == "assigned"]
    unknown = [r for r in rows if r["possession_status"] == "unknown"]
    unavailable = [r for r in rows if r["possession_status"] == "unavailable"]
    team0 = [r for r in assigned if r["possession_team"] == 0]
    team1 = [r for r in assigned if r["possession_team"] == 1]

    # Transitions: change of assigned team relative to the previous assigned
    # frame in time order (unavailable/unknown gaps do not erase the last team).
    transitions = 0
    prev: int | None = None
    for r in rows:
        if r["possession_status"] != "assigned":
            continue
        team = int(r["possession_team"])
        if prev is not None and team != prev:
            transitions += 1
        prev = team

    dists = [
        float(r["nearest_player_distance"])
        for r in assigned
        if r.get("nearest_player_distance") is not None
    ]
    dists_sorted = sorted(dists)

    def pct(n: int) -> float:
        return round(100.0 * n / total, 2) if total else 0.0

    return {
        "metric_type": ESTIMATE_LABEL,
        "coordinate_system": COORDINATE_SYSTEM,
        "ground_truth": False,
        "method": "nearest_player_within_radius_with_debounce",
        "possession_radius_yards": radius,
        "debounce_frames": debounce_frames,
        "total_frames": total,
        "frames_assigned": len(assigned),
        "frames_unknown": len(unknown),
        "frames_unavailable": len(unavailable),
        "possession_coverage_percent": pct(len(assigned)),
        "team0_possession_percent_of_all_frames": pct(len(team0)),
        "team1_possession_percent_of_all_frames": pct(len(team1)),
        "unknown_percent_of_all_frames": pct(len(unknown)),
        "unavailable_percent_of_all_frames": pct(len(unavailable)),
        "team0_share_of_assigned_percent": (
            round(100.0 * len(team0) / len(assigned), 2) if assigned else 0.0
        ),
        "team1_share_of_assigned_percent": (
            round(100.0 * len(team1) / len(assigned), 2) if assigned else 0.0
        ),
        "possession_transitions": transitions,
        "median_nearest_player_distance_when_assigned": (
            None if not dists_sorted else round(dists_sorted[len(dists_sorted) // 2], 3)
        ),
        "radius_justification": (
            "5.0 StatsBomb yards (~4.6 m): larger than a stride-scale control bubble, "
            "smaller than this clip's median nearest-player distance on joinable frames "
            "so distant nearest players remain unknown."
        ),
    }


def index_players(
    trajectory_rows: list[dict],
    team_by_track: dict[int, int],
) -> dict[int, list[tuple[int, int, float, float]]]:
    by_frame: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
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
        by_frame[int(row["frame"])].append((track_id, int(team), float(x), float(y)))
    return by_frame
