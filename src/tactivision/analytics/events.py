"""Heuristic event candidates from possession + ball + nearest-player links.

Not ground-truth match events. Each event stores evidence features and confidence.
"""

from __future__ import annotations

import math
from typing import Any

from tactivision.analytics.io import coerce_float, coerce_int
from tactivision.analytics.schema import COORDINATE_SYSTEM, HEURISTIC


def _assigned(row: dict) -> bool:
    return row.get("possession_status") == "assigned" and row.get("possession_team") is not None


def detect_events(
    possession_rows: list[dict],
    *,
    min_pass_displacement_yards: float = 4.0,
    min_carry_frames: int = 3,
    min_carry_displacement_yards: float = 3.0,
    progression_yards: float = 8.0,
) -> list[dict]:
    """Emit ordered heuristic event candidates.

    Pass / carry / progression only fire on assigned-possession frames with
    valid ball coordinates. Possession changes use the assigned timeline.
    """
    rows = sorted(possession_rows, key=lambda r: (int(r["frame"]), float(r["timestamp"])))
    events: list[dict] = []
    prev_assigned: dict | None = None
    carry_start: dict | None = None
    progress_anchor: dict | None = None

    for row in rows:
        if not _assigned(row):
            # Close open carry when possession breaks.
            if carry_start is not None:
                _maybe_emit_carry(events, carry_start, prev_assigned, min_carry_frames, min_carry_displacement_yards)
                carry_start = None
            progress_anchor = None
            continue

        team = int(row["possession_team"])
        track = coerce_int(row.get("nearest_player_track_id"))
        bx = coerce_float(row.get("ball_x"))
        by = coerce_float(row.get("ball_y"))
        if bx is None or by is None or track is None:
            continue

        if prev_assigned is not None:
            prev_team = int(prev_assigned["possession_team"])
            if team != prev_team:
                events.append(
                    _event(
                        "possession_change",
                        row,
                        team=team,
                        player_track_id=track,
                        confidence=0.7,
                        evidence={
                            "from_team": prev_team,
                            "to_team": team,
                            "nearest_distance": coerce_float(row.get("nearest_player_distance")),
                        },
                    )
                )
                events.append(
                    _event(
                        "ball_recovery",
                        row,
                        team=team,
                        player_track_id=track,
                        confidence=0.55,
                        evidence={
                            "prior_team": prev_team,
                            "nearest_distance": coerce_float(row.get("nearest_player_distance")),
                        },
                    )
                )
                carry_start = row
                progress_anchor = row
            else:
                prev_track = coerce_int(prev_assigned.get("nearest_player_track_id"))
                pbx = coerce_float(prev_assigned.get("ball_x"))
                pby = coerce_float(prev_assigned.get("ball_y"))
                if (
                    prev_track is not None
                    and track != prev_track
                    and pbx is not None
                    and pby is not None
                ):
                    displacement = math.hypot(bx - pbx, by - pby)
                    if displacement >= min_pass_displacement_yards:
                        events.append(
                            _event(
                                "pass_candidate",
                                row,
                                team=team,
                                player_track_id=track,
                                confidence=min(0.85, 0.4 + displacement / 40.0),
                                evidence={
                                    "from_track_id": prev_track,
                                    "to_track_id": track,
                                    "ball_displacement_yards": round(displacement, 3),
                                    "from_ball": [pbx, pby],
                                    "to_ball": [bx, by],
                                },
                            )
                        )
                        carry_start = row
                        progress_anchor = row
                if track == coerce_int((carry_start or row).get("nearest_player_track_id")):
                    if carry_start is None:
                        carry_start = row
                else:
                    if carry_start is not None:
                        _maybe_emit_carry(
                            events, carry_start, prev_assigned, min_carry_frames, min_carry_displacement_yards
                        )
                    carry_start = row

                if progress_anchor is None:
                    progress_anchor = row
                else:
                    ax = coerce_float(progress_anchor.get("ball_x"))
                    if ax is not None:
                        # StatsBomb: team attacking +x is unknown; use absolute
                        # progression along length as territorial change.
                        delta_x = bx - ax
                        if abs(delta_x) >= progression_yards:
                            events.append(
                                _event(
                                    "ball_progression",
                                    row,
                                    team=team,
                                    player_track_id=track,
                                    confidence=0.5,
                                    evidence={
                                        "delta_x_yards": round(delta_x, 3),
                                        "from_x": ax,
                                        "to_x": bx,
                                        "window_seconds": round(
                                            float(row["timestamp"]) - float(progress_anchor["timestamp"]),
                                            3,
                                        ),
                                    },
                                )
                            )
                            progress_anchor = row
        else:
            carry_start = row
            progress_anchor = row

        prev_assigned = row

    if carry_start is not None and prev_assigned is not None:
        _maybe_emit_carry(events, carry_start, prev_assigned, min_carry_frames, min_carry_displacement_yards)

    events.sort(key=lambda e: (e["frame"], e["timestamp"], e["event_type"]))
    return events


def _maybe_emit_carry(
    events: list[dict],
    start: dict,
    end: dict | None,
    min_frames: int,
    min_disp: float,
) -> None:
    if end is None:
        return
    frames = int(end["frame"]) - int(start["frame"]) + 1
    if frames < min_frames:
        return
    sx = coerce_float(start.get("ball_x"))
    sy = coerce_float(start.get("ball_y"))
    ex = coerce_float(end.get("ball_x"))
    ey = coerce_float(end.get("ball_y"))
    if None in (sx, sy, ex, ey):
        return
    disp = math.hypot(ex - sx, ey - sy)
    if disp < min_disp:
        return
    track = coerce_int(start.get("nearest_player_track_id"))
    events.append(
        _event(
            "carry_candidate",
            end,
            team=int(start["possession_team"]),
            player_track_id=track,
            confidence=min(0.8, 0.35 + disp / 30.0),
            evidence={
                "start_frame": int(start["frame"]),
                "end_frame": int(end["frame"]),
                "duration_seconds": round(float(end["timestamp"]) - float(start["timestamp"]), 3),
                "displacement_yards": round(disp, 3),
                "track_id": track,
            },
        )
    )


def _event(
    event_type: str,
    row: dict,
    *,
    team: int | None,
    player_track_id: int | None,
    confidence: float,
    evidence: dict[str, Any],
) -> dict:
    return {
        "frame": int(row["frame"]),
        "timestamp": round(float(row["timestamp"]), 3),
        "event_type": event_type,
        "team": team,
        "player_track_id": player_track_id,
        "confidence": round(confidence, 3),
        "evidence": evidence,
        "provenance": HEURISTIC,
        "ground_truth": False,
        "coordinate_system": COORDINATE_SYSTEM,
        "note": "Heuristic candidate from proximity possession + ball motion; not official event data.",
    }


def summarize_events(events: list[dict]) -> dict:
    by_type: dict[str, int] = {}
    for event in events:
        by_type[event["event_type"]] = by_type.get(event["event_type"], 0) + 1
    return {
        "metric_type": HEURISTIC,
        "provenance": HEURISTIC,
        "ground_truth": False,
        "total_events": len(events),
        "by_type": by_type,
        "note": (
            "Pass/carry/progression/recovery are proximity heuristics on sparse "
            "player–ball joins. Do not treat counts as official match events."
        ),
    }
