"""Interpretable tactical-state labels from possession + team shape + ball.

States are measurable feature thresholds, not coach annotations.
"""

from __future__ import annotations

from tactivision.analytics.io import coerce_float, coerce_int
from tactivision.analytics.schema import COORDINATE_SYSTEM, HEURISTIC


def build_tactical_rows(
    possession_rows: list[dict],
    team_shape_rows: list[dict],
    *,
    compact_threshold: float = 12.0,
    expanded_threshold: float = 18.0,
    attacking_x_team0: float = 70.0,
    attacking_x_team1: float = 50.0,
) -> list[dict]:
    """One tactical row per possession frame that can be joined to shape.

    Team attacking directions are unknown on this clip; attacking/defensive
    labels use ball x relative to pitch thirds as a territorial proxy only.
    """
    shape_by_frame = {int(r["frame"]): r for r in team_shape_rows}
    rows: list[dict] = []
    prev_team: int | None = None

    for prow in sorted(possession_rows, key=lambda r: (int(r["frame"]), float(r["timestamp"]))):
        frame = int(prow["frame"])
        status = prow.get("possession_status")
        team = coerce_int(prow.get("possession_team")) if status == "assigned" else None
        ball_x = coerce_float(prow.get("ball_x"))
        ball_y = coerce_float(prow.get("ball_y"))
        shape = shape_by_frame.get(frame)

        state = "insufficient_data"
        features: dict = {
            "possession_status": status,
            "possession_team": team,
            "ball_x": ball_x,
            "ball_y": ball_y,
        }
        confidence = 0.2

        if status == "unavailable" or ball_x is None:
            state = "ball_missing"
            confidence = 0.9
        elif status == "unknown":
            state = "contested_or_loose"
            confidence = 0.45
        elif team is not None:
            if prev_team is not None and team != prev_team:
                state = "transition"
                confidence = 0.65
            else:
                # Territorial proxy by pitch thirds.
                if ball_x >= 80:
                    state = "attacking_territory"
                elif ball_x <= 40:
                    state = "defensive_territory"
                else:
                    state = "possession_midfield"
                confidence = 0.5

            if shape is not None:
                prefix = f"team{team}_"
                compact = coerce_float(shape.get(f"{prefix}compactness"))
                width = coerce_float(shape.get(f"{prefix}width"))
                features["team_compactness"] = compact
                features["team_width"] = width
                features["team_n_players"] = coerce_int(shape.get(f"{prefix}n_players"))
                if compact is not None:
                    if compact <= compact_threshold and state != "transition":
                        state = f"{state}|compact"
                        confidence = min(0.75, confidence + 0.1)
                    elif compact >= expanded_threshold and state != "transition":
                        state = f"{state}|expanded"
                        confidence = min(0.75, confidence + 0.1)

        # Lateral bias from ball y (StatsBomb: 0=touchline, 80=opposite).
        lateral = "central"
        if ball_y is not None:
            if ball_y < 26.5:
                lateral = "left_side"
            elif ball_y > 53.5:
                lateral = "right_side"
        features["lateral_bias"] = lateral

        rows.append(
            {
                "frame": frame,
                "timestamp": round(float(prow["timestamp"]), 3),
                "tactical_state": state,
                "lateral_bias": lateral,
                "possession_team": team,
                "confidence": round(confidence, 3),
                "features": features,
                "provenance": HEURISTIC,
                "ground_truth": False,
                "coordinate_system": COORDINATE_SYSTEM,
                "note": (
                    "Heuristic labels from possession status, ball pitch thirds, "
                    "and optional team compactness. Not coach/tactical ground truth."
                ),
            }
        )
        if team is not None:
            prev_team = team

    return rows


def summarize_tactics(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    lateral: dict[str, int] = {}
    for row in rows:
        counts[row["tactical_state"]] = counts.get(row["tactical_state"], 0) + 1
        lateral[row["lateral_bias"]] = lateral.get(row["lateral_bias"], 0) + 1
    return {
        "metric_type": HEURISTIC,
        "provenance": HEURISTIC,
        "ground_truth": False,
        "frames": len(rows),
        "state_counts": counts,
        "lateral_bias_counts": lateral,
        "note": "Counts cover possession timeline frames, including unavailable/unknown.",
    }
