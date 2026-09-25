"""Possession heuristic contracts (synthetic fixtures)."""

from __future__ import annotations

from tactivision.spatial.possession import (
    POSSESSION_RADIUS_YARDS,
    build_possession_rows,
    raw_possession,
    summarize_possession,
)


def test_nearby_player_assigns_team():
    result = raw_possession(
        ball_xy=(50.0, 40.0),
        players=[(7, 1, 51.0, 40.0), (3, 0, 80.0, 40.0)],
        radius=5.0,
    )
    assert result["possession_status"] == "assigned"
    assert result["possession_team"] == 1
    assert result["nearest_player_track_id"] == 7
    assert result["nearest_player_distance"] == 1.0


def test_beyond_radius_is_unknown_not_nearest_team():
    result = raw_possession(
        ball_xy=(50.0, 40.0),
        players=[(7, 1, 60.0, 40.0)],  # 10 yd > 5
        radius=5.0,
    )
    assert result["possession_status"] == "unknown"
    assert result["possession_team"] is None
    assert result["nearest_player_team"] == 1


def test_missing_ball_is_unavailable():
    result = raw_possession(ball_xy=None, players=[(1, 0, 50.0, 40.0)])
    assert result["possession_status"] == "unavailable"
    assert result["possession_team"] is None


def test_summary_percentages_bounded_and_not_ground_truth():
    ball_rows = [
        {"frame": 0, "timestamp": 0.0, "pitch_valid": True, "pitch_x": 50.0, "pitch_y": 40.0},
        {"frame": 1, "timestamp": 0.04, "pitch_valid": True, "pitch_x": 50.5, "pitch_y": 40.0},
        {"frame": 2, "timestamp": 0.08, "pitch_valid": False, "pitch_x": None, "pitch_y": None},
        {"frame": 3, "timestamp": 0.12, "pitch_valid": True, "pitch_x": 90.0, "pitch_y": 40.0},
    ]
    players_by_frame = {
        0: [(1, 0, 50.2, 40.0)],
        1: [(1, 0, 50.3, 40.0)],
        3: [(9, 1, 99.0, 40.0)],  # far → unknown
    }
    rows = build_possession_rows(
        ball_rows=ball_rows,
        players_by_frame=players_by_frame,
        radius=POSSESSION_RADIUS_YARDS,
        debounce_frames=2,
    )
    summary = summarize_possession(rows, radius=POSSESSION_RADIUS_YARDS, debounce_frames=2)
    assert summary["ground_truth"] is False
    assert summary["method"].startswith("nearest_player")
    total_pct = (
        summary["team0_possession_percent_of_all_frames"]
        + summary["team1_possession_percent_of_all_frames"]
        + summary["unknown_percent_of_all_frames"]
        + summary["unavailable_percent_of_all_frames"]
    )
    assert abs(total_pct - 100.0) < 0.05
    assert 0.0 <= summary["possession_coverage_percent"] <= 100.0
    assert summary["frames_assigned"] + summary["frames_unknown"] + summary["frames_unavailable"] == 4
