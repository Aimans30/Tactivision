"""Tests for heuristic analytics (no GPU / no video required)."""

from __future__ import annotations

from tactivision.analytics.ball_metrics import summarize_ball
from tactivision.analytics.events import detect_events, summarize_events
from tactivision.analytics.formation import estimate_formation_at_frame
from tactivision.analytics.qa import answer_question
from tactivision.analytics.tactics import build_tactical_rows


def test_ball_summary_no_interpolation():
    rows = [
        {"frame": 0, "timestamp": 0.0, "pitch_valid": True, "pitch_x": 10.0, "pitch_y": 40.0},
        {"frame": 1, "timestamp": 0.04, "pitch_valid": False, "pitch_x": None, "pitch_y": None},
        {"frame": 2, "timestamp": 0.08, "pitch_valid": True, "pitch_x": 12.0, "pitch_y": 40.0},
    ]
    summary = summarize_ball(rows)
    assert summary["frames_with_pitch_ball"] == 2
    assert summary["segments"] == 2
    assert summary["missing_intervals"] == 1
    assert summary["ground_truth"] is False


def test_possession_change_and_pass_candidate():
    rows = [
        {
            "frame": 0,
            "timestamp": 0.0,
            "possession_status": "assigned",
            "possession_team": 0,
            "nearest_player_track_id": 1,
            "nearest_player_distance": 2.0,
            "ball_x": 50.0,
            "ball_y": 40.0,
        },
        {
            "frame": 1,
            "timestamp": 0.5,
            "possession_status": "assigned",
            "possession_team": 0,
            "nearest_player_track_id": 2,
            "nearest_player_distance": 2.0,
            "ball_x": 58.0,
            "ball_y": 40.0,
        },
        {
            "frame": 2,
            "timestamp": 1.0,
            "possession_status": "assigned",
            "possession_team": 1,
            "nearest_player_track_id": 9,
            "nearest_player_distance": 1.5,
            "ball_x": 60.0,
            "ball_y": 41.0,
        },
    ]
    events = detect_events(rows)
    types = {e["event_type"] for e in events}
    assert "pass_candidate" in types
    assert "possession_change" in types
    assert "ball_recovery" in types
    summary = summarize_events(events)
    assert summary["total_events"] >= 3
    assert summary["ground_truth"] is False


def test_formation_insufficient_players():
    estimate = estimate_formation_at_frame([(1, 0, 40.0, 20.0), (2, 0, 42.0, 30.0)])
    assert estimate["teams"]["0"]["valid"] is False


def test_formation_line_signature():
    players = []
    for i, x in enumerate([30, 32, 34, 50, 52, 54, 70, 72, 74]):
        players.append((i, 0, float(x), 40.0 + i))
    estimate = estimate_formation_at_frame(players)
    assert estimate["teams"]["0"]["valid"] is True
    assert estimate["teams"]["0"]["shape_signature"] is not None
    assert estimate["ground_truth"] is False


def test_tactics_ball_missing():
    poss = [{"frame": 0, "timestamp": 0.0, "possession_status": "unavailable", "possession_team": None, "ball_x": None, "ball_y": None}]
    rows = build_tactical_rows(poss, [])
    assert rows[0]["tactical_state"] == "ball_missing"


def test_qa_does_not_invent_when_empty_context():
    answer = answer_question("How did possession change?", {"possession_summary": {"possession_coverage_percent": 2.27, "team0_possession_percent_of_all_frames": 1.0, "team1_possession_percent_of_all_frames": 1.2, "unknown_percent_of_all_frames": 4.0, "possession_transitions": 1, "possession_radius_yards": 5.0, "debounce_frames": 2}})
    assert answer["ground_truth"] is False
    assert any("2.27" in f for f in answer["facts"])
