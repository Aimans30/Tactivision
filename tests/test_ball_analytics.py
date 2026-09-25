"""Unit tests for ball coverage / segment / speed contracts (synthetic fixtures)."""

from __future__ import annotations

from tactivision.analytics.ball_metrics import (
    ball_segments,
    ball_speed_samples,
    missing_intervals,
    summarize_ball,
)


def _row(frame: int, ts: float, *, valid: bool, x: float | None = None, y: float | None = None) -> dict:
    return {
        "frame": frame,
        "timestamp": ts,
        "pitch_valid": valid,
        "pitch_x": x,
        "pitch_y": y,
    }


def test_observed_vs_missing_and_segments():
    rows = [
        _row(0, 0.00, valid=True, x=10.0, y=40.0),
        _row(1, 0.04, valid=True, x=11.0, y=40.0),
        _row(2, 0.08, valid=False),
        _row(3, 0.12, valid=True, x=12.0, y=41.0),
    ]
    segments = ball_segments(rows)
    gaps = missing_intervals(rows)
    assert len(segments) == 2
    assert segments[0]["frames"] == 2
    assert segments[1]["frames"] == 1
    assert len(gaps) == 1
    assert gaps[0]["frames"] == 1


def test_coverage_and_speed_on_contiguous_samples():
    rows = [
        _row(0, 0.00, valid=True, x=0.0, y=0.0),
        _row(1, 0.04, valid=True, x=4.0, y=0.0),  # 100 yd/s
        _row(2, 0.08, valid=False),
    ]
    summary = summarize_ball(rows, fps=25.0)
    assert summary["total_frames"] == 3
    assert summary["frames_with_pitch_ball"] == 2
    assert summary["pitch_coverage_percent"] == 66.67
    assert 0.0 <= summary["pitch_coverage_percent"] <= 100.0
    speeds = ball_speed_samples(rows, max_gap_seconds=0.12)
    assert len(speeds) == 1
    assert speeds[0]["speed_yards_per_second"] == 100.0
    assert speeds[0]["displacement_yards"] == 4.0
    assert summary["max_speed_yards_per_second"] == 100.0
    assert summary["ground_truth"] is False


def test_invalid_pitch_rows_do_not_contribute_speed():
    rows = [
        _row(0, 0.00, valid=False),
        _row(1, 0.04, valid=False),
        _row(2, 0.08, valid=True, x=1.0, y=1.0),
    ]
    assert ball_speed_samples(rows) == []
    summary = summarize_ball(rows)
    assert summary["speed_samples"] == 0
    assert summary["median_speed_yards_per_second"] is None
