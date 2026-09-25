"""Tests for per-frame player pitch projection (no video / GPU)."""

from __future__ import annotations

from tactivision.calibration.project_players import (
    build_pitch_position_row,
    summarize_pitch_positions,
)


def _identity_like_h() -> list[list[float]]:
    # Homography that roughly maps pixels to pitch-scale values for unit test.
    return [[0.1, 0.0, 10.0], [0.0, 0.1, 20.0], [0.0, 0.0, 1.0]]


def test_invalid_calibration_yields_no_players():
    row = build_pitch_position_row(
        frame_id=0,
        timestamp=0.0,
        tracks=[{"track_id": 1, "bbox": [100, 100, 120, 160]}],
        calibration={"calibration_valid": False, "homography": None},
    )
    assert row["calibrated"] is False
    assert row["players"] == []
    assert row["calibration_source"] == "per_frame"


def test_missing_calibration_yields_no_players():
    row = build_pitch_position_row(
        frame_id=1,
        timestamp=0.04,
        tracks=[{"track_id": 2, "bbox": [100, 100, 120, 160]}],
        calibration=None,
    )
    assert row["calibrated"] is False
    assert row["players"] == []


def test_valid_calibration_projects_on_pitch_feet():
    cal = {
        "calibration_valid": True,
        "homography": _identity_like_h(),
        "reprojection_error": 1.2,
    }
    # Foot at bottom-center of bbox → ~ (11, 36) under this H.
    tracks = [{"track_id": 7, "bbox": [100.0, 200.0, 120.0, 360.0]}]
    row = build_pitch_position_row(frame_id=2, timestamp=0.08, tracks=tracks, calibration=cal)
    assert row["calibrated"] is True
    assert len(row["players"]) == 1
    assert row["players"][0]["track_id"] == 7
    assert row["reprojection_error"] == 1.2


def test_summarize_pitch_positions():
    rows = [
        build_pitch_position_row(
            frame_id=0,
            timestamp=0.0,
            tracks=[],
            calibration={"calibration_valid": False},
        ),
        {
            "frame_id": 1,
            "timestamp": 0.04,
            "calibrated": True,
            "reprojection_error": 1.0,
            "players": [{"track_id": 1, "x": 50.0, "y": 40.0}],
            "calibration_source": "per_frame",
        },
    ]
    summary = summarize_pitch_positions(rows)
    assert summary["calibrated_frames"] == 1
    assert summary["player_coordinate_samples"] == 1
    assert summary["interpolation"] is False
