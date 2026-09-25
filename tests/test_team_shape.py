"""Team-shape spatial contracts (synthetic fixtures)."""

from __future__ import annotations

from tactivision.spatial.team_shape import MIN_PLAYERS, build_team_shape_rows, team_shape_at_timestamp


def test_centroid_width_length():
    points = [(10.0, 20.0), (30.0, 20.0), (20.0, 40.0)]
    shape = team_shape_at_timestamp(points, min_players=3)
    assert shape is not None
    assert shape["n_players"] == 3
    assert shape["centroid_x"] == 20.0
    assert shape["centroid_y"] == round((20.0 + 20.0 + 40.0) / 3.0, 3)
    assert shape["width"] == 20.0
    assert shape["length"] == 20.0
    assert shape["compactness"] > 0


def test_insufficient_players_returns_none():
    assert team_shape_at_timestamp([(0.0, 0.0), (1.0, 1.0)], min_players=MIN_PLAYERS) is None


def test_build_rows_marks_invalid_without_malformed_numbers():
    traj = [
        {"track_id": 1, "frame": 0, "timestamp": 0.0, "x_smooth": 40.0, "y_smooth": 30.0, "calibration_valid": True},
        {"track_id": 2, "frame": 0, "timestamp": 0.0, "x_smooth": 42.0, "y_smooth": 32.0, "calibration_valid": True},
    ]
    rows = build_team_shape_rows(traj, {1: 0, 2: 0}, min_players=3)
    assert len(rows) == 1
    assert rows[0]["team0_valid"] is False
    assert rows[0]["team0_centroid_x"] == ""
    assert rows[0]["team0_n_available"] == 2
