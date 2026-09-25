from tactivision.spatial.metrics import measure_track


def _point(frame: int, timestamp: float, x: float, y: float, valid: bool = True) -> dict:
    return {
        "track_id": 7,
        "frame": frame,
        "timestamp": timestamp,
        "x_raw": x + 2.0,
        "y_raw": y + 1.0,
        "x_smooth": x,
        "y_smooth": y,
        "calibration_error": 1.0,
        "calibration_valid": valid,
    }


def test_gap_longer_than_limit_is_not_counted() -> None:
    samples = [
        _point(0, 0.0, 10.0, 10.0),
        _point(12, 0.48, 13.0, 14.0),
        _point(200, 5.0, 80.0, 70.0),
        _point(212, 5.48, 82.0, 70.0) | {"x_raw": 90.0, "y_raw": 70.0},
    ]
    row = measure_track(samples)
    assert row is not None
    # 3-4-5 and 2, not the cut from (13, 14) to (80, 70)
    assert row["distance_yards"] == 7.0
    assert row["duration_seconds"] == 0.96
    assert row["valid_samples"] == 4
    assert row["distance_yards_raw"] != row["distance_yards"]
    assert row["metric_type"] == "estimated_model_derived"


def test_invalid_calibration_pair_is_skipped() -> None:
    samples = [
        _point(0, 0.0, 0.0, 0.0, valid=False),
        _point(12, 0.48, 10.0, 0.0),
    ]
    assert measure_track(samples) is None
