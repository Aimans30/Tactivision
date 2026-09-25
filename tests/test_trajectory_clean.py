from tactivision.spatial.clean import (
    MAX_GAP_SECONDS,
    MAX_SPEED_YARDS_PER_SECOND,
    PitchSample,
    reject_jumps,
    smooth_track,
)


def _sample(frame: int, timestamp: float, x: float, y: float) -> PitchSample:
    return PitchSample(7, frame, timestamp, x, y, 1.0, True)


def test_impossible_jump_is_removed() -> None:
    samples = [
        _sample(0, 0.0, 40.0, 40.0),
        _sample(12, 0.48, 41.0, 40.2),
        _sample(24, 0.96, 100.0, 70.0),
    ]
    kept, removed = reject_jumps(samples)
    assert removed == 1
    assert [sample.frame for sample in kept] == [0, 12]
    assert MAX_SPEED_YARDS_PER_SECOND == 12.0


def test_smooth_keeps_raw_and_does_not_cross_a_gap() -> None:
    first = [_sample(index * 12, index * 0.48, 20.0 + index, 30.0) for index in range(5)]
    later = [_sample(200 + index * 12, 10.0 + index * 0.48, 80.0, 10.0 + index) for index in range(5)]
    rows = smooth_track(first + later)
    assert len(rows) == 10
    assert all(row["calibration_valid"] for row in rows)
    assert rows[0]["x_raw"] == 20.0
    assert abs(rows[2]["x_smooth"] - rows[2]["x_raw"]) < 1.0
    assert rows[5]["x_raw"] == 80.0
    assert rows[5]["x_smooth"] < 85.0
    assert MAX_GAP_SECONDS < 1.0
