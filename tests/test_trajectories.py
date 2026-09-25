import json

from tactivision.spatial.trajectories import build_trajectories, foot_point


def _row(frame_id: int, timestamp: float, tracks: list[dict]) -> dict:
    return {"frame_id": frame_id, "timestamp": timestamp, "tracks": tracks}


def _player(track_id: int, bbox: list[float]) -> dict:
    return {"track_id": track_id, "class": "player", "confidence": 0.9, "bbox": bbox}


def test_foot_point_is_bottom_center() -> None:
    assert foot_point([10, 20, 30, 80]) == (20.0, 80.0)


def test_short_and_jumping_tracks_are_dropped_or_split() -> None:
    rows = []
    # A steady 3 second walk, one pixel per frame.
    for index in range(76):
        rows.append(_row(index, index / 25, [_player(7, [100 + index, 200, 120 + index, 280])]))
    # A flicker shorter than 2 seconds.
    for index in range(10):
        rows.append(_row(index, index / 25, [_player(8, [400, 200, 420, 280])]))
    # A cut: same ID teleports after 1 second, then continues for 3 seconds.
    for index in range(25):
        rows.append(_row(200 + index, 8 + index / 25, [_player(9, [50, 300, 70, 360])]))
    for index in range(80):
        rows.append(_row(300 + index, 12 + index / 25, [_player(9, [900, 100, 920, 160])]))

    kept = build_trajectories(rows, min_seconds=2.0, max_jump_px=120.0)
    by_id = {}
    for item in kept:
        by_id.setdefault(item.track_id, []).append(item)

    assert 7 in by_id
    assert 8 not in by_id
    assert len(by_id[9]) == 1
    assert by_id[9][0].points[0].timestamp >= 12
    payload = json.loads(json.dumps(kept[0].to_dict()))
    assert payload["track_id"] == kept[0].track_id
    assert payload["points"][0]["y"] == 280.0
