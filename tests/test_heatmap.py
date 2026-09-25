import json
from pathlib import Path

import cv2

from tactivision.spatial.heatmap import smoothed_points


def test_heatmap_uses_smoothed_valid_points_only() -> None:
    rows = [
        {
            "track_id": 3,
            "x_raw": 1.0,
            "y_raw": 1.0,
            "x_smooth": 40.0,
            "y_smooth": 20.0,
            "calibration_valid": True,
        },
        {
            "track_id": 3,
            "x_raw": 90.0,
            "y_raw": 70.0,
            "x_smooth": 50.0,
            "y_smooth": 30.0,
            "calibration_valid": False,
        },
        {
            "track_id": 8,
            "x_raw": 2.0,
            "y_raw": 2.0,
            "x_smooth": 60.0,
            "y_smooth": 40.0,
            "calibration_valid": True,
        },
    ]
    points = smoothed_points(rows)
    assert points == [(3, 40.0, 20.0), (8, 60.0, 40.0)]
    assert len({track_id for track_id, _, _ in points}) == 2


def test_render_writes_a_pitch_image(tmp_path: Path) -> None:
    from tactivision.spatial.heatmap import render_heatmap

    image = render_heatmap([(60.0, 40.0)], title="track 3")
    path = tmp_path / "track.png"
    assert cv2.imwrite(str(path), image)
    assert path.stat().st_size > 0
    assert json.dumps({"ok": True})
