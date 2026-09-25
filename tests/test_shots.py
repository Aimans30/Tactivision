from pathlib import Path

from tactivision.tracking.shots import ShotDetector, finalize_shots, load_soccernet_boundaries
import numpy as np


def _frame(color: tuple[int, int, int]) -> object:
    return np.full((48, 64, 3), color, dtype=np.uint8)


def test_hard_cut_is_detected_and_soft_change_is_not() -> None:
    detector = ShotDetector(min_hist_distance=0.45, min_cut_gap_frames=1)
    base = _frame((20, 160, 30))
    noisy = base.copy()
    noisy[10:20, 10:20] = (25, 150, 35)
    assert detector.is_cut(base, 0) is False
    assert detector.is_cut(noisy, 1) is False
    assert detector.is_cut(_frame((40, 40, 200)), 2) is True


def test_shots_are_closed_at_clip_end() -> None:
    shots = finalize_shots(
        [(0, 0.0, "start"), (75, 3.0, "cut"), (200, 8.0, "cut")],
        last_frame=249,
        last_time=9.96,
    )
    assert [shot.shot_id for shot in shots] == [0, 1, 2]
    assert shots[0].duration_seconds == 3.0
    assert shots[1].start_frame == 75
    assert shots[1].end_frame == 199
    assert shots[2].end_time == 9.96


def test_soccernet_labels_ignore_other_half_and_clip_end(tmp_path: Path) -> None:
    path = tmp_path / "Labels-cameras.json"
    path.write_text(
        """
        {
          "annotations": [
            {"change_type": "abrupt", "gameTime": "1 - 00:10", "label": "Main camera center", "position": "10000"},
            {"change_type": "abrupt", "gameTime": "1 - 00:40", "label": "Close-up", "position": "40000"},
            {"change_type": "abrupt", "gameTime": "2 - 00:05", "label": "Main camera center", "position": "5000"}
          ]
        }
        """,
        encoding="utf-8",
    )
    boundaries = load_soccernet_boundaries(path, half=1, max_seconds=30.0)
    assert [round(item.timestamp, 1) for item in boundaries] == [0.0, 10.0]
    assert boundaries[1].label == "Main camera center"
