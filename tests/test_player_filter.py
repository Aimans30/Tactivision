import cv2
import numpy as np

from tactivision.filtering.players import assign_teams, cluster_teams, inside_pitch, pitch_keep, torso_color


def test_playable_pitch_rejects_touchline_staff() -> None:
    assert inside_pitch(60, 40)
    assert not inside_pitch(60, 82.5)
    assert not inside_pitch(-1, 20)


def test_track_mostly_outside_is_rejected() -> None:
    result = pitch_keep(4, [(60, 82.5), (59, 81.0), (58, 82.0), (40, 40)])
    assert result.kept is False
    assert result.reason == "off_pitch"


def test_track_with_no_projection_is_rejected() -> None:
    result = pitch_keep(9, [])
    assert result.reason == "no_pitch_position"
    assert result.kept is False


def test_two_kits_split_and_referee_color_is_not_a_team() -> None:
    white = np.array([[210, 210, 210], [200, 205, 208], [215, 212, 214], [198, 202, 206]], dtype=float)
    red = np.array([[30, 30, 170], [28, 32, 165], [35, 28, 175], [32, 30, 168]], dtype=float)
    yellow = np.array([[40, 220, 230]], dtype=float)
    features = np.vstack([white, red, yellow])
    labels, keep, _ = cluster_teams(features)
    assert keep[:4].all()
    assert keep[4:8].all()
    assert len(set(labels[:4].tolist())) == 1
    assert len(set(labels[4:8].tolist())) == 1
    assert labels[0] != labels[4]
    assert not keep[8]


def test_assign_teams_keeps_two_groups() -> None:
    on_pitch = [pitch_keep(index, [(20, 40), (22, 41)]) for index in (1, 2, 3, 4)]
    features = {
        1: np.array([210, 210, 210], dtype=float),
        2: np.array([200, 205, 208], dtype=float),
        3: np.array([30, 30, 170], dtype=float),
        4: np.array([28, 32, 165], dtype=float),
    }
    assigned = assign_teams(on_pitch, features)
    kept = [item for item in assigned if item.kept]
    assert {item.team for item in kept} == {0, 1}
    assert all(item.reason == "team" for item in kept)


def test_torso_color_ignores_box_edges() -> None:
    frame = np.zeros((40, 20, 3), dtype=np.uint8)
    frame[8:20, 6:14] = (20, 20, 180)
    color = torso_color(frame, [0, 0, 20, 40])
    assert color is not None
    assert color[2] > 100
    assert cv2 is not None
