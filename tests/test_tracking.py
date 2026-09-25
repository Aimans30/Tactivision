from tactivision.tracking.schemas import FrameTracks, Track


def test_frame_tracks_keep_persistent_ids() -> None:
    row = FrameTracks(
        frame_id=25,
        timestamp=1.0,
        shot_id=2,
        tracks=(Track(track_id=7, class_name="player", confidence=0.91, bbox=(10, 20, 30, 80)),),
    )

    assert row.to_dict()["tracks"][0]["track_id"] == 7
    assert row.to_dict()["shot_id"] == 2
    assert row.to_dict()["tracks"][0]["class"] == "player"
