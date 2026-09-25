from tactivision.detection.schemas import Detection, FrameDetections


def test_frame_detection_matches_expected_shape() -> None:
    row = FrameDetections(
        frame_id=1842,
        timestamp=73.684,
        detections=(
            Detection(class_name="player", confidence=0.94123, bbox=(412.04, 180.2, 468.49, 342.06)),
        ),
    )

    assert row.to_dict() == {
        "frame_id": 1842,
        "timestamp": 73.684,
        "detections": [
            {
                "class": "player",
                "confidence": 0.9412,
                "bbox": [412.0, 180.2, 468.5, 342.1],
            }
        ],
    }
