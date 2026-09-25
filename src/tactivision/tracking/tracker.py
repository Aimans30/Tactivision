"""ByteTrack over YOLO detections. BoT-SORT is the later comparison."""

from __future__ import annotations

from pathlib import Path

from tactivision.tracking.schemas import Track


class PlayerTracker:
    def __init__(
        self,
        model_path: str | Path = "models/detection/yolo11n.pt",
        *,
        tracker_name: str = "bytetrack.yaml",
        confidence: float = 0.35,
        image_size: int = 1280,
        person_class_id: int = 0,
        player_label: str = "player",
        device: str | None = None,
    ) -> None:
        import torch
        from ultralytics import YOLO

        if device is None or device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tracker_name = tracker_name
        self.confidence = confidence
        self.image_size = image_size
        self.person_class_id = person_class_id
        self.player_label = player_label
        self.model = YOLO(str(model_path))

    def reset(self) -> None:
        self.model.predictor = None

    def update(self, frame: object) -> tuple[Track, ...]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_name,
            conf=self.confidence,
            classes=[self.person_class_id],
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        tracks: list[Track] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
            for track_id, xyxy, score in zip(ids, boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True):
                x1, y1, x2, y2 = xyxy
                tracks.append(
                    Track(
                        track_id=None if track_id is None else int(track_id),
                        class_name=self.player_label,
                        confidence=float(score),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
        return tuple(tracks)
