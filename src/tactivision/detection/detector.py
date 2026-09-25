"""YOLO person detector, labeled as players for the football MVP."""

from __future__ import annotations

from pathlib import Path

from tactivision.detection.schemas import Detection


class PlayerDetector:
    """Wrap Ultralytics YOLO. COCO class 0 (person) is the player class for now."""

    def __init__(
        self,
        model_path: str | Path = "models/detection/yolo11n.pt",
        *,
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
        self.confidence = confidence
        self.image_size = image_size
        self.person_class_id = person_class_id
        self.player_label = player_label
        self.model = YOLO(str(model_path))

    def detect(self, frame: object) -> tuple[Detection, ...]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[self.person_class_id],
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for xyxy, score in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True):
                x1, y1, x2, y2 = xyxy
                detections.append(
                    Detection(
                        class_name=self.player_label,
                        confidence=float(score),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
        return tuple(detections)
