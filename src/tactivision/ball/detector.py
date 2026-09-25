"""Football ball detector. Separate from the player COCO-person detector."""

from __future__ import annotations

from pathlib import Path

from tactivision.ball.schemas import BallDetection


class BallDetector:
    """Ultralytics YOLO fine-tuned for soccer ball detection.

    Default weights: SoccerNet-v3D ``yolo-sn-ball.pt`` (ball-only).
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.25,
        image_size: int = 1280,
        ball_class_id: int | None = None,
        ball_label: str = "ball",
        device: str | None = None,
    ) -> None:
        import torch
        from ultralytics import YOLO

        if device is None or device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.confidence = confidence
        self.image_size = image_size
        self.ball_label = ball_label
        self.model = YOLO(str(model_path))
        names = self.model.names or {}
        if ball_class_id is None:
            # Prefer an explicit ball/sports ball class name when present.
            ball_ids = [
                int(idx)
                for idx, name in names.items()
                if str(name).lower().replace("_", " ") in {"ball", "sports ball", "football", "soccer ball"}
            ]
            if len(ball_ids) == 1:
                ball_class_id = ball_ids[0]
            elif len(names) == 1:
                ball_class_id = int(next(iter(names.keys())))
            else:
                ball_class_id = 0
        self.ball_class_id = int(ball_class_id)
        self.class_names = {int(k): str(v) for k, v in names.items()}

    def detect(self, frame: object) -> tuple[BallDetection, ...]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[self.ball_class_id],
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        detections: list[BallDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for xyxy, score in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True):
                x1, y1, x2, y2 = xyxy
                detections.append(
                    BallDetection(
                        class_name=self.ball_label,
                        confidence=float(score),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
        detections.sort(key=lambda det: det.confidence, reverse=True)
        return tuple(detections)
