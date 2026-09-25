"""Pitch line landmarks from a keypoint model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tactivision.calibration.pitch import pitch_vertices


class PitchCalibrator:
    def __init__(self, model_path: str | Path, *, confidence: float = 0.5, device: str = "auto") -> None:
        import torch
        from ultralytics import YOLO

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.confidence = confidence
        self.model = YOLO(str(model_path))
        self.vertices = pitch_vertices()

    def correspondences(self, frame: object) -> tuple[np.ndarray, np.ndarray] | None:
        results = self.model.predict(frame, conf=0.25, imgsz=640, device=self.device, verbose=False)
        if not results or results[0].keypoints is None:
            return None
        keypoints = results[0].keypoints
        if keypoints.xy is None or len(keypoints.xy) == 0:
            return None
        xy = keypoints.xy[0].detach().cpu().numpy()
        conf = keypoints.conf
        if conf is None:
            visible = np.ones(len(xy), dtype=bool)
        else:
            visible = conf[0].detach().cpu().numpy() >= self.confidence
        if xy.shape[0] != len(self.vertices):
            return None
        # A keypoint at (0, 0) with low confidence is "not in frame".
        visible &= np.linalg.norm(xy, axis=1) > 1.0
        if int(visible.sum()) < 4:
            return None
        return xy[visible], self.vertices[visible]
