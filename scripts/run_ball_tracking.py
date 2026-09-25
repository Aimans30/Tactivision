"""Run ball detection + temporal tracking on a video clip.

Separate from the player YOLO11n / ByteTrack pipeline. Image-space only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.ball.detector import BallDetector  # noqa: E402
from tactivision.ball.schemas import FrameBallDetections  # noqa: E402
from tactivision.ball.tracker import BallTracker  # noqa: E402
from tactivision.pipeline.runner import _open_writer  # noqa: E402
from tactivision.video.reader import VideoReader  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "ball" / "yolo-sn-ball.pt"
DEFAULT_VIDEO = Path(
    "data/raw/soccernet/europe_uefa-champions-league/2016-2017/"
    "2017-04-18 - 21-45 Real Madrid 4 - 2 Bayern Munich/1_720p.mkv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TactiVision ball tracking baseline")
    parser.add_argument("--input", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=Path("data/processed/1_720p/ball"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-gap-frames", type=int, default=15)
    parser.add_argument("--max-center-distance", type=float, default=120.0)
    return parser.parse_args()


def draw_ball(frame: object, observation, detections, timestamp: float) -> object:
    canvas = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 1, cv2.LINE_AA)
    if observation.observed and observation.bbox is not None and observation.center is not None:
        x1, y1, x2, y2 = (int(v) for v in observation.bbox)
        cx, cy = (int(v) for v in observation.center)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 4, (0, 140, 255), -1, cv2.LINE_AA)
        label = f"ball id={observation.track_id} {observation.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )
        status = "observed"
    else:
        status = "missing"
    cv2.putText(
        canvas,
        f"t={timestamp:.2f}s  ball={status}  raw_dets={len(detections)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def summarize(observations: list[dict], detections_rows: list[dict], meta: dict) -> dict:
    frames = len(observations)
    observed = [row for row in observations if row.get("observed")]
    confidences = [float(row["confidence"]) for row in observed if row.get("confidence") is not None]
    # track segments: contiguous observed runs sharing the same track_id
    segments: list[dict] = []
    current: dict | None = None
    for row in observations:
        if not row.get("observed"):
            if current is not None:
                segments.append(current)
                current = None
            continue
        tid = row.get("track_id")
        if current is None or current["track_id"] != tid:
            if current is not None:
                segments.append(current)
            current = {
                "track_id": tid,
                "start_frame": row["frame_id"],
                "end_frame": row["frame_id"],
                "start_time": row["timestamp"],
                "end_time": row["timestamp"],
                "frames": 1,
            }
        else:
            current["end_frame"] = row["frame_id"]
            current["end_time"] = row["timestamp"]
            current["frames"] += 1
    if current is not None:
        segments.append(current)

    longest = max((seg["frames"] for seg in segments), default=0)
    frames_with_det = sum(1 for row in detections_rows if row.get("detections"))
    return {
        "metric_type": "estimated_model_derived",
        "ground_truth_available": False,
        "coordinate_space": "image_pixels",
        "pitch_projection": False,
        "model": meta["model"],
        "model_source": meta["model_source"],
        "confidence_threshold": meta["confidence"],
        "image_size": meta["image_size"],
        "device": meta["device"],
        "tracking_method": meta["tracking_method"],
        "tracker_params": meta["tracker_params"],
        "video": meta["video"],
        "max_seconds": meta["max_seconds"],
        "frames_processed": frames,
        "frames_with_ball_detection": frames_with_det,
        "frames_with_tracked_observation": len(observed),
        "detection_coverage_percent": round(100.0 * frames_with_det / frames, 2) if frames else 0.0,
        "ball_track_segments": len(segments),
        "longest_continuous_segment_frames": longest,
        "longest_continuous_segment_seconds": round(longest / meta["fps"], 3) if meta.get("fps") else None,
        "median_confidence": None if not confidences else round(float(statistics.median(confidences)), 4),
        "mean_confidence": None if not confidences else round(float(statistics.mean(confidences)), 4),
        "class_names_from_model": meta.get("class_names"),
        "ball_class_id": meta.get("ball_class_id"),
        "segments": segments,
    }


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        print(f"error: ball model not found: {args.model}", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"error: video not found: {args.input}", file=sys.stderr)
        return 2

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    det_path = out / "ball_detections.jsonl"
    traj_path = out / "ball_trajectory.jsonl"

    detector = BallDetector(
        args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
    )
    tracker = BallTracker(
        max_gap_frames=args.max_gap_frames,
        max_center_distance_px=args.max_center_distance,
    )

    writer = None
    detections_rows: list[dict] = []
    observations: list[dict] = []
    frames = 0
    fps = 25.0

    with VideoReader(args.input) as reader, det_path.open("w", encoding="utf-8") as det_f, traj_path.open(
        "w", encoding="utf-8"
    ) as traj_f:
        info = reader.metadata()
        fps = info.fps or 25.0
        for frame_id, timestamp, frame in reader.iter_frames():
            if args.max_seconds is not None and timestamp >= args.max_seconds:
                break
            dets = detector.detect(frame)
            frame_dets = FrameBallDetections(frame_id, timestamp, dets)
            det_row = frame_dets.to_dict()
            det_f.write(json.dumps(det_row) + "\n")
            detections_rows.append(det_row)

            obs = tracker.update(frame_id, timestamp, dets)
            obs_row = obs.to_dict()
            traj_f.write(json.dumps(obs_row) + "\n")
            observations.append(obs_row)

            image = draw_ball(frame, obs, dets, timestamp)
            if writer is None:
                h, w = frame.shape[:2]
                writer = _open_writer(out / "ball_preview", fps, (w, h))
            writer.write(image)
            frames += 1
            if frames % 100 == 0:
                print(f"ball-tracked {frames} frames, t={timestamp:.1f}s", flush=True)

    if writer is not None:
        writer.release()

    preview = next(out.glob("ball_preview.*"), None)
    meta = {
        "model": str(args.model).replace("\\", "/"),
        "model_source": (
            "https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/yolo-sn-ball.pt "
            "(SoccerNet-v3 trained YOLOv11 ball detector)"
        ),
        "confidence": args.confidence,
        "image_size": args.image_size,
        "device": detector.device,
        "tracking_method": (
            "greedy single-hypothesis association on the highest-confidence ball detection "
            "per frame (IoU or center-distance); missing frames left unobserved"
        ),
        "tracker_params": {
            "min_iou": tracker.min_iou,
            "max_center_distance_px": tracker.max_center_distance_px,
            "max_gap_frames": tracker.max_gap_frames,
        },
        "video": str(args.input).replace("\\", "/"),
        "max_seconds": args.max_seconds,
        "fps": fps,
        "class_names": detector.class_names,
        "ball_class_id": detector.ball_class_id,
        "preview": None if preview is None else str(preview).replace("\\", "/"),
    }

    summary = summarize(observations, detections_rows, meta)
    summary["preview"] = meta["preview"]
    (out / "ball_tracking_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(det_path)
    print(traj_path)
    print(preview)
    print(out / "ball_tracking_summary.json")
    print(
        f"frames={summary['frames_processed']} "
        f"detected={summary['frames_with_ball_detection']} "
        f"coverage={summary['detection_coverage_percent']}% "
        f"segments={summary['ball_track_segments']} "
        f"longest={summary['longest_continuous_segment_frames']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
