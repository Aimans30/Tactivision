"""Per-frame pitch calibration for the 30-second clip.

Runs the existing PitchCalibrator on every frame. Does not interpolate
homographies. Does not modify ball/player detection or tracking.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.calibration.detector import PitchCalibrator  # noqa: E402
from tactivision.calibration.homography import (  # noqa: E402
    estimate_homography,
    mean_reprojection_error,
)
from tactivision.video.reader import VideoReader  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "calibration" / "yolo-football-pitch-detection.pt"
DEFAULT_VIDEO = Path(
    "data/raw/soccernet/europe_uefa-champions-league/2016-2017/"
    "2017-04-18 - 21-45 Real Madrid 4 - 2 Bayern Munich/1_720p.mkv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-frame pitch calibration coverage")
    parser.add_argument("--input", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/1_720p/calibration/per_frame"),
    )
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--max-error", type=float, default=3.0)
    parser.add_argument("--min-correspondences", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        print(f"error: pitch model not found: {args.model}", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"error: video not found: {args.input}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "calibration.jsonl"
    calibrator = PitchCalibrator(args.model)

    rows: list[dict] = []
    fps = 25.0
    with VideoReader(args.input) as reader, out_path.open("w", encoding="utf-8") as handle:
        info = reader.metadata()
        fps = info.fps or 25.0
        for frame_id, timestamp, frame in reader.iter_frames():
            if args.max_seconds is not None and timestamp >= args.max_seconds:
                break

            record: dict = {
                "frame_id": frame_id,
                "timestamp": round(float(timestamp), 3),
                "has_sufficient_correspondences": False,
                "n_correspondences": 0,
                "n_inliers": 0,
                "calibration_valid": False,
                "reprojection_error": None,
                "homography": None,
                "failure_reason": None,
                "coordinate_system": "statsbomb_120x80_yards",
            }

            correspondences = calibrator.correspondences(frame)
            if correspondences is None:
                record["failure_reason"] = "insufficient_or_no_keypoints"
                handle.write(json.dumps(record) + "\n")
                rows.append(record)
                if (frame_id + 1) % 100 == 0:
                    print(f"calibrated {frame_id + 1} frames...", flush=True)
                continue

            image_points, pitch_points = correspondences
            n_corr = int(len(image_points))
            record["n_correspondences"] = n_corr
            if n_corr < args.min_correspondences:
                record["failure_reason"] = "insufficient_correspondences"
                handle.write(json.dumps(record) + "\n")
                rows.append(record)
                continue

            record["has_sufficient_correspondences"] = True
            homography, inliers = estimate_homography(image_points, pitch_points)
            if homography is None or inliers is None or int(inliers.sum()) < args.min_correspondences:
                record["failure_reason"] = "homography_estimation_failed"
                handle.write(json.dumps(record) + "\n")
                rows.append(record)
                continue

            n_inliers = int(inliers.sum())
            record["n_inliers"] = n_inliers
            error = mean_reprojection_error(image_points, pitch_points, homography, inliers)
            record["reprojection_error"] = round(float(error), 4)
            if error > args.max_error:
                record["failure_reason"] = "reprojection_error_above_threshold"
                handle.write(json.dumps(record) + "\n")
                rows.append(record)
                continue

            record["calibration_valid"] = True
            record["homography"] = [[round(float(v), 8) for v in row] for row in homography.tolist()]
            handle.write(json.dumps(record) + "\n")
            rows.append(record)
            if (frame_id + 1) % 100 == 0:
                valid_so_far = sum(1 for r in rows if r["calibration_valid"])
                print(
                    f"calibrated {frame_id + 1} frames, valid={valid_so_far}",
                    flush=True,
                )

    total = len(rows)
    sufficient = sum(1 for r in rows if r["has_sufficient_correspondences"])
    valid = [r for r in rows if r["calibration_valid"]]
    invalid = total - len(valid)
    errors = [float(r["reprojection_error"]) for r in valid if r["reprojection_error"] is not None]
    reasons: dict[str, int] = {}
    for r in rows:
        if r["calibration_valid"]:
            continue
        key = r.get("failure_reason") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1

    # Compare to previous ~2 FPS grid (every 12 frames at 25 fps).
    prev_step = 12
    prev_candidates = [r for r in rows if int(r["frame_id"]) % prev_step == 0]
    prev_valid = sum(1 for r in prev_candidates if r["calibration_valid"])

    summary = {
        "metric_type": "estimated_model_derived",
        "coordinate_system": "statsbomb_120x80_yards",
        "video": str(args.input).replace("\\", "/"),
        "model": str(args.model).replace("\\", "/"),
        "max_seconds": args.max_seconds,
        "max_error_yards": args.max_error,
        "min_correspondences": args.min_correspondences,
        "fps": fps,
        "total_frames": total,
        "frames_with_sufficient_correspondences": sufficient,
        "valid_homographies": len(valid),
        "invalid_frames": invalid,
        "calibration_coverage_percent": round(100.0 * len(valid) / total, 2) if total else 0.0,
        "reprojection_error_valid_frames": {
            "count": len(errors),
            "minimum": None if not errors else round(min(errors), 4),
            "median": None if not errors else round(float(statistics.median(errors)), 4),
            "mean": None if not errors else round(float(statistics.mean(errors)), 4),
            "maximum": None if not errors else round(max(errors), 4),
        },
        "failure_reason_counts": reasons,
        "previous_2fps_comparison": {
            "note": "Previous player calibration sampled every 12th frame (~2 FPS at 25 FPS).",
            "candidate_frames_on_that_grid": len(prev_candidates),
            "valid_on_that_grid_under_this_run": prev_valid,
            "previous_pitch_positions_calibrated_frames": 63,
            "this_run_valid_homographies": len(valid),
        },
        "interpolation": False,
        "outputs": {"calibration_jsonl": str(out_path).replace("\\", "/")},
    }

    (args.output / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(out_path)
    print(args.output / "calibration_summary.json")
    print(
        f"total={total} sufficient={sufficient} valid={len(valid)} "
        f"invalid={invalid} coverage={summary['calibration_coverage_percent']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
