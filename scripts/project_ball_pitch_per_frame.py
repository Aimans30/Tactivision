"""Project ball trajectory using per-frame calibration.jsonl (no H interpolation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.calibration.homography import transform_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ball pitch projection from per-frame calibration")
    parser.add_argument(
        "--ball-trajectory",
        type=Path,
        default=Path("data/processed/1_720p/ball/ball_trajectory.jsonl"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("data/processed/1_720p/calibration/per_frame/calibration.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/1_720p/ball/ball_pitch_trajectory_per_frame.jsonl"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def contiguous_segments(rows: list[dict]) -> list[dict]:
    segments: list[dict] = []
    current: dict | None = None
    for row in rows:
        if not row.get("pitch_valid"):
            if current is not None:
                segments.append(current)
                current = None
            continue
        if current is None:
            current = {
                "start_frame": row["frame"],
                "end_frame": row["frame"],
                "start_time": row["timestamp"],
                "end_time": row["timestamp"],
                "frames": 1,
            }
        else:
            current["end_frame"] = row["frame"]
            current["end_time"] = row["timestamp"]
            current["frames"] += 1
    if current is not None:
        segments.append(current)
    return segments


def main() -> int:
    args = parse_args()
    if not args.ball_trajectory.is_file():
        print(f"error: missing {args.ball_trajectory}", file=sys.stderr)
        return 2
    if not args.calibration.is_file():
        print(f"error: missing {args.calibration}", file=sys.stderr)
        return 2

    ball_rows = load_jsonl(args.ball_trajectory)
    cal_by_frame = {int(r["frame_id"]): r for r in load_jsonl(args.calibration)}

    out_rows: list[dict] = []
    for row in ball_rows:
        frame_id = int(row["frame_id"])
        timestamp = float(row["timestamp"])
        observed = bool(row.get("observed"))
        cal = cal_by_frame.get(frame_id)
        calibration_valid = bool(cal and cal.get("calibration_valid") and cal.get("homography"))

        record: dict = {
            "frame": frame_id,
            "timestamp": round(timestamp, 3),
            "observed": observed,
            "calibration_valid": calibration_valid,
            "pitch_valid": False,
            "image_x": None,
            "image_y": None,
            "pitch_x": None,
            "pitch_y": None,
            "track_id": row.get("track_id"),
            "confidence": row.get("confidence"),
            "metric_type": "estimated_model_derived",
            "coordinate_system": "statsbomb_120x80_yards",
            "calibration_source": "per_frame",
        }

        if not observed:
            out_rows.append(record)
            continue

        center = row.get("center")
        if not center or len(center) != 2:
            out_rows.append(record)
            continue
        image_x, image_y = float(center[0]), float(center[1])
        record["image_x"] = round(image_x, 1)
        record["image_y"] = round(image_y, 1)

        if not calibration_valid:
            out_rows.append(record)
            continue

        H = np.asarray(cal["homography"], dtype=np.float64)
        projected = transform_points(np.array([[image_x, image_y]], dtype=np.float32), H)[0]
        record["pitch_valid"] = True
        record["pitch_x"] = round(float(projected[0]), 3)
        record["pitch_y"] = round(float(projected[1]), 3)
        out_rows.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in out_rows:
            handle.write(json.dumps(record) + "\n")

    observed = [r for r in out_rows if r["observed"]]
    projected = [r for r in out_rows if r["pitch_valid"]]
    segments = contiguous_segments(out_rows)
    longest = max((s["frames"] for s in segments), default=0)
    summary = {
        "metric_type": "estimated_model_derived",
        "ground_truth": False,
        "coordinate_system": "statsbomb_120x80_yards",
        "calibration_source": str(args.calibration).replace("\\", "/"),
        "ball_trajectory_source": str(args.ball_trajectory).replace("\\", "/"),
        "total_frames": len(out_rows),
        "observed_ball_frames": len(observed),
        "valid_ball_pitch_projections": len(projected),
        "projection_percentage_of_observed": (
            round(100.0 * len(projected) / len(observed), 2) if observed else 0.0
        ),
        "continuous_projected_segments": len(segments),
        "longest_projected_segment_frames": longest,
        "longest_projected_segment_seconds": round(longest / 25.0, 3),
        "previous_2fps_projection_valid": 54,
        "interpolation": False,
        "output": str(args.output).replace("\\", "/"),
    }
    summary_path = args.output.with_name("ball_pitch_projection_per_frame_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(summary_path)
    print(
        f"observed={summary['observed_ball_frames']} "
        f"projected={summary['valid_ball_pitch_projections']} "
        f"pct={summary['projection_percentage_of_observed']}% "
        f"segments={summary['continuous_projected_segments']} "
        f"longest={longest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
