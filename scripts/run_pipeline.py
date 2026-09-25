"""Canonical TactiVision end-to-end pipeline CLI.

Process an arbitrary football video into ``data/processed/<run-id>/``.

Example::

    python scripts/run_pipeline.py --video path/to/clip.mp4 --run-id demo_clip --max-seconds 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.pipeline.match_pipeline import run_match_pipeline  # noqa: E402
from tactivision.pipeline.paths import validate_run_id  # noqa: E402
from tactivision.pipeline.runner import run_ingestion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TactiVision end-to-end match pipeline (or ingest-only)"
    )
    parser.add_argument(
        "--video",
        "--input",
        dest="video",
        type=Path,
        required=True,
        help="Path to a football video",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Output directory name under data/processed/ (default: video stem)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Processed runs root (default: data/processed)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional clip length cap in seconds (omit to process until EOF)",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only write metadata.json (legacy Milestone-0 behavior)",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="With --ingest-only: also write sampled frames",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip sampled detection pass (tracking still runs)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip writing tracking/ball preview videos",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video
    run_id = args.run_id or video.stem
    try:
        validate_run_id(run_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.ingest_only:
        try:
            run_dir = run_ingestion(
                video,
                args.output_root,
                run_id=run_id,
                extract=args.extract_frames,
                sample_fps=args.sample_fps,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(run_dir / "metadata.json")
        return 0

    try:
        summary = run_match_pipeline(
            video,
            run_id,
            processed_root=args.output_root,
            max_seconds=args.max_seconds,
            device=args.device,
            write_previews=not args.no_preview,
            skip_detection=args.skip_detection,
            detection_sample_fps=args.sample_fps,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(summary["run_dir"])
    print(summary["match_bundle"])
    stages = summary.get("stages") or {}
    tracking = stages.get("tracking") or {}
    calibration = stages.get("calibration") or {}
    ball = stages.get("ball_pitch") or {}
    possession = stages.get("possession") or {}
    print(
        f"frames={tracking.get('frames')} "
        f"tracks={tracking.get('unique_track_ids')} "
        f"calibrated={calibration.get('calibrated_frames')}/{calibration.get('frames')} "
        f"ball_projected={ball.get('valid_ball_pitch_projections')} "
        f"possession_assigned={possession.get('frames_assigned')} "
        f"runtime_s={summary.get('runtime_seconds')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
