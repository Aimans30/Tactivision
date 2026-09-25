"""Detect players on a sampled clip. Does not track yet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.pipeline.runner import run_detection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TactiVision player detection")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", default="models/detection/yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Stop after this many seconds. Use 0 to scan the whole video.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_seconds = None if args.max_seconds == 0 else args.max_seconds
    try:
        run_dir = run_detection(
            args.input,
            args.output,
            model_path=args.model,
            confidence=args.confidence,
            image_size=args.image_size,
            sample_fps=args.sample_fps,
            max_seconds=max_seconds,
            device=args.device,
        )
    except (FileNotFoundError, ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(run_dir / "detections.jsonl")
    print(run_dir / "detection_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
