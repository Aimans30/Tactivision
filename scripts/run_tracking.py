"""Track players with ByteTrack. Uses consecutive frames, not a 1 fps sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.pipeline.runner import run_tracking  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TactiVision player tracking")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", default="models/detection/yolo11n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_seconds = None if args.max_seconds == 0 else args.max_seconds
    try:
        run_dir = run_tracking(
            args.input,
            args.output,
            model_path=args.model,
            tracker_name=args.tracker,
            confidence=args.confidence,
            image_size=args.image_size,
            max_seconds=max_seconds,
            device=args.device,
        )
    except (FileNotFoundError, ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(run_dir / "tracks.jsonl")
    print(run_dir / "tracking_summary.json")
    print(run_dir / "visualizations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
