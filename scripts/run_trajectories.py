"""Build image-space trajectories from a tracks.jsonl file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.pipeline.runner import run_trajectories  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TactiVision trajectories")
    parser.add_argument("--input", required=True, type=Path, help="Source video")
    parser.add_argument("--tracks", required=True, type=Path, help="tracks.jsonl from ByteTrack")
    parser.add_argument("--output", type=Path, default=Path("data/processed/1_720p"))
    parser.add_argument("--min-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out = run_trajectories(args.input, args.tracks, args.output, min_seconds=args.min_seconds)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(out / "trajectories.json")
    print(out / "visualizations" / "trajectories.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
