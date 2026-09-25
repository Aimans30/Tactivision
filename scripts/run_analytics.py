"""Build derived analytics + match_bundle.json from existing processed outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.analytics.bundle import write_match_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble TactiVision analytics bundle")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/processed/1_720p"),
        help="Processed match directory",
    )
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_dir.is_dir():
        print(f"missing run dir: {args.run_dir}", file=sys.stderr)
        return 1
    if not (args.run_dir / "metrics" / "possession.csv").exists():
        print("possession.csv missing — run scripts/run_possession.py first", file=sys.stderr)
        return 1
    bundle = write_match_bundle(args.run_dir, clip_seconds=args.clip_seconds)
    print(args.run_dir / "match_bundle.json")
    print(args.run_dir / "metrics" / "events.jsonl")
    print(args.run_dir / "metrics" / "tactical_states.jsonl")
    print(args.run_dir / "metrics" / "formation_summary.json")
    print(args.run_dir / "metrics" / "ball_analytics_summary.json")
    summaries = bundle.get("summaries") or {}
    events = summaries.get("events") or {}
    ball = summaries.get("ball") or {}
    print(
        f"events={events.get('total_events')} "
        f"ball_coverage={ball.get('pitch_coverage_percent')}% "
        f"formation_modal={((summaries.get('formation') or {}).get('modal_shape_by_team'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
