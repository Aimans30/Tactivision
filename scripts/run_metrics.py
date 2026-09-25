"""Estimate distance and speed from cleaned pitch trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.spatial.metrics import measure_track, summarize  # noqa: E402

COLUMNS = [
    "track_id",
    "duration_seconds",
    "distance_yards",
    "mean_speed_yards_per_second",
    "max_speed_yards_per_second",
    "valid_samples",
    "distance_yards_raw",
    "mean_speed_yards_per_second_raw",
    "max_speed_yards_per_second_raw",
    "metric_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimated pitch distance and speed")
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("data/processed/1_720p/pitch_trajectories.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/1_720p/metrics/player_metrics.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    by_track: dict[int, list[dict]] = defaultdict(list)
    for line in args.trajectories.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_track[int(row["track_id"])].append(row)

    measured = []
    for track_id in sorted(by_track):
        row = measure_track(by_track[track_id])
        if row is not None:
            measured.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(measured)

    summary = summarize(measured)
    summary_path = args.output.with_name("player_metrics_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
