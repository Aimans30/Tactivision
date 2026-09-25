"""Compute team-level pitch shape from cleaned trajectories and team labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.spatial.team_shape import (  # noqa: E402
    MIN_PLAYERS,
    build_team_shape_rows,
    summarize_team_shape,
)

COLUMNS = [
    "frame",
    "timestamp",
    "metric_type",
    "coordinate_system",
    "coordinates",
    "min_players_required",
    "team0_valid",
    "team0_n_available",
    "team0_n_players",
    "team0_centroid_x",
    "team0_centroid_y",
    "team0_width",
    "team0_length",
    "team0_compactness",
    "team1_valid",
    "team1_n_available",
    "team1_n_players",
    "team1_centroid_x",
    "team1_centroid_y",
    "team1_width",
    "team1_length",
    "team1_compactness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Team shape metrics from pitch trajectories")
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("data/processed/1_720p/pitch_trajectories.jsonl"),
    )
    parser.add_argument(
        "--player-filter",
        type=Path,
        default=Path("data/processed/1_720p/player_filter.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/1_720p/metrics/team_shape.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory_rows = [
        json.loads(line)
        for line in args.trajectories.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    filters = json.loads(args.player_filter.read_text(encoding="utf-8"))
    team_by_track = {
        int(row["track_id"]): int(row["team"])
        for row in filters
        if row.get("kept") and row.get("team") is not None
    }

    rows = build_team_shape_rows(trajectory_rows, team_by_track, min_players=MIN_PLAYERS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_team_shape(rows, team_by_track)
    summary["sources"] = {
        "trajectories": str(args.trajectories).replace("\\", "/"),
        "player_filter": str(args.player_filter).replace("\\", "/"),
    }
    summary_path = args.output.with_name("team_shape_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(summary_path)
    print(
        f"timestamps={summary['timestamps_total']} "
        f"both_valid={summary['timestamps_both_teams_valid']} "
        f"team0_valid={summary['teams']['0']['valid_samples']} "
        f"team1_valid={summary['teams']['1']['valid_samples']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
