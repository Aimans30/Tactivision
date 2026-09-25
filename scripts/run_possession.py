"""Baseline proximity possession from existing pitch ball + player trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.spatial.possession import (  # noqa: E402
    DEBOUNCE_FRAMES,
    POSSESSION_RADIUS_YARDS,
    build_possession_rows,
    index_players,
    summarize_possession,
)

COLUMNS = [
    "frame",
    "timestamp",
    "ball_x",
    "ball_y",
    "nearest_player_track_id",
    "nearest_player_team",
    "nearest_player_distance",
    "possession_team",
    "possession_status",
    "unavailable_reason",
    "debounce_note",
    "possession_radius_yards",
    "metric_type",
    "coordinate_system",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proximity possession baseline")
    parser.add_argument(
        "--ball",
        type=Path,
        default=Path("data/processed/1_720p/ball/ball_pitch_trajectory_per_frame.jsonl"),
    )
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
        default=Path("data/processed/1_720p/metrics/possession.csv"),
    )
    parser.add_argument(
        "--viz",
        type=Path,
        default=Path("data/processed/1_720p/visualizations/possession"),
    )
    parser.add_argument("--radius", type=float, default=POSSESSION_RADIUS_YARDS)
    parser.add_argument("--debounce-frames", type=int, default=DEBOUNCE_FRAMES)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_timeline(rows: list[dict], path: Path) -> None:
    times = np.array([r["timestamp"] for r in rows], dtype=float)
    # Encode: team0=0, team1=1, unknown=-1, unavailable=-2
    values = []
    colors = []
    for r in rows:
        status = r["possession_status"]
        if status == "assigned" and r["possession_team"] == 0:
            values.append(0)
            colors.append("#E8A317")
        elif status == "assigned" and r["possession_team"] == 1:
            values.append(1)
            colors.append("#2F6FED")
        elif status == "unknown":
            values.append(-1)
            colors.append("#9a9a9a")
        else:
            values.append(-2)
            colors.append("#d0d0d0")
    values = np.array(values)

    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=140)
    fig.patch.set_facecolor("#f4f4f0")
    ax.set_facecolor("#f4f4f0")
    ax.scatter(times, values, c=colors, s=10, marker="|", linewidths=1.5)
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_yticklabels(["unavailable", "unknown", "team 0", "team 1"])
    ax.set_xlabel("Timestamp (seconds)")
    ax.set_title("Possession timeline (proximity heuristic, model-derived)")
    ax.set_xlim(float(times.min()) - 0.2, float(times.max()) + 0.2)
    ax.set_ylim(-2.6, 1.6)
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.8)
    ax.text(
        0.01,
        0.02,
        "Not ground-truth match possession. Gaps = missing ball/players or distance > radius.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#444444",
        va="bottom",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ball_rows = load_jsonl(args.ball)
    traj_rows = load_jsonl(args.trajectories)
    filters = json.loads(args.player_filter.read_text(encoding="utf-8"))
    team_by_track = {
        int(r["track_id"]): int(r["team"])
        for r in filters
        if r.get("kept") and r.get("team") is not None
    }
    players_by_frame = index_players(traj_rows, team_by_track)
    rows = build_possession_rows(
        ball_rows=ball_rows,
        players_by_frame=players_by_frame,
        radius=args.radius,
        debounce_frames=args.debounce_frames,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") if row.get(k) is not None else "" for k in COLUMNS})

    summary = summarize_possession(rows, radius=args.radius, debounce_frames=args.debounce_frames)
    summary["sources"] = {
        "ball": str(args.ball).replace("\\", "/"),
        "trajectories": str(args.trajectories).replace("\\", "/"),
        "player_filter": str(args.player_filter).replace("\\", "/"),
    }
    summary["player_frames_with_samples"] = len(players_by_frame)
    summary["frames_with_ball_and_players"] = sum(
        1
        for r in rows
        if r.get("ball_x") is not None and r.get("nearest_player_track_id") is not None
    )
    summary_path = args.output.with_name("possession_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    args.viz.mkdir(parents=True, exist_ok=True)
    render_timeline(rows, args.viz / "possession_timeline.png")

    print(args.output)
    print(summary_path)
    print(args.viz / "possession_timeline.png")
    print(
        f"assigned={summary['frames_assigned']} unknown={summary['frames_unknown']} "
        f"unavailable={summary['frames_unavailable']} "
        f"team0%={summary['team0_possession_percent_of_all_frames']} "
        f"team1%={summary['team1_possession_percent_of_all_frames']} "
        f"transitions={summary['possession_transitions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
