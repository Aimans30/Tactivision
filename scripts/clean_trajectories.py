"""Clean filtered pitch tracks. Does not rerun detection, tracking, or calibration.

Expects ``pitch_positions.jsonl`` from the canonical per-frame player projection
(``scripts/project_players_pitch_per_frame.py``). Smoothing/jump rules unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.spatial.clean import (  # noqa: E402
    JUMP_NOISE_YARDS,
    MAX_GAP_SECONDS,
    MAX_SPEED_YARDS_PER_SECOND,
    SAVGOL_POLYORDER,
    SAVGOL_WINDOW,
    PitchSample,
    reject_jumps,
    smooth_track,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and smooth filtered pitch tracks")
    parser.add_argument("--positions", type=Path, default=Path("data/processed/1_720p/pitch_positions.jsonl"))
    parser.add_argument("--filter", type=Path, default=Path("data/processed/1_720p/player_filter.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/1_720p/pitch_trajectories.jsonl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kept_ids = {
        int(row["track_id"])
        for row in json.loads(args.filter.read_text(encoding="utf-8"))
        if row["kept"]
    }
    by_track: dict[int, list[PitchSample]] = defaultdict(list)
    invalid = 0
    for line in args.positions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        valid = bool(row.get("calibrated"))
        for player in row["players"]:
            track_id = int(player["track_id"])
            if track_id not in kept_ids:
                continue
            if not valid:
                invalid += 1
                continue
            by_track[track_id].append(
                PitchSample(
                    track_id,
                    int(row["frame_id"]),
                    float(row["timestamp"]),
                    float(player["x"]),
                    float(player["y"]),
                    float(row["reprojection_error"]),
                    True,
                )
            )

    rows: list[dict] = []
    jumps_removed = 0
    short_removed = 0
    tracks_after = 0
    for track_id in sorted(by_track):
        kept, removed = reject_jumps(by_track[track_id])
        jumps_removed += removed
        smoothed = smooth_track(kept)
        short_removed += len(kept) - len(smoothed)
        if not smoothed:
            continue
        tracks_after += 1
        rows.extend(smoothed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "tracks_before": len(kept_ids),
        "tracks_with_samples": len(by_track),
        "tracks_after": tracks_after,
        "points_removed_invalid_calibration": invalid,
        "points_removed_jumps": jumps_removed,
        "points_removed_too_short_to_smooth": short_removed,
        "points_removed": invalid + jumps_removed + short_removed,
        "points_written": len(rows),
        "smoothing": {
            "method": "savitzky_golay",
            "window_length": SAVGOL_WINDOW,
            "polyorder": SAVGOL_POLYORDER,
            "axes": ["x", "y"],
            "mode": "interp",
            "max_speed_yards_per_second": MAX_SPEED_YARDS_PER_SECOND,
            "jump_noise_yards": JUMP_NOISE_YARDS,
            "max_gap_seconds": MAX_GAP_SECONDS,
        },
    }
    summary_path = args.output.with_name("pitch_trajectories_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
