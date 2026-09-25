"""Filter existing tracks. Does not rerun detection, tracking, or calibration."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.filtering.players import assign_teams, kit_feature, pitch_keep, torso_color  # noqa: E402
from tactivision.video.reader import VideoReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter non-players from existing tracks")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/1_720p"))
    parser.add_argument("--max-seconds", type=float, default=30.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    track_rows = load_jsonl(args.tracks)
    position_rows = load_jsonl(args.positions)

    track_ids = {
        int(track["track_id"])
        for row in track_rows
        if row["timestamp"] < args.max_seconds
        for track in row["tracks"]
        if track.get("track_id") is not None
    }
    positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in position_rows:
        if not row.get("calibrated") or row["timestamp"] >= args.max_seconds:
            continue
        for player in row["players"]:
            positions[int(player["track_id"])].append((float(player["x"]), float(player["y"])))

    boxes: dict[tuple[int, int], list[float]] = {}
    for row in track_rows:
        if row["timestamp"] >= args.max_seconds:
            continue
        for track in row["tracks"]:
            if track.get("track_id") is None:
                continue
            boxes[(int(row["frame_id"]), int(track["track_id"]))] = track["bbox"]

    pitch_results = [pitch_keep(track_id, positions.get(track_id, [])) for track_id in sorted(track_ids)]
    on_pitch = [item for item in pitch_results if item.kept]
    inside_frames: dict[int, list[int]] = defaultdict(list)
    for row in position_rows:
        if not row.get("calibrated") or row["timestamp"] >= args.max_seconds:
            continue
        frame_id = int(row["frame_id"])
        for player in row["players"]:
            track_id = int(player["track_id"])
            if not (0 <= float(player["x"]) <= 120 and 0 <= float(player["y"]) <= 80):
                continue
            if (frame_id, track_id) in boxes:
                inside_frames[track_id].append(frame_id)
    sample_frames = {track_id: frames[:8] for track_id, frames in inside_frames.items()}

    needed = {frame_id for frames in sample_frames.values() for frame_id in frames}
    colors: dict[int, list[np.ndarray]] = defaultdict(list)
    if needed:
        with VideoReader(args.input) as reader:
            for frame_id, timestamp, frame in reader.iter_frames():
                if timestamp >= args.max_seconds:
                    break
                if frame_id not in needed:
                    continue
                for track_id, frames in sample_frames.items():
                    if frame_id not in frames:
                        continue
                    color = torso_color(frame, boxes[(frame_id, track_id)])
                    if color is not None:
                        colors[track_id].append(color)

    features = {
        track_id: kit_feature(np.median(np.stack(samples), axis=0))
        for track_id, samples in colors.items()
        if samples
    }
    team_results = assign_teams(on_pitch, features)
    by_id = {item.track_id: item for item in pitch_results}
    by_id.update({item.track_id: item for item in team_results})
    results = [by_id[track_id] for track_id in sorted(track_ids)]

    removed = [item for item in results if not item.kept]
    kept = [item for item in results if item.kept]
    reasons: dict[str, int] = defaultdict(int)
    for item in removed:
        reasons[item.reason] += 1
    team_counts: dict[str, int] = defaultdict(int)
    team_colors: dict[int, list[np.ndarray]] = defaultdict(list)
    for item in kept:
        team_counts[str(item.team)] += 1
        if item.track_id in features:
            team_colors[int(item.team)].append(features[item.track_id])

    summary = {
        "tracks_before": len(results),
        "tracks_removed": len(removed),
        "tracks_remaining": len(kept),
        "removed_by_reason": dict(reasons),
        "team_counts": dict(team_counts),
        "team_mean_feature": {
            str(team): [round(float(value), 1) for value in np.mean(np.stack(samples), axis=0)]
            for team, samples in team_colors.items()
        },
        "rejected_examples": [item.to_dict() for item in removed[:12]],
        "kept_examples": [item.to_dict() for item in kept[:12]],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "player_filter.json").write_text(
        json.dumps([item.to_dict() for item in results], indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "player_filter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_example_frame(args, results, boxes, features)
    print(args.output / "player_filter_summary.json")
    return 0


def _write_example_frame(args: argparse.Namespace, results: list, boxes: dict, features: dict) -> None:
    status = {item.track_id: item for item in results}
    target = 120
    with VideoReader(args.input) as reader:
        for frame_id, _timestamp, frame in reader.iter_frames():
            if frame_id != target:
                continue
            canvas = frame.copy()
            for (fid, track_id), bbox in boxes.items():
                if fid != target:
                    continue
                item = status.get(track_id)
                if item is None:
                    continue
                x1, y1, x2, y2 = (int(value) for value in bbox)
                if item.kept and item.team == 0:
                    color = (255, 180, 40)
                elif item.kept and item.team == 1:
                    color = (40, 40, 220)
                else:
                    color = (160, 160, 160)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            path = args.output / "visualizations" / "player_filter.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), canvas)
            break
    _ = features


if __name__ == "__main__":
    raise SystemExit(main())
