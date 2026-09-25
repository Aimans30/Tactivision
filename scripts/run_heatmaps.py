"""Draw pitch heatmaps from cleaned trajectories. Does not rerun earlier stages."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.spatial.heatmap import SPARSE_SAMPLE_THRESHOLD, render_heatmap, smoothed_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pitch heatmaps from cleaned trajectories")
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("data/processed/1_720p/pitch_trajectories.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/1_720p/visualizations/heatmaps"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.trajectories.read_text(encoding="utf-8").splitlines() if line.strip()]
    points = smoothed_points(rows)
    by_track: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for track_id, x, y in points:
        by_track[track_id].append((x, y))

    args.output.mkdir(parents=True, exist_ok=True)
    for track_id in sorted(by_track):
        image = render_heatmap(by_track[track_id], title=f"track {track_id}  n={len(by_track[track_id])}")
        if not cv2.imwrite(str(args.output / f"track_{track_id:04d}.png"), image):
            raise OSError(f"Failed to write heatmap for track {track_id}")

    aggregate = render_heatmap(
        [(x, y) for _track_id, x, y in points],
        title=f"all tracks  n={len(points)}",
    )
    aggregate_path = args.output / "aggregate_heatmap.png"
    if not cv2.imwrite(str(aggregate_path), aggregate):
        raise OSError(f"Failed to write {aggregate_path}")

    xs = [x for _track_id, x, y in points]
    ys = [y for _track_id, x, y in points]
    sparse = sorted(track_id for track_id, samples in by_track.items() if len(samples) < SPARSE_SAMPLE_THRESHOLD)
    summary = {
        "tracks_included": len(by_track),
        "total_coordinate_samples": len(points),
        "coordinate_range": {
            "x_min": round(min(xs), 2),
            "x_max": round(max(xs), 2),
            "y_min": round(min(ys), 2),
            "y_max": round(max(ys), 2),
        },
        "sparse_sample_threshold": SPARSE_SAMPLE_THRESHOLD,
        "tracks_with_sparse_coverage": sparse,
        "coordinate_system": "statsbomb_120x80",
        "coordinates": "x_smooth,y_smooth",
    }
    (args.output / "heatmap_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(aggregate_path)
    print(args.output / "heatmap_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
