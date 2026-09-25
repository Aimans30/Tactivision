"""Project tracked players onto the pitch using per-frame calibration.jsonl.

Canonical player pitch path. Does not re-run detection, tracking, filtering,
or homography estimation. Does not interpolate homographies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.calibration.project_players import (  # noqa: E402
    build_pitch_position_row,
    summarize_pitch_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Player pitch projection from per-frame calibration"
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path("data/processed/1_720p/tracks.jsonl"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("data/processed/1_720p/calibration/per_frame/calibration.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/1_720p/pitch_positions.jsonl"),
    )
    parser.add_argument("--max-seconds", type=float, default=30.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    if not args.tracks.is_file():
        print(f"error: missing {args.tracks}", file=sys.stderr)
        return 2
    if not args.calibration.is_file():
        print(f"error: missing {args.calibration}", file=sys.stderr)
        return 2

    track_rows = load_jsonl(args.tracks)
    cal_by_frame = {int(r["frame_id"]): r for r in load_jsonl(args.calibration)}

    out_rows: list[dict] = []
    for row in track_rows:
        frame_id = int(row["frame_id"])
        timestamp = float(row["timestamp"])
        if args.max_seconds is not None and timestamp >= args.max_seconds:
            break
        out_rows.append(
            build_pitch_position_row(
                frame_id=frame_id,
                timestamp=timestamp,
                tracks=row.get("tracks") or [],
                calibration=cal_by_frame.get(frame_id),
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in out_rows:
            handle.write(json.dumps(record) + "\n")

    summary = summarize_pitch_positions(out_rows)
    summary["tracks_source"] = str(args.tracks).replace("\\", "/")
    summary["calibration_source_path"] = str(args.calibration).replace("\\", "/")
    summary["output"] = str(args.output).replace("\\", "/")
    summary_path = args.output.with_name("pitch_positions_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Keep a root calibration_summary pointer aligned with the canonical path.
    pointer = {
        "coordinate_system": "statsbomb_120x80_yards",
        "calibration_source": "per_frame",
        "canonical": True,
        "per_frame_calibration": str(args.calibration).replace("\\", "/"),
        "pitch_positions": str(args.output).replace("\\", "/"),
        "pitch_positions_summary": str(summary_path).replace("\\", "/"),
        "calibrated_frames": summary["calibrated_frames"],
        "total_frames": summary["total_frames"],
        "player_coordinate_samples": summary["player_coordinate_samples"],
        "interpolation": False,
        "note": (
            "Canonical player pitch projection uses per-frame homographies shared with ball projection."
        ),
    }
    pointer_path = args.output.parent / "calibration_summary.json"
    pointer_path.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")

    print(args.output)
    print(summary_path)
    print(pointer_path)
    print(
        f"frames={summary['total_frames']} calibrated={summary['calibrated_frames']} "
        f"player_samples={summary['player_coordinate_samples']} "
        f"median_err={summary['reprojection_error_on_calibrated_frames']['median']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
