"""Visualize team-shape CSV on the StatsBomb pitch and over time.

Does not rerun detection, tracking, calibration, smoothing, or team labeling.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from tactivision.calibration.pitch import PITCH_EDGES, PITCH_LENGTH, PITCH_WIDTH, pitch_vertices  # noqa: E402

TEAM0_COLOR = (40, 170, 230)  # BGR amber-ish for OpenCV
TEAM1_COLOR = (220, 140, 40)  # BGR blue
TEAM0_MPL = "#E8A317"
TEAM1_MPL = "#2F6FED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize team shape metrics")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/1_720p/metrics/team_shape.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/1_720p/visualizations/team_shape"),
    )
    return parser.parse_args()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: str) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "frame": int(raw["frame"]),
                    "timestamp": float(raw["timestamp"]),
                    "team0_valid": _as_bool(raw["team0_valid"]),
                    "team1_valid": _as_bool(raw["team1_valid"]),
                    "team0_centroid_x": _as_float(raw["team0_centroid_x"]),
                    "team0_centroid_y": _as_float(raw["team0_centroid_y"]),
                    "team1_centroid_x": _as_float(raw["team1_centroid_x"]),
                    "team1_centroid_y": _as_float(raw["team1_centroid_y"]),
                    "team0_width": _as_float(raw["team0_width"]),
                    "team1_width": _as_float(raw["team1_width"]),
                    "team0_length": _as_float(raw["team0_length"]),
                    "team1_length": _as_float(raw["team1_length"]),
                    "team0_n_players": _as_float(raw["team0_n_players"]),
                    "team1_n_players": _as_float(raw["team1_n_players"]),
                }
            )
    rows.sort(key=lambda row: (row["timestamp"], row["frame"]))
    return rows


def render_centroids(rows: list[dict], path: Path) -> None:
    width, height = 1000, 700
    margin = 50
    canvas = np.full((height, width, 3), (28, 96, 42), dtype=np.uint8)
    scale_x = (width - 2 * margin) / PITCH_LENGTH
    scale_y = (height - 2 * margin) / PITCH_WIDTH

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(margin + x * scale_x), int(margin + y * scale_y)

    vertices = pitch_vertices()
    for start, end in PITCH_EDGES:
        cv2.line(
            canvas,
            to_px(*vertices[start - 1]),
            to_px(*vertices[end - 1]),
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    center = to_px(PITCH_LENGTH / 2, PITCH_WIDTH / 2)
    radius_px = int((915.0 / 7000.0 * PITCH_WIDTH) * scale_y)
    cv2.circle(canvas, center, max(radius_px, 1), (230, 230, 230), 1, cv2.LINE_AA)

    def draw_team(team: int, color: tuple[int, int, int]) -> int:
        prev_px: tuple[int, int] | None = None
        prev_was_valid = False
        first_px: tuple[int, int] | None = None
        last_px: tuple[int, int] | None = None
        n = 0
        for row in rows:
            valid = row["team0_valid"] if team == 0 else row["team1_valid"]
            if not valid:
                prev_was_valid = False
                prev_px = None
                continue
            x = row["team0_centroid_x"] if team == 0 else row["team1_centroid_x"]
            y = row["team0_centroid_y"] if team == 0 else row["team1_centroid_y"]
            assert x is not None and y is not None
            px = to_px(x, y)
            if prev_was_valid and prev_px is not None:
                cv2.line(canvas, prev_px, px, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, px, 5, color, -1, cv2.LINE_AA)
            if first_px is None:
                first_px = px
            last_px = px
            prev_px = px
            prev_was_valid = True
            n += 1
        if first_px is not None and last_px is not None:
            cv2.circle(canvas, first_px, 8, color, 2, cv2.LINE_AA)
            cv2.drawMarker(canvas, last_px, color, cv2.MARKER_SQUARE, 12, 2, cv2.LINE_AA)
        return n

    n0 = draw_team(0, TEAM0_COLOR)
    n1 = draw_team(1, TEAM1_COLOR)

    cv2.putText(
        canvas,
        "Team centroids (model-derived, valid samples only)",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Team 0  n={n0}   start=open circle   end=square",
        (14, height - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEAM0_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Team 1  n={n1}   gaps not interpolated",
        (14, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEAM1_COLOR,
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Failed to write {path}")


def render_series(
    rows: list[dict],
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    times = np.array([row["timestamp"] for row in rows], dtype=float)
    y0 = np.array(
        [
            row[f"team0_{metric}"] if row["team0_valid"] and row[f"team0_{metric}"] is not None else np.nan
            for row in rows
        ],
        dtype=float,
    )
    y1 = np.array(
        [
            row[f"team1_{metric}"] if row["team1_valid"] and row[f"team1_{metric}"] is not None else np.nan
            for row in rows
        ],
        dtype=float,
    )
    invalid0 = ~np.isfinite(y0)
    invalid1 = ~np.isfinite(y1)

    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=140)
    fig.patch.set_facecolor("#f4f4f0")
    ax.set_facecolor("#f4f4f0")

    # Mark timestamps where a team measurement is missing/invalid.
    for t, bad0, bad1 in zip(times, invalid0, invalid1, strict=True):
        if bad0:
            ax.axvline(t, color=TEAM0_MPL, alpha=0.12, linewidth=1.0, zorder=0)
        if bad1:
            ax.axvline(t, color=TEAM1_MPL, alpha=0.12, linewidth=1.0, zorder=0)

    ax.plot(times, y0, color=TEAM0_MPL, linewidth=2.0, marker="o", markersize=4.5, label="Team 0 (valid)")
    ax.plot(times, y1, color=TEAM1_MPL, linewidth=2.0, marker="o", markersize=4.5, label="Team 1 (valid)")

    # Rug marks at the bottom for timestamps where that team has no valid measurement.
    finite_vals = np.concatenate([y0[np.isfinite(y0)], y1[np.isfinite(y1)]])
    y_max = float(np.max(finite_vals)) if finite_vals.size else 1.0
    ax.set_ylim(-0.08 * y_max, y_max * 1.12)
    if invalid0.any():
        ax.vlines(
            times[invalid0],
            ymin=-0.06 * y_max,
            ymax=-0.01 * y_max,
            colors=TEAM0_MPL,
            linewidth=1.4,
            label="Team 0 insufficient / invalid",
        )
    if invalid1.any():
        ax.vlines(
            times[invalid1],
            ymin=-0.06 * y_max,
            ymax=-0.01 * y_max,
            colors=TEAM1_MPL,
            linewidth=1.4,
            linestyles="dashed",
            label="Team 1 insufficient / invalid",
        )
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8)

    ax.set_xlabel("Timestamp (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(float(times.min()) - 0.3, float(times.max()) + 0.3)
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.8)
    ax.legend(loc="best", frameon=True, fontsize=8)
    ax.text(
        0.01,
        0.02,
        "Model-derived estimates. Line gaps and bottom ticks = insufficient valid players (not interpolated).",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#444444",
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_readme(path: Path, rows: list[dict]) -> None:
    n0 = sum(1 for row in rows if row["team0_valid"])
    n1 = sum(1 for row in rows if row["team1_valid"])
    both = sum(1 for row in rows if row["team0_valid"] and row["team1_valid"])
    path.write_text(
        f"""# Team-shape visualizations

Model-derived estimates from `metrics/team_shape.csv` for the `1_720p` clip.

## Files

| File | Content |
| --- | --- |
| `team_centroids.png` | StatsBomb 120×80 pitch with team centroid trajectories |
| `team_width_over_time.png` | Team width (`max x − min x`) vs timestamp |
| `team_length_over_time.png` | Team length (`max y − min y`) vs timestamp |

## Important limitations

- These are **model-derived estimates**, not ground-truth tactical annotations.
- Only timestamps with **sufficient valid player coordinates** (at least 3 per team in the shape CSV) contribute plotted values.
- This is **not formation recognition**. Centroids and spans do not identify a 4-3-3, press shape, or lineup.
- **Fragmented tracks** and **incomplete player visibility** affect every measurement: missing detections, kit-filter drops, and short ByteTrack IDs change who is present at each timestamp.

## Plotting rules used here

- Centroid paths connect only **consecutive valid samples** in time order. Invalid timestamps break the polyline; gaps are **not** interpolated.
- Width/length series use NaN breaks so matplotlib does not draw through missing values. Invalid timestamps are also marked with light vertical guides and `x` markers.

## Counts on this clip

- Timestamps in CSV: {len(rows)}
- Team 0 valid shape samples plotted: {n0}
- Team 1 valid shape samples plotted: {n1}
- Timestamps with both teams valid: {both}
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input)
    args.output.mkdir(parents=True, exist_ok=True)

    render_centroids(rows, args.output / "team_centroids.png")
    render_series(
        rows,
        metric="width",
        ylabel="Width (yards, max x − min x)",
        title="Team width over time (model-derived)",
        path=args.output / "team_width_over_time.png",
    )
    render_series(
        rows,
        metric="length",
        ylabel="Length (yards, max y − min y)",
        title="Team length over time (model-derived)",
        path=args.output / "team_length_over_time.png",
    )
    write_readme(args.output / "README.md", rows)

    for name in (
        "team_centroids.png",
        "team_width_over_time.png",
        "team_length_over_time.png",
        "README.md",
    ):
        print(args.output / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
