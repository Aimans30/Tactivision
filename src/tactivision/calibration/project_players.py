"""Project tracked player feet onto the pitch using stored per-frame homographies.

Does not re-estimate or interpolate homographies. Invalid calibration yields an
invalid frame record with no invented player coordinates.
"""

from __future__ import annotations

import numpy as np

from tactivision.calibration.homography import on_pitch, transform_points
from tactivision.spatial.trajectories import foot_point

COORDINATE_SYSTEM = "statsbomb_120x80_yards"
CALIBRATION_SOURCE = "per_frame"


def project_tracks_with_homography(
    tracks: list[dict],
    homography: np.ndarray,
) -> list[dict]:
    """Return on-pitch player dicts ``{track_id, x, y}`` for one frame."""
    ids: list[int] = []
    feet: list[tuple[float, float]] = []
    for track in tracks:
        if track.get("track_id") is None:
            continue
        ids.append(int(track["track_id"]))
        feet.append(foot_point(track["bbox"]))
    if not feet:
        return []
    projected = transform_points(np.array(feet, dtype=np.float32), homography)
    keep = on_pitch(projected)
    return [
        {"track_id": track_id, "x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}
        for track_id, point, visible in zip(ids, projected, keep, strict=True)
        if visible
    ]


def build_pitch_position_row(
    *,
    frame_id: int,
    timestamp: float,
    tracks: list[dict],
    calibration: dict | None,
) -> dict:
    """One ``pitch_positions.jsonl`` row from a per-frame calibration record.

    When calibration is missing or invalid, ``calibrated`` is false and
    ``players`` is empty — coordinates are never invented.
    """
    calibration_valid = bool(
        calibration
        and calibration.get("calibration_valid")
        and calibration.get("homography")
    )
    error = None
    if calibration is not None and calibration.get("reprojection_error") is not None:
        error = float(calibration["reprojection_error"])

    players: list[dict] = []
    if calibration_valid:
        assert calibration is not None
        H = np.asarray(calibration["homography"], dtype=np.float64)
        players = project_tracks_with_homography(tracks, H)

    return {
        "frame_id": int(frame_id),
        "timestamp": round(float(timestamp), 3),
        "calibrated": calibration_valid,
        "reprojection_error": None if error is None else round(error, 3),
        "players": players,
        "calibration_source": CALIBRATION_SOURCE,
        "coordinate_system": COORDINATE_SYSTEM,
        "metric_type": "estimated_model_derived",
    }


def summarize_pitch_positions(rows: list[dict]) -> dict:
    calibrated = [r for r in rows if r.get("calibrated")]
    player_samples = sum(len(r.get("players") or []) for r in calibrated)
    errors = [
        float(r["reprojection_error"])
        for r in calibrated
        if r.get("reprojection_error") is not None
    ]
    errors_sorted = sorted(errors)
    return {
        "metric_type": "estimated_model_derived",
        "coordinate_system": COORDINATE_SYSTEM,
        "calibration_source": CALIBRATION_SOURCE,
        "interpolation": False,
        "total_frames": len(rows),
        "calibrated_frames": len(calibrated),
        "invalid_or_uncalibrated_frames": len(rows) - len(calibrated),
        "player_coordinate_samples": player_samples,
        "reprojection_error_on_calibrated_frames": {
            "count": len(errors_sorted),
            "minimum": None if not errors_sorted else round(errors_sorted[0], 4),
            "median": (
                None
                if not errors_sorted
                else round(errors_sorted[len(errors_sorted) // 2], 4)
            ),
            "maximum": None if not errors_sorted else round(errors_sorted[-1], 4),
        },
        "note": (
            "Player pitch positions use stored per-frame homographies only. "
            "Homographies are not interpolated. Invalid calibration frames have "
            "no player coordinates."
        ),
    }
