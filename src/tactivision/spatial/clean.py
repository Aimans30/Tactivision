"""Clean filtered pitch tracks. No distance or speed is computed here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

# A sprint is about 11 yards/s. The extra yards cover calibration jitter, not a cut.
MAX_SPEED_YARDS_PER_SECOND = 12.0
JUMP_NOISE_YARDS = 3.0
# Samples are 2 Hz. A wider gap is a missing frame, not the next step of a run.
MAX_GAP_SECONDS = 0.75
SAVGOL_WINDOW = 5
SAVGOL_POLYORDER = 2


@dataclass(frozen=True)
class PitchSample:
    track_id: int
    frame: int
    timestamp: float
    x_raw: float
    y_raw: float
    calibration_error: float
    calibration_valid: bool


def reject_jumps(samples: list[PitchSample]) -> tuple[list[PitchSample], int]:
    """Drop a sample that a player cannot have reached from the previous kept sample."""
    ordered = sorted(samples, key=lambda sample: sample.frame)
    kept: list[PitchSample] = []
    removed = 0
    for sample in ordered:
        if not kept:
            kept.append(sample)
            continue
        previous = kept[-1]
        dt = sample.timestamp - previous.timestamp
        if dt <= 0:
            removed += 1
            continue
        jump = float(np.hypot(sample.x_raw - previous.x_raw, sample.y_raw - previous.y_raw))
        limit = MAX_SPEED_YARDS_PER_SECOND * dt + JUMP_NOISE_YARDS
        if jump > limit:
            removed += 1
            continue
        kept.append(sample)
    return kept, removed


def smooth_track(samples: list[PitchSample]) -> list[dict]:
    """Smooth each contiguous stretch. A cut is not treated as the next sample."""
    if not samples:
        return []
    rows: list[dict] = []
    for segment in _segments(samples):
        rows.extend(_smooth_segment(segment))
    return rows


def _segments(samples: list[PitchSample]) -> list[list[PitchSample]]:
    segments: list[list[PitchSample]] = [[samples[0]]]
    for sample in samples[1:]:
        if sample.timestamp - segments[-1][-1].timestamp > MAX_GAP_SECONDS:
            segments.append([sample])
        else:
            segments[-1].append(sample)
    return segments


def _smooth_segment(samples: list[PitchSample]) -> list[dict]:
    count = len(samples)
    window = min(SAVGOL_WINDOW, count if count % 2 == 1 else count - 1)
    if window <= SAVGOL_POLYORDER:
        return []
    x_raw = np.array([sample.x_raw for sample in samples], dtype=np.float64)
    y_raw = np.array([sample.y_raw for sample in samples], dtype=np.float64)
    if window == count and count < SAVGOL_WINDOW:
        # Too short for the chosen window. A shorter odd window still smooths the stretch.
        pass
    x_smooth = savgol_filter(x_raw, window_length=window, polyorder=SAVGOL_POLYORDER, mode="interp")
    y_smooth = savgol_filter(y_raw, window_length=window, polyorder=SAVGOL_POLYORDER, mode="interp")
    return [
        {
            "track_id": sample.track_id,
            "frame": sample.frame,
            "timestamp": round(sample.timestamp, 3),
            "x_raw": round(sample.x_raw, 2),
            "y_raw": round(sample.y_raw, 2),
            "x_smooth": round(float(x_smooth[index]), 2),
            "y_smooth": round(float(y_smooth[index]), 2),
            "calibration_error": round(sample.calibration_error, 3),
            "calibration_valid": sample.calibration_valid,
        }
        for index, sample in enumerate(samples)
    ]
