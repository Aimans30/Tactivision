"""Canonical run-directory layout for a processed match clip."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_run_id(run_id: str) -> str:
    text = run_id.strip()
    if not _RUN_ID_RE.match(text):
        raise ValueError(
            "run_id must be 1–64 chars: letters, digits, '.', '_', '-' "
            f"(got {run_id!r})"
        )
    if text in {".", ".."}:
        raise ValueError(f"invalid run_id: {run_id!r}")
    return text


def project_relative(path: Path, project_root: Path) -> str:
    """Return a portable project-relative path, or basename if outside the repo."""
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        return str(resolved.relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


@dataclass(frozen=True)
class RunPaths:
    """Artifact paths under ``data/processed/<run_id>/``."""

    root: Path

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def detections(self) -> Path:
        return self.root / "detections.jsonl"

    @property
    def tracks(self) -> Path:
        return self.root / "tracks.jsonl"

    @property
    def tracking_summary(self) -> Path:
        return self.root / "tracking_summary.json"

    @property
    def tracks_preview_dir(self) -> Path:
        return self.root / "visualizations" / "tracks"

    @property
    def calibration_dir(self) -> Path:
        return self.root / "calibration" / "per_frame"

    @property
    def calibration(self) -> Path:
        return self.calibration_dir / "calibration.jsonl"

    @property
    def pitch_positions(self) -> Path:
        return self.root / "pitch_positions.jsonl"

    @property
    def player_filter(self) -> Path:
        return self.root / "player_filter.json"

    @property
    def pitch_trajectories(self) -> Path:
        return self.root / "pitch_trajectories.jsonl"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def ball_dir(self) -> Path:
        return self.root / "ball"

    @property
    def ball_trajectory(self) -> Path:
        return self.ball_dir / "ball_trajectory.jsonl"

    @property
    def ball_pitch(self) -> Path:
        return self.ball_dir / "ball_pitch_trajectory_per_frame.jsonl"

    @property
    def match_bundle(self) -> Path:
        return self.root / "match_bundle.json"

    @property
    def pipeline_summary(self) -> Path:
        return self.root / "pipeline_summary.json"


def make_run_paths(processed_root: Path | str, run_id: str) -> RunPaths:
    rid = validate_run_id(run_id)
    root = Path(processed_root) / rid
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(root=root)
