"""Assemble a coherent match analytics bundle from existing + new outputs."""

from __future__ import annotations

from pathlib import Path

from tactivision.analytics.ball_metrics import summarize_ball
from tactivision.analytics.events import detect_events, summarize_events
from tactivision.analytics.formation import summarize_formation_over_time
from tactivision.analytics.io import (
    load_csv,
    load_json,
    load_jsonl,
    team_by_track_from_filter,
    write_json,
    write_jsonl,
)
from tactivision.analytics.schema import ESTIMATE, HEURISTIC
from tactivision.analytics.tactics import build_tactical_rows, summarize_tactics
from tactivision.spatial.possession import index_players


def build_match_bundle(
    run_dir: Path | str,
    *,
    clip_seconds: float = 30.0,
    video_source: str | None = None,
) -> dict:
    """Compute derived analytics and return an in-memory bundle + file plan."""
    root = Path(run_dir)
    metrics = root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)

    ball_path = root / "ball" / "ball_pitch_trajectory_per_frame.jsonl"
    traj_path = root / "pitch_trajectories.jsonl"
    filter_path = root / "player_filter.json"

    ball_rows = load_jsonl(ball_path) if ball_path.is_file() else []
    traj_rows = load_jsonl(traj_path) if traj_path.is_file() else []
    filter_rows = load_json(filter_path) if filter_path.is_file() else []
    if not isinstance(filter_rows, list):
        filter_rows = []
    possession_rows = (
        load_csv(metrics / "possession.csv") if (metrics / "possession.csv").is_file() else []
    )
    team_shape_rows = (
        load_csv(metrics / "team_shape.csv") if (metrics / "team_shape.csv").is_file() else []
    )

    team_by_track = team_by_track_from_filter(filter_rows)
    players_by_frame = index_players(traj_rows, team_by_track)
    timestamps = {int(r["frame"]): float(r["timestamp"]) for r in traj_rows if "frame" in r}

    ball_summary = summarize_ball(ball_rows)
    events = detect_events(possession_rows) if possession_rows else []
    events_summary = summarize_events(events)
    tactical_rows = build_tactical_rows(possession_rows, team_shape_rows) if possession_rows else []
    tactics_summary = summarize_tactics(tactical_rows)
    formation = summarize_formation_over_time(players_by_frame, timestamps)

    formation_frames = formation.pop("per_frame")

    artifacts = {
        "metadata": "metadata.json",
        "player_filter": "player_filter.json",
        "pitch_trajectories": "pitch_trajectories.jsonl",
        "player_metrics": "metrics/player_metrics.csv",
        "player_metrics_summary": "metrics/player_metrics_summary.json",
        "team_shape": "metrics/team_shape.csv",
        "team_shape_summary": "metrics/team_shape_summary.json",
        "possession": "metrics/possession.csv",
        "possession_summary": "metrics/possession_summary.json",
        "ball_pitch": "ball/ball_pitch_trajectory_per_frame.jsonl",
        "ball_tracking_summary": "ball/ball_tracking_summary.json",
        "ball_analytics_summary": "metrics/ball_analytics_summary.json",
        "events": "metrics/events.jsonl",
        "events_summary": "metrics/events_summary.json",
        "tactical_states": "metrics/tactical_states.jsonl",
        "tactical_states_summary": "metrics/tactical_states_summary.json",
        "formation_summary": "metrics/formation_summary.json",
        "formation_frames": "metrics/formation_frames.jsonl",
        "heatmaps": "visualizations/heatmaps/",
        "team_shape_viz": "visualizations/team_shape/",
        "possession_viz": "visualizations/possession/",
        "tracking_preview": "visualizations/tracks/preview.mp4",
        "trajectories_preview": "visualizations/trajectories.mp4",
        "ball_preview": "ball/ball_preview.mp4",
        "validation": "validation_report.json",
        "pipeline_summary": "pipeline_summary.json",
    }

    existing = {
        k: v
        for k, v in artifacts.items()
        if (root / v).exists()
        or k.startswith(("events", "tactical", "formation", "ball_analytics", "pipeline"))
    }

    metadata = load_json(root / "metadata.json") if (root / "metadata.json").exists() else {}
    if not isinstance(metadata, dict):
        metadata = {}
    possession_summary = (
        load_json(metrics / "possession_summary.json")
        if (metrics / "possession_summary.json").exists()
        else {}
    )
    player_metrics_summary = (
        load_json(metrics / "player_metrics_summary.json")
        if (metrics / "player_metrics_summary.json").exists()
        else {}
    )
    team_shape_summary = (
        load_json(metrics / "team_shape_summary.json")
        if (metrics / "team_shape_summary.json").exists()
        else {}
    )

    source = video_source
    if source is None:
        source = metadata.get("source") or metadata.get("video") or root.name
    if isinstance(source, str) and (
        ":\\" in source or source.startswith("/home/") or source.startswith("/Users/")
    ):
        source = Path(source).name

    bundle = {
        "match_id": root.name,
        "clip_seconds": clip_seconds,
        "coordinate_system": "statsbomb_120x80_yards",
        "provenance_policy": {
            "tracking": ESTIMATE,
            "calibration": ESTIMATE,
            "possession": HEURISTIC,
            "events": HEURISTIC,
            "tactics": HEURISTIC,
            "formation": HEURISTIC,
            "note": "No ground-truth labels for possession/events/tactics/formation on this clip.",
        },
        "video": {
            "source": source,
            "preview": artifacts["tracking_preview"],
            "fps": metadata.get("fps", 25.0),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
        },
        "artifacts": existing,
        "summaries": {
            "possession": possession_summary if isinstance(possession_summary, dict) else {},
            "player_metrics": player_metrics_summary if isinstance(player_metrics_summary, dict) else {},
            "team_shape": team_shape_summary if isinstance(team_shape_summary, dict) else {},
            "ball": ball_summary,
            "events": events_summary,
            "tactics": tactics_summary,
            "formation": {k: v for k, v in formation.items() if k != "per_frame"},
        },
        "limitations": [
            "Player tracks are fragmented ByteTrack IDs, not persistent identities.",
            "Possession/events/tactics/formation are heuristics, not official stats.",
            "Missing ball observations are never interpolated.",
            "Metrics are model-derived estimates, not tracking-device measurements.",
        ],
    }

    return {
        "ball_summary": ball_summary,
        "events": events,
        "events_summary": events_summary,
        "tactical_rows": tactical_rows,
        "tactics_summary": tactics_summary,
        "formation": formation,
        "formation_frames": formation_frames,
        "bundle": bundle,
    }


def write_match_bundle(
    run_dir: Path | str,
    *,
    clip_seconds: float = 30.0,
    video_source: str | None = None,
) -> dict:
    root = Path(run_dir)
    metrics = root / "metrics"
    computed = build_match_bundle(root, clip_seconds=clip_seconds, video_source=video_source)

    write_json(metrics / "ball_analytics_summary.json", computed["ball_summary"])
    write_jsonl(metrics / "events.jsonl", computed["events"])
    write_json(metrics / "events_summary.json", computed["events_summary"])
    write_jsonl(metrics / "tactical_states.jsonl", computed["tactical_rows"])
    write_json(metrics / "tactical_states_summary.json", computed["tactics_summary"])
    write_json(metrics / "formation_summary.json", computed["formation"])
    write_jsonl(metrics / "formation_frames.jsonl", computed["formation_frames"])
    write_json(root / "match_bundle.json", computed["bundle"])
    return computed["bundle"]
