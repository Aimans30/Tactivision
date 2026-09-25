"""End-to-end match pipeline for an arbitrary football video.

Produces ``data/processed/<run_id>/`` artifacts consumed by the API/dashboard.
Does not enable appearance remapping. Does not overwrite unrelated runs.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tactivision.analytics.bundle import write_match_bundle
from tactivision.ball.detector import BallDetector
from tactivision.ball.schemas import FrameBallDetections
from tactivision.ball.tracker import BallTracker
from tactivision.calibration.detector import PitchCalibrator
from tactivision.calibration.homography import (
    estimate_homography,
    mean_reprojection_error,
    transform_points,
)
from tactivision.calibration.project_players import (
    build_pitch_position_row,
    summarize_pitch_positions,
)
from tactivision.detection.detector import PlayerDetector
from tactivision.detection.schemas import FrameDetections
from tactivision.detection.visualize import draw_detections, draw_tracks
from tactivision.filtering.players import assign_teams, kit_feature, pitch_keep, torso_color
from tactivision.pipeline.paths import RunPaths, make_run_paths, project_relative, validate_run_id
from tactivision.pipeline.runner import _open_writer
from tactivision.spatial.clean import PitchSample, reject_jumps, smooth_track
from tactivision.spatial.metrics import measure_track, summarize as summarize_player_metrics
from tactivision.spatial.possession import (
    DEBOUNCE_FRAMES,
    POSSESSION_RADIUS_YARDS,
    build_possession_rows,
    index_players,
    summarize_possession,
)
from tactivision.spatial.team_shape import MIN_PLAYERS, build_team_shape_rows, summarize_team_shape
from tactivision.tracking.schemas import FrameTracks
from tactivision.tracking.tracker import PlayerTracker
from tactivision.video.metadata import VideoMetadata, write_metadata
from tactivision.video.reader import VideoReader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_PLAYER_MODEL = "models/detection/yolo11n.pt"
DEFAULT_PITCH_MODEL = PROJECT_ROOT / "models" / "calibration" / "yolo-football-pitch-detection.pt"
DEFAULT_BALL_MODEL = PROJECT_ROOT / "models" / "ball" / "yolo-sn-ball.pt"

POSSESSION_COLUMNS = [
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

PLAYER_METRIC_COLUMNS = [
    "track_id",
    "duration_seconds",
    "distance_yards",
    "mean_speed_yards_per_second",
    "max_speed_yards_per_second",
    "valid_samples",
    "distance_yards_raw",
    "mean_speed_yards_per_second_raw",
    "max_speed_yards_per_second_raw",
    "metric_type",
]

TEAM_SHAPE_COLUMNS = [
    "frame",
    "timestamp",
    "metric_type",
    "coordinate_system",
    "coordinates",
    "min_players_required",
    "team0_valid",
    "team0_n_available",
    "team0_n_players",
    "team0_centroid_x",
    "team0_centroid_y",
    "team0_width",
    "team0_length",
    "team0_compactness",
    "team1_valid",
    "team1_n_available",
    "team1_n_players",
    "team1_centroid_x",
    "team1_centroid_y",
    "team1_width",
    "team1_length",
    "team1_compactness",
]


def _rel(path: Path) -> str:
    return project_relative(path, PROJECT_ROOT)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ingest_video(
    video: Path,
    paths: RunPaths,
    *,
    run_id: str,
) -> VideoMetadata:
    """Write metadata.json; does not copy the source video."""
    with VideoReader(video) as reader:
        meta = reader.metadata()
    payload = meta.to_dict()
    payload["run_id"] = run_id
    payload["source"] = _rel(video)
    payload["source_filename"] = video.name
    _write_json(paths.metadata, payload)
    return meta


def stage_detection(
    video: Path,
    paths: RunPaths,
    *,
    model_path: str,
    confidence: float,
    image_size: int,
    max_seconds: float | None,
    sample_fps: float,
    device: str | None,
) -> dict:
    detector = PlayerDetector(model_path, confidence=confidence, image_size=image_size, device=device)
    frame_rows = 0
    player_count = 0
    vis_dir = paths.root / "visualizations"
    saved_vis = 0
    with VideoReader(video) as reader, paths.detections.open("w", encoding="utf-8") as handle:
        for frame_id, timestamp, frame in reader.iter_frames(sample_fps=sample_fps):
            if max_seconds is not None and timestamp >= max_seconds:
                break
            found = detector.detect(frame)
            handle.write(json.dumps(FrameDetections(frame_id, timestamp, found).to_dict()) + "\n")
            frame_rows += 1
            player_count += len(found)
            if saved_vis < 4 and found:
                vis_dir.mkdir(parents=True, exist_ok=True)
                image = draw_detections(frame, found, timestamp)
                cv2.imwrite(str(vis_dir / f"det_{frame_id:06d}.jpg"), image)
                saved_vis += 1
    summary = {
        "video": video.name,
        "model": model_path,
        "device": detector.device,
        "sample_fps": sample_fps,
        "max_seconds": max_seconds,
        "frames": frame_rows,
        "players": player_count,
        "visualizations": saved_vis,
    }
    _write_json(paths.root / "detection_summary.json", summary)
    return summary


def stage_tracking(
    video: Path,
    paths: RunPaths,
    *,
    model_path: str,
    tracker_name: str,
    confidence: float,
    image_size: int,
    max_seconds: float | None,
    device: str | None,
    write_preview: bool,
) -> dict:
    tracker = PlayerTracker(
        model_path,
        tracker_name=tracker_name,
        confidence=confidence,
        image_size=image_size,
        device=device,
    )
    tracker.reset()
    paths.tracks_preview_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    frame_rows = 0
    track_boxes = 0
    spans: dict[int, list[float]] = {}
    with VideoReader(video) as reader, paths.tracks.open("w", encoding="utf-8") as handle:
        info = reader.metadata()
        for frame_id, timestamp, frame in reader.iter_frames():
            if max_seconds is not None and timestamp >= max_seconds:
                break
            found = tracker.update(frame)
            handle.write(json.dumps(FrameTracks(frame_id, timestamp, found).to_dict()) + "\n")
            track_boxes += len(found)
            for track in found:
                if track.track_id is None:
                    continue
                span = spans.setdefault(int(track.track_id), [timestamp, timestamp])
                span[1] = timestamp
            if write_preview:
                image = draw_tracks(frame, found, timestamp)
                if writer is None:
                    writer = _open_writer(
                        paths.tracks_preview_dir / "preview",
                        info.fps or 25.0,
                        (info.width, info.height),
                    )
                writer.write(image)
            frame_rows += 1
            if frame_rows % 100 == 0:
                print(f"  tracking {frame_rows} frames...", flush=True)
    if writer is not None:
        writer.release()
    preview = next(paths.tracks_preview_dir.glob("preview.*"), None)
    summary = {
        "video": video.name,
        "model": model_path,
        "tracker": tracker_name,
        "device": tracker.device,
        "max_seconds": max_seconds,
        "frames": frame_rows,
        "track_boxes": track_boxes,
        "unique_track_ids": len(spans),
        "preview": None if preview is None else _rel(preview),
    }
    _write_json(paths.tracking_summary, summary)
    return summary


def stage_calibration(
    video: Path,
    paths: RunPaths,
    *,
    model_path: Path,
    max_seconds: float | None,
    max_error: float = 3.0,
    min_correspondences: int = 4,
) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(f"pitch calibration model not found: {model_path}")
    paths.calibration_dir.mkdir(parents=True, exist_ok=True)
    calibrator = PitchCalibrator(model_path)
    rows: list[dict] = []
    with VideoReader(video) as reader, paths.calibration.open("w", encoding="utf-8") as handle:
        for frame_id, timestamp, frame in reader.iter_frames():
            if max_seconds is not None and timestamp >= max_seconds:
                break
            record: dict[str, Any] = {
                "frame_id": frame_id,
                "timestamp": round(float(timestamp), 3),
                "has_sufficient_correspondences": False,
                "n_correspondences": 0,
                "n_inliers": 0,
                "calibration_valid": False,
                "reprojection_error": None,
                "homography": None,
                "failure_reason": None,
                "coordinate_system": "statsbomb_120x80_yards",
            }
            correspondences = calibrator.correspondences(frame)
            if correspondences is None:
                record["failure_reason"] = "insufficient_or_no_keypoints"
            else:
                image_points, pitch_points = correspondences
                n_corr = int(len(image_points))
                record["n_correspondences"] = n_corr
                if n_corr < min_correspondences:
                    record["failure_reason"] = "insufficient_correspondences"
                else:
                    record["has_sufficient_correspondences"] = True
                    homography, inliers = estimate_homography(image_points, pitch_points)
                    if (
                        homography is None
                        or inliers is None
                        or int(inliers.sum()) < min_correspondences
                    ):
                        record["failure_reason"] = "homography_estimation_failed"
                    else:
                        n_inliers = int(inliers.sum())
                        record["n_inliers"] = n_inliers
                        error = mean_reprojection_error(
                            image_points, pitch_points, homography, inliers
                        )
                        record["reprojection_error"] = round(float(error), 4)
                        if error > max_error:
                            record["failure_reason"] = "reprojection_error_above_threshold"
                        else:
                            record["calibration_valid"] = True
                            record["homography"] = [
                                [round(float(v), 8) for v in row] for row in homography.tolist()
                            ]
            handle.write(json.dumps(record) + "\n")
            rows.append(record)
            if (frame_id + 1) % 100 == 0:
                print(f"  calibrated {frame_id + 1} frames...", flush=True)
    valid = [r for r in rows if r["calibration_valid"]]
    errors = [float(r["reprojection_error"]) for r in valid if r["reprojection_error"] is not None]
    summary = {
        "frames": len(rows),
        "calibrated_frames": len(valid),
        "calibration_coverage_percent": round(100.0 * len(valid) / len(rows), 2) if rows else 0.0,
        "median_reprojection_error": (
            None if not errors else round(float(sorted(errors)[len(errors) // 2]), 4)
        ),
        "interpolation": False,
        "calibration_source": "per_frame",
        "output": _rel(paths.calibration),
    }
    _write_json(paths.calibration_dir / "calibration_summary.json", summary)
    return summary


def stage_project_players(paths: RunPaths, *, max_seconds: float | None) -> dict:
    track_rows = _load_jsonl(paths.tracks)
    cal_by_frame = {int(r["frame_id"]): r for r in _load_jsonl(paths.calibration)}
    out_rows: list[dict] = []
    for row in track_rows:
        timestamp = float(row["timestamp"])
        if max_seconds is not None and timestamp >= max_seconds:
            break
        out_rows.append(
            build_pitch_position_row(
                frame_id=int(row["frame_id"]),
                timestamp=timestamp,
                tracks=row.get("tracks") or [],
                calibration=cal_by_frame.get(int(row["frame_id"])),
            )
        )
    with paths.pitch_positions.open("w", encoding="utf-8") as handle:
        for record in out_rows:
            handle.write(json.dumps(record) + "\n")
    summary = summarize_pitch_positions(out_rows)
    summary["tracks_source"] = _rel(paths.tracks)
    summary["calibration_source_path"] = _rel(paths.calibration)
    summary["output"] = _rel(paths.pitch_positions)
    _write_json(paths.root / "pitch_positions_summary.json", summary)
    pointer = {
        "coordinate_system": "statsbomb_120x80_yards",
        "calibration_source": "per_frame",
        "canonical": True,
        "per_frame_calibration": _rel(paths.calibration),
        "pitch_positions": _rel(paths.pitch_positions),
        "calibrated_frames": summary["calibrated_frames"],
        "total_frames": summary["total_frames"],
        "player_coordinate_samples": summary["player_coordinate_samples"],
        "interpolation": False,
    }
    _write_json(paths.root / "calibration_summary.json", pointer)
    return summary


def stage_filter_players(
    video: Path,
    paths: RunPaths,
    *,
    max_seconds: float | None,
) -> dict:
    track_rows = _load_jsonl(paths.tracks)
    position_rows = _load_jsonl(paths.pitch_positions)
    track_ids = {
        int(track["track_id"])
        for row in track_rows
        if max_seconds is None or float(row["timestamp"]) < max_seconds
        for track in row.get("tracks") or []
        if track.get("track_id") is not None
    }
    positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in position_rows:
        if not row.get("calibrated"):
            continue
        if max_seconds is not None and float(row["timestamp"]) >= max_seconds:
            continue
        for player in row.get("players") or []:
            positions[int(player["track_id"])].append((float(player["x"]), float(player["y"])))

    boxes: dict[tuple[int, int], list[float]] = {}
    for row in track_rows:
        if max_seconds is not None and float(row["timestamp"]) >= max_seconds:
            continue
        for track in row.get("tracks") or []:
            if track.get("track_id") is None:
                continue
            boxes[(int(row["frame_id"]), int(track["track_id"]))] = track["bbox"]

    pitch_results = [pitch_keep(track_id, positions.get(track_id, [])) for track_id in sorted(track_ids)]
    on_pitch = [item for item in pitch_results if item.kept]
    inside_frames: dict[int, list[int]] = defaultdict(list)
    for row in position_rows:
        if not row.get("calibrated"):
            continue
        if max_seconds is not None and float(row["timestamp"]) >= max_seconds:
            continue
        frame_id = int(row["frame_id"])
        for player in row.get("players") or []:
            track_id = int(player["track_id"])
            if not (0 <= float(player["x"]) <= 120 and 0 <= float(player["y"]) <= 80):
                continue
            if (frame_id, track_id) in boxes:
                inside_frames[track_id].append(frame_id)
    sample_frames = {track_id: frames[:8] for track_id, frames in inside_frames.items()}
    needed = {frame_id for frames in sample_frames.values() for frame_id in frames}
    colors: dict[int, list[np.ndarray]] = defaultdict(list)
    if needed:
        with VideoReader(video) as reader:
            for frame_id, timestamp, frame in reader.iter_frames():
                if max_seconds is not None and timestamp >= max_seconds:
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
    results = [by_id[tid] for tid in sorted(track_ids)] if track_ids else []
    kept = [item for item in results if item.kept]
    removed = [item for item in results if not item.kept]
    reasons: dict[str, int] = defaultdict(int)
    for item in removed:
        reasons[item.reason] += 1
    summary = {
        "tracks_before": len(results),
        "tracks_removed": len(removed),
        "tracks_remaining": len(kept),
        "removed_by_reason": dict(reasons),
        "team_counts": {
            str(team): sum(1 for item in kept if item.team == team) for team in (0, 1)
        },
    }
    _write_json(paths.player_filter, [item.to_dict() for item in results])
    _write_json(paths.root / "player_filter_summary.json", summary)
    return summary


def stage_clean_trajectories(paths: RunPaths) -> dict:
    filter_rows = json.loads(paths.player_filter.read_text(encoding="utf-8")) if paths.player_filter.is_file() else []
    kept_ids = {int(row["track_id"]) for row in filter_rows if row.get("kept")}
    by_track: dict[int, list[PitchSample]] = defaultdict(list)
    for row in _load_jsonl(paths.pitch_positions):
        if not row.get("calibrated"):
            continue
        for player in row.get("players") or []:
            track_id = int(player["track_id"])
            if track_id not in kept_ids:
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
    for track_id in sorted(by_track):
        kept, removed = reject_jumps(by_track[track_id])
        jumps_removed += removed
        rows.extend(smooth_track(kept))
    with paths.pitch_trajectories.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "tracks": len(by_track),
        "samples": len(rows),
        "jumps_removed": jumps_removed,
        "output": _rel(paths.pitch_trajectories),
    }
    _write_json(paths.root / "pitch_trajectories_summary.json", summary)
    return summary


def stage_player_metrics(paths: RunPaths) -> dict:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in _load_jsonl(paths.pitch_trajectories):
        by_track[int(row["track_id"])].append(row)
    measured = []
    for track_id in sorted(by_track):
        row = measure_track(by_track[track_id])
        if row is not None:
            measured.append(row)
    out = paths.metrics_dir / "player_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAYER_METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(measured)
    summary = summarize_player_metrics(measured)
    _write_json(paths.metrics_dir / "player_metrics_summary.json", summary)
    return summary


def stage_team_shape(paths: RunPaths) -> dict:
    traj_rows = _load_jsonl(paths.pitch_trajectories)
    filters = json.loads(paths.player_filter.read_text(encoding="utf-8")) if paths.player_filter.is_file() else []
    team_by_track = {
        int(row["track_id"]): int(row["team"])
        for row in filters
        if row.get("kept") and row.get("team") is not None
    }
    rows = build_team_shape_rows(traj_rows, team_by_track, min_players=MIN_PLAYERS)
    out = paths.metrics_dir / "team_shape.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEAM_SHAPE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_team_shape(rows, team_by_track)
    summary["sources"] = {
        "trajectories": _rel(paths.pitch_trajectories),
        "player_filter": _rel(paths.player_filter),
    }
    _write_json(paths.metrics_dir / "team_shape_summary.json", summary)
    return summary


def stage_ball(
    video: Path,
    paths: RunPaths,
    *,
    model_path: Path,
    max_seconds: float | None,
    confidence: float = 0.25,
    image_size: int = 1280,
    device: str | None = None,
    write_preview: bool = True,
) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(f"ball model not found: {model_path}")
    paths.ball_dir.mkdir(parents=True, exist_ok=True)
    detector = BallDetector(model_path, confidence=confidence, image_size=image_size, device=device)
    tracker = BallTracker()
    det_path = paths.ball_dir / "ball_detections.jsonl"
    writer = None
    detections_rows: list[dict] = []
    observations: list[dict] = []
    with VideoReader(video) as reader, det_path.open("w", encoding="utf-8") as det_f, paths.ball_trajectory.open(
        "w", encoding="utf-8"
    ) as traj_f:
        info = reader.metadata()
        fps = info.fps or 25.0
        for frame_id, timestamp, frame in reader.iter_frames():
            if max_seconds is not None and timestamp >= max_seconds:
                break
            dets = detector.detect(frame)
            det_row = FrameBallDetections(frame_id, timestamp, dets).to_dict()
            det_f.write(json.dumps(det_row) + "\n")
            detections_rows.append(det_row)
            obs = tracker.update(frame_id, timestamp, dets)
            obs_row = obs.to_dict()
            traj_f.write(json.dumps(obs_row) + "\n")
            observations.append(obs_row)
            if write_preview:
                canvas = frame.copy()
                if obs.observed and obs.bbox is not None:
                    x1, y1, x2, y2 = (int(v) for v in obs.bbox)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 255), 2)
                if writer is None:
                    writer = _open_writer(paths.ball_dir / "ball_preview", fps, (info.width, info.height))
                writer.write(canvas)
            if len(observations) % 100 == 0:
                print(f"  ball-tracked {len(observations)} frames...", flush=True)
    if writer is not None:
        writer.release()
    observed = [r for r in observations if r.get("observed")]
    summary = {
        "frames_processed": len(observations),
        "frames_with_tracked_observation": len(observed),
        "detection_coverage_percent": (
            round(100.0 * len(observed) / len(observations), 2) if observations else 0.0
        ),
        "preview": _rel(next(paths.ball_dir.glob("ball_preview.*"), paths.ball_dir / "ball_preview.mp4")),
    }
    _write_json(paths.ball_dir / "ball_tracking_summary.json", summary)
    return summary


def stage_project_ball(paths: RunPaths) -> dict:
    ball_rows = _load_jsonl(paths.ball_trajectory)
    cal_by_frame = {int(r["frame_id"]): r for r in _load_jsonl(paths.calibration)}
    out_rows: list[dict] = []
    for row in ball_rows:
        frame_id = int(row["frame_id"])
        timestamp = float(row["timestamp"])
        observed = bool(row.get("observed"))
        cal = cal_by_frame.get(frame_id)
        calibration_valid = bool(cal and cal.get("calibration_valid") and cal.get("homography"))
        record: dict[str, Any] = {
            "frame": frame_id,
            "timestamp": round(timestamp, 3),
            "observed": observed,
            "calibration_valid": calibration_valid,
            "pitch_valid": False,
            "image_x": None,
            "image_y": None,
            "pitch_x": None,
            "pitch_y": None,
            "track_id": row.get("track_id"),
            "confidence": row.get("confidence"),
            "metric_type": "estimated_model_derived",
            "coordinate_system": "statsbomb_120x80_yards",
            "calibration_source": "per_frame",
        }
        if observed and row.get("center") and len(row["center"]) == 2 and calibration_valid:
            image_x, image_y = float(row["center"][0]), float(row["center"][1])
            record["image_x"] = round(image_x, 1)
            record["image_y"] = round(image_y, 1)
            H = np.asarray(cal["homography"], dtype=np.float64)
            projected = transform_points(np.array([[image_x, image_y]], dtype=np.float32), H)[0]
            record["pitch_valid"] = True
            record["pitch_x"] = round(float(projected[0]), 3)
            record["pitch_y"] = round(float(projected[1]), 3)
        out_rows.append(record)
    with paths.ball_pitch.open("w", encoding="utf-8") as handle:
        for record in out_rows:
            handle.write(json.dumps(record) + "\n")
    observed = [r for r in out_rows if r["observed"]]
    projected = [r for r in out_rows if r["pitch_valid"]]
    summary = {
        "total_frames": len(out_rows),
        "observed_ball_frames": len(observed),
        "valid_ball_pitch_projections": len(projected),
        "projection_percentage_of_observed": (
            round(100.0 * len(projected) / len(observed), 2) if observed else 0.0
        ),
        "calibration_source": _rel(paths.calibration),
        "output": _rel(paths.ball_pitch),
        "interpolation": False,
    }
    _write_json(paths.ball_dir / "ball_pitch_projection_per_frame_summary.json", summary)
    return summary


def stage_possession(paths: RunPaths) -> dict:
    ball_rows = _load_jsonl(paths.ball_pitch)
    traj_rows = _load_jsonl(paths.pitch_trajectories)
    filters = json.loads(paths.player_filter.read_text(encoding="utf-8")) if paths.player_filter.is_file() else []
    team_by_track = {
        int(r["track_id"]): int(r["team"])
        for r in filters
        if r.get("kept") and r.get("team") is not None
    }
    players_by_frame = index_players(traj_rows, team_by_track)
    rows = build_possession_rows(
        ball_rows=ball_rows,
        players_by_frame=players_by_frame,
        radius=POSSESSION_RADIUS_YARDS,
        debounce_frames=DEBOUNCE_FRAMES,
    )
    out = paths.metrics_dir / "possession.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POSSESSION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") if row.get(k) is not None else "" for k in POSSESSION_COLUMNS})
    summary = summarize_possession(rows, radius=POSSESSION_RADIUS_YARDS, debounce_frames=DEBOUNCE_FRAMES)
    summary["sources"] = {
        "ball": _rel(paths.ball_pitch),
        "trajectories": _rel(paths.pitch_trajectories),
        "player_filter": _rel(paths.player_filter),
    }
    summary["player_frames_with_samples"] = len(players_by_frame)
    _write_json(paths.metrics_dir / "possession_summary.json", summary)
    viz = paths.root / "visualizations" / "possession"
    viz.mkdir(parents=True, exist_ok=True)
    # Minimal timeline image (skip plot if no rows).
    if rows:
        try:
            import matplotlib.pyplot as plt

            times = np.array([r["timestamp"] for r in rows], dtype=float)
            values = []
            for r in rows:
                status = r["possession_status"]
                if status == "assigned" and r["possession_team"] == 0:
                    values.append(0)
                elif status == "assigned" and r["possession_team"] == 1:
                    values.append(1)
                elif status == "unknown":
                    values.append(-1)
                else:
                    values.append(-2)
            fig, ax = plt.subplots(figsize=(10, 2.8), dpi=120)
            ax.plot(times, values, linewidth=1.0)
            ax.set_yticks([-2, -1, 0, 1])
            ax.set_yticklabels(["unavailable", "unknown", "team 0", "team 1"])
            ax.set_title("Estimated possession (heuristic)")
            fig.tight_layout()
            fig.savefig(viz / "possession_timeline.png")
            plt.close(fig)
        except Exception:
            pass
    return summary


def run_match_pipeline(
    video: Path | str,
    run_id: str,
    *,
    processed_root: Path | str = DEFAULT_PROCESSED,
    max_seconds: float | None = None,
    player_model: str = DEFAULT_PLAYER_MODEL,
    pitch_model: Path | str = DEFAULT_PITCH_MODEL,
    ball_model: Path | str = DEFAULT_BALL_MODEL,
    tracker_name: str = "bytetrack.yaml",
    confidence: float = 0.35,
    image_size: int = 1280,
    detection_sample_fps: float = 1.0,
    device: str | None = None,
    write_previews: bool = True,
    skip_detection: bool = False,
) -> dict:
    """Process ``video`` into ``processed_root/<run_id>/`` and return a summary."""
    source = Path(video)
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    rid = validate_run_id(run_id)
    paths = make_run_paths(processed_root, rid)
    t0 = time.perf_counter()
    print(f"=== ingest {source.name} -> {paths.root} ===", flush=True)
    meta = ingest_video(source, paths, run_id=rid)

    stages: dict[str, Any] = {"metadata": meta.to_dict()}

    if not skip_detection:
        print("=== detection ===", flush=True)
        stages["detection"] = stage_detection(
            source,
            paths,
            model_path=player_model,
            confidence=confidence,
            image_size=image_size,
            max_seconds=max_seconds,
            sample_fps=detection_sample_fps,
            device=device,
        )

    print("=== tracking (ByteTrack) ===", flush=True)
    stages["tracking"] = stage_tracking(
        source,
        paths,
        model_path=player_model,
        tracker_name=tracker_name,
        confidence=confidence,
        image_size=image_size,
        max_seconds=max_seconds,
        device=device,
        write_preview=write_previews,
    )

    print("=== per-frame calibration ===", flush=True)
    stages["calibration"] = stage_calibration(
        source,
        paths,
        model_path=Path(pitch_model),
        max_seconds=max_seconds,
    )

    print("=== player pitch projection ===", flush=True)
    stages["pitch_positions"] = stage_project_players(paths, max_seconds=max_seconds)

    print("=== player filter ===", flush=True)
    stages["player_filter"] = stage_filter_players(source, paths, max_seconds=max_seconds)

    print("=== trajectory cleaning ===", flush=True)
    stages["trajectories"] = stage_clean_trajectories(paths)

    print("=== player metrics ===", flush=True)
    stages["player_metrics"] = stage_player_metrics(paths)

    print("=== team shape ===", flush=True)
    stages["team_shape"] = stage_team_shape(paths)

    print("=== ball tracking ===", flush=True)
    stages["ball"] = stage_ball(
        source,
        paths,
        model_path=Path(ball_model),
        max_seconds=max_seconds,
        device=device,
        write_preview=write_previews,
    )

    print("=== ball pitch projection ===", flush=True)
    stages["ball_pitch"] = stage_project_ball(paths)

    print("=== possession ===", flush=True)
    stages["possession"] = stage_possession(paths)

    track_rows = _load_jsonl(paths.tracks)
    if track_rows:
        clip_seconds = float(track_rows[-1]["timestamp"]) + (1.0 / (meta.fps or 25.0))
    elif max_seconds is not None:
        clip_seconds = float(max_seconds)
    else:
        clip_seconds = float(meta.duration_seconds) if meta.duration_seconds else 0.0
    if max_seconds is not None:
        clip_seconds = min(clip_seconds, float(max_seconds))

    print("=== analytics bundle ===", flush=True)
    bundle = write_match_bundle(
        paths.root,
        clip_seconds=float(clip_seconds or 0.0),
        video_source=_rel(source),
    )
    stages["bundle"] = {
        "match_id": bundle.get("match_id"),
        "artifacts": list((bundle.get("artifacts") or {}).keys()),
        "summaries": list((bundle.get("summaries") or {}).keys()),
    }

    runtime = round(time.perf_counter() - t0, 2)
    summary = {
        "run_id": rid,
        "run_dir": _rel(paths.root),
        "video": _rel(source),
        "max_seconds": max_seconds,
        "runtime_seconds": runtime,
        "stages": stages,
        "match_bundle": _rel(paths.match_bundle),
    }
    _write_json(paths.pipeline_summary, summary)
    print(f"=== done in {runtime}s -> {paths.match_bundle} ===", flush=True)
    return summary
