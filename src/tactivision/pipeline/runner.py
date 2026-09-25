"""Pipeline stages. The web app consumes this output; it does not own it."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from tactivision.detection.detector import PlayerDetector
from tactivision.detection.schemas import FrameDetections
from tactivision.detection.visualize import color_for_track, draw_detections, draw_tracks
from tactivision.spatial.trajectories import build_trajectories, write_trajectories
from tactivision.tracking.schemas import FrameTracks
from tactivision.tracking.tracker import PlayerTracker
from tactivision.tracking.shots import ShotDetector, finalize_shots, find_camera_labels, load_soccernet_boundaries
from tactivision.video.frame_extractor import extract_frames
from tactivision.video.reader import VideoReader


def run_ingestion(
    input_path: Path | str,
    output_root: Path | str = Path("data/processed"),
    *,
    run_id: str | None = None,
    extract: bool = False,
    sample_fps: float = 1.0,
) -> Path:
    """Create ``output_root/<run_id>/metadata.json``.

    ``run_id`` defaults to the video stem. Frames are written only when
    ``extract`` is true, at ``sample_fps``. Does not copy the source video.
    """
    from tactivision.pipeline.paths import make_run_paths, project_relative

    source = Path(input_path)
    rid = run_id or source.stem
    paths = make_run_paths(output_root, rid)
    project_root = Path(__file__).resolve().parents[3]

    with VideoReader(source) as reader:
        meta = reader.metadata()
        payload = meta.to_dict()
        payload["run_id"] = rid
        payload["source"] = project_relative(source, project_root)
        payload["source_filename"] = source.name
        (paths.root / "metadata.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        if extract:
            extract_frames(reader, paths.root / "frames", sample_fps=sample_fps)

    return paths.root


def run_detection(
    input_path: Path | str,
    output_root: Path | str = Path("data/processed"),
    *,
    model_path: str = "models/detection/yolo11n.pt",
    confidence: float = 0.35,
    image_size: int = 1280,
    sample_fps: float = 1.0,
    max_seconds: float | None = None,
    visualize_limit: int = 8,
    device: str | None = None,
) -> Path:
    """Detect players on sampled frames and write detections.jsonl.

    ``max_seconds`` caps the clip. Omit it only when you mean to scan the match.
    """
    source = Path(input_path)
    run_dir = run_ingestion(source, output_root, extract=False)
    detections_path = run_dir / "detections.jsonl"
    vis_dir = run_dir / "visualizations"
    if visualize_limit > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    detector = PlayerDetector(
        model_path,
        confidence=confidence,
        image_size=image_size,
        device=device,
    )
    saved_vis = 0
    frame_rows = 0
    player_count = 0

    with VideoReader(source) as reader, detections_path.open("w", encoding="utf-8") as handle:
        for frame_id, timestamp, frame in reader.iter_frames(sample_fps=sample_fps):
            if max_seconds is not None and timestamp >= max_seconds:
                break
            found = detector.detect(frame)
            row = FrameDetections(frame_id, timestamp, found)
            handle.write(json.dumps(row.to_dict()) + "\n")
            frame_rows += 1
            player_count += len(found)
            if saved_vis < visualize_limit and found:
                image = draw_detections(frame, found, timestamp)
                vis_path = vis_dir / f"frame_{frame_id:06d}.jpg"
                if not cv2.imwrite(str(vis_path), image):
                    raise OSError(f"Failed to write visualization: {vis_path}")
                saved_vis += 1

    summary = {
        "video": source.name,
        "model": model_path,
        "device": detector.device,
        "sample_fps": sample_fps,
        "max_seconds": max_seconds,
        "frames": frame_rows,
        "players": player_count,
        "visualizations": saved_vis,
    }
    (run_dir / "detection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    for ext, fourcc_name in (("mp4", "mp4v"), ("avi", "MJPG")):
        candidate = path.with_suffix(f".{ext}")
        writer = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*fourcc_name), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise OSError(f"Could not open a video writer at {path}")


def run_tracking(
    input_path: Path | str,
    output_root: Path | str = Path("data/processed"),
    *,
    model_path: str = "models/detection/yolo11n.pt",
    tracker_name: str = "bytetrack.yaml",
    confidence: float = 0.35,
    image_size: int = 1280,
    max_seconds: float | None = 30.0,
    still_every_seconds: float = 5.0,
    device: str | None = None,
) -> Path:
    """Track players on consecutive frames and write an annotated preview.

    Unlike detection, this does not subsample. ByteTrack needs neighboring frames.
    """
    source = Path(input_path)
    run_dir = run_ingestion(source, output_root, extract=False)
    tracker_dir = run_dir / Path(tracker_name).stem
    tracks_path = tracker_dir / "tracks.jsonl"
    still_dir = tracker_dir / "visualizations"
    still_dir.mkdir(parents=True, exist_ok=True)

    tracker = PlayerTracker(
        model_path,
        tracker_name=tracker_name,
        confidence=confidence,
        image_size=image_size,
        device=device,
    )
    tracker.reset()
    writer: cv2.VideoWriter | None = None
    frame_rows = 0
    spans: dict[int, list[float]] = {}
    next_still = 0.0

    with VideoReader(source) as reader, tracks_path.open("w", encoding="utf-8") as handle:
        info = reader.metadata()
        for frame_id, timestamp, frame in reader.iter_frames():
            if max_seconds is not None and timestamp >= max_seconds:
                break
            found = tracker.update(frame)
            handle.write(json.dumps(FrameTracks(frame_id, timestamp, found).to_dict()) + "\n")
            image = draw_tracks(frame, found, timestamp)
            if writer is None:
                writer = _open_writer(still_dir / "preview", info.fps or 25.0, (info.width, info.height))
            writer.write(image)
            for track in found:
                if track.track_id is None:
                    continue
                span = spans.setdefault(track.track_id, [timestamp, timestamp])
                span[1] = timestamp
            if timestamp + 1e-6 >= next_still:
                still = still_dir / f"frame_{frame_id:06d}.jpg"
                if not cv2.imwrite(str(still), image):
                    raise OSError(f"Failed to write track still: {still}")
                next_still += still_every_seconds
            frame_rows += 1
            if frame_rows % 100 == 0:
                print(f"tracked {frame_rows} frames, t={timestamp:.1f}s", flush=True)

    if writer is not None:
        writer.release()

    preview = next(still_dir.glob("preview.*"), None)
    summary = {
        "video": source.name,
        "model": model_path,
        "tracker": tracker_name,
        "device": tracker.device,
        "max_seconds": max_seconds,
        "frames": frame_rows,
        "unique_track_ids": len(spans),
        "ids_lasting_2s": sum(1 for start, end in spans.values() if end - start >= 2.0),
        "preview": None if preview is None else str(preview),
    }
    (tracker_dir / "tracking_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return tracker_dir


def run_shot_tracking(
    input_path: Path | str,
    output_root: Path | str = Path("data/processed"),
    *,
    model_path: str = "models/detection/yolo11n.pt",
    tracker_name: str = "bytetrack.yaml",
    confidence: float = 0.35,
    image_size: int = 1280,
    max_seconds: float | None = 30.0,
    still_every_seconds: float = 5.0,
    device: str | None = None,
    min_hist_distance: float = 0.45,
    half: int = 1,
) -> Path:
    """Track each continuous camera shot with a fresh ByteTrack state.

    Track IDs are not preserved across cuts. Association is ``(shot_id, track_id)``.
    SoccerNet ``Labels-cameras.json`` is preferred when present beside the video.
    """
    source = Path(input_path)
    run_dir = run_ingestion(source, output_root, extract=False)
    out_dir = run_dir / "bytetrack_shots"
    tracks_path = out_dir / "tracks.jsonl"
    still_dir = out_dir / "visualizations"
    still_dir.mkdir(parents=True, exist_ok=True)

    tracker = PlayerTracker(
        model_path,
        tracker_name=tracker_name,
        confidence=confidence,
        image_size=image_size,
        device=device,
    )
    labels_path = find_camera_labels(source)
    labeled = (
        load_soccernet_boundaries(labels_path, half=half, max_seconds=max_seconds)
        if labels_path is not None
        else []
    )
    use_labels = len(labeled) > 0
    detector = ShotDetector(min_hist_distance=min_hist_distance)
    tracker.reset()
    writer: cv2.VideoWriter | None = None
    frame_rows = 0
    shot_id = 0
    next_label_index = 1
    shot_starts: list[tuple[int, float, str | None]] = []
    spans: dict[tuple[int, int], list[float]] = {}
    next_still = 0.0
    last_frame = 0
    last_time = 0.0

    with VideoReader(source) as reader, tracks_path.open("w", encoding="utf-8") as handle:
        info = reader.metadata()
        for frame_id, timestamp, frame in reader.iter_frames():
            if max_seconds is not None and timestamp >= max_seconds:
                break
            if frame_rows == 0:
                start_label = labeled[0].label if labeled else "clip_start"
                shot_starts.append((frame_id, timestamp, start_label))
                if not use_labels:
                    detector.is_cut(frame, frame_id)
            else:
                cut = False
                label = None
                if use_labels:
                    while next_label_index < len(labeled) and timestamp + 1e-6 >= labeled[next_label_index].timestamp:
                        cut = True
                        label = labeled[next_label_index].label
                        next_label_index += 1
                elif detector.is_cut(frame, frame_id):
                    cut = True
                    label = "histogram_cut"
                if cut:
                    shot_id += 1
                    shot_starts.append((frame_id, timestamp, label))
                    tracker.reset()
            found = tracker.update(frame)
            handle.write(json.dumps(FrameTracks(frame_id, timestamp, found, shot_id=shot_id).to_dict()) + "\n")
            image = draw_tracks(frame, found, timestamp)
            cv2.putText(
                image,
                f"shot={shot_id}",
                (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if writer is None:
                writer = _open_writer(still_dir / "preview", info.fps or 25.0, (info.width, info.height))
            writer.write(image)
            for track in found:
                if track.track_id is None:
                    continue
                key = (shot_id, track.track_id)
                span = spans.setdefault(key, [timestamp, timestamp])
                span[1] = timestamp
            if timestamp + 1e-6 >= next_still:
                still = still_dir / f"frame_{frame_id:06d}.jpg"
                if not cv2.imwrite(str(still), image):
                    raise OSError(f"Failed to write track still: {still}")
                next_still += still_every_seconds
            frame_rows += 1
            last_frame = frame_id
            last_time = timestamp
            if frame_rows % 100 == 0:
                print(f"shot-tracked {frame_rows} frames, shot={shot_id}, t={timestamp:.1f}s", flush=True)

    if writer is not None:
        writer.release()

    shots = finalize_shots(shot_starts, last_frame=last_frame, last_time=last_time)
    per_shot: dict[int, dict] = {}
    for (sid, _track_id), (start, end) in spans.items():
        bucket = per_shot.setdefault(sid, {"tracks": 0, "tracks_lasting_2s": 0})
        bucket["tracks"] += 1
        if end - start >= 2.0:
            bucket["tracks_lasting_2s"] += 1

    shot_rows = []
    for shot in shots:
        stats = per_shot.get(shot.shot_id, {"tracks": 0, "tracks_lasting_2s": 0})
        row = shot.to_dict()
        row.update(stats)
        shot_rows.append(row)

    lasting = sum(1 for start, end in spans.values() if end - start >= 2.0)
    preview = next(still_dir.glob("preview.*"), None)
    summary = {
        "video": source.name,
        "model": model_path,
        "tracker": tracker_name,
        "mode": "shot_aware",
        "shot_source": "soccernet_labels" if use_labels else "histogram",
        "labels_path": None if labels_path is None else str(labels_path),
        "device": tracker.device,
        "max_seconds": max_seconds,
        "frames": frame_rows,
        "shots": len(shots),
        "unique_track_ids": len(spans),
        "ids_lasting_2s": lasting,
        "shot_table": shot_rows,
        "preview": None if preview is None else str(preview),
        "baseline_comparison": {
            "continuous_bytetrack_unique_ids": 223,
            "continuous_bytetrack_ids_lasting_2s": 76,
            "shot_aware_unique_ids": len(spans),
            "shot_aware_ids_lasting_2s": lasting,
            "within_shot_fragmentation_reduced": len(spans) < 223 and lasting >= 76,
            "note": (
                "SoccerNet Labels-cameras.json has no first-half camera change before 33.6s. "
                "This 30-second clip is one continuous shot, so shot resets do not reduce ID count."
                if use_labels and len(shots) <= 1
                else "Compare continuous ByteTrack against shot-aware resets."
            ),
        },
    }
    (out_dir / "shots.json").write_text(json.dumps(shot_rows, indent=2) + "\n", encoding="utf-8")
    (out_dir / "tracking_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out_dir


def run_trajectories(
    video_path: Path | str,
    tracks_path: Path | str,
    output_dir: Path | str,
    *,
    min_seconds: float = 2.0,
) -> Path:
    """Draw image-space trails for tracks that survive cuts and last long enough."""
    source = Path(video_path)
    tracks_file = Path(tracks_path)
    out = Path(output_dir)
    vis_dir = out / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in tracks_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    trajectories = build_trajectories(rows, min_seconds=min_seconds)
    write_trajectories(out / "trajectories.json", trajectories, min_seconds=min_seconds)

    writer: cv2.VideoWriter | None = None
    still_path = vis_dir / "trajectories.jpg"
    with VideoReader(source) as reader:
        info = reader.metadata()
        for frame_id, timestamp, frame in reader.iter_frames():
            if timestamp >= 30.0:
                break
            canvas = frame.copy()
            for trajectory in trajectories:
                visible = [point for point in trajectory.points if point.timestamp <= timestamp + 1e-6]
                if len(visible) < 2:
                    continue
                color = color_for_track(trajectory.track_id)
                pts = [(int(point.x), int(point.y)) for point in visible]
                for start, end in zip(pts, pts[1:], strict=False):
                    cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
                cv2.circle(canvas, pts[-1], 4, color, -1)
            cv2.putText(
                canvas,
                f"t={timestamp:.2f}s  paths={sum(1 for item in trajectories if item.points[0].timestamp <= timestamp)}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if writer is None:
                writer = _open_writer(vis_dir / "trajectories", info.fps or 25.0, (info.width, info.height))
            writer.write(canvas)
            if still_path.exists() is False and timestamp >= 10.0:
                if not cv2.imwrite(str(still_path), canvas):
                    raise OSError(f"Failed to write {still_path}")
    if writer is not None:
        writer.release()
    return out
