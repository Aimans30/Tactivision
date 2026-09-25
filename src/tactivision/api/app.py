"""FastAPI surface over processed match analytics.

Serves cached CV/analytics artifacts; does not re-run detection/tracking.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tactivision.analytics.io import load_csv, load_json, load_jsonl
from tactivision.analytics.qa import answer_question, load_context
from tactivision.api.uploads import (
    ALLOWED_EXTENSIONS,
    DEFAULT_MAX_SECONDS,
    MAX_UPLOAD_BYTES,
    UPLOADS_ENABLED,
    ensure_run_slot_free,
    extension_allowed,
    new_upload_run_id,
    parse_max_seconds,
    public_error_message,
    sanitize_upload_filename,
    upload_destination,
    uploads_disabled_reason,
)
from tactivision.pipeline.match_pipeline import run_match_pipeline
from tactivision.video.reader import VideoReader

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
WEB_DIR = ROOT / "web"


app = FastAPI(
    title="TactiVision API",
    description="Model-derived football analytics over processed match clips.",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskBody(BaseModel):
    question: str


def _match_dir(match_id: str) -> Path:
    path = PROCESSED / match_id
    if not path.is_dir():
        raise HTTPException(404, f"Unknown match_id: {match_id}")
    return path


def _first_existing(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _media_map(root: Path) -> dict[str, Path | None]:
    tracks = root / "visualizations" / "tracks"
    return {
        "tracking": _first_existing(tracks / "preview.mp4", tracks / "preview.avi"),
        "trajectories": _first_existing(
            root / "visualizations" / "trajectories.mp4",
            root / "visualizations" / "trajectories.avi",
        ),
        "ball": _first_existing(
            root / "ball" / "ball_preview.mp4",
            root / "ball" / "ball_preview.avi",
        ),
        "radar": _first_existing(
            root / "visualizations" / "pitch_radar.mp4",
            root / "visualizations" / "pitch_radar.avi",
        ),
        "possession_timeline": _first_existing(
            root / "visualizations" / "possession" / "possession_timeline.png"
        ),
    }


def _match_list_entry(path: Path) -> dict:
    bundle_path = path / "match_bundle.json"
    meta_path = path / "metadata.json"
    entry: dict = {
        "match_id": path.name,
        "has_bundle": bundle_path.exists(),
        "has_possession": (path / "metrics" / "possession.csv").exists(),
        "has_tracking_preview": _media_map(path)["tracking"] is not None,
    }
    metadata: dict = {}
    if meta_path.exists():
        loaded = load_json(meta_path)
        if isinstance(loaded, dict):
            metadata = loaded
    bundle: dict = {}
    if bundle_path.exists():
        loaded = load_json(bundle_path)
        if isinstance(loaded, dict):
            bundle = loaded
            entry["clip_seconds"] = bundle.get("clip_seconds")
            entry["summaries_available"] = list((bundle.get("summaries") or {}).keys())

    video = bundle.get("video") if isinstance(bundle.get("video"), dict) else {}
    source = (
        metadata.get("source")
        or metadata.get("source_filename")
        or video.get("source")
        or metadata.get("video")
        or path.name
    )
    if isinstance(source, str) and ("\\" in source or source.startswith("/")):
        # Keep list payloads portable.
        source = Path(source).name

    entry["source"] = source
    entry["fps"] = metadata.get("fps") or video.get("fps")
    entry["width"] = metadata.get("width") or video.get("width")
    entry["height"] = metadata.get("height") or video.get("height")
    entry["frame_count"] = metadata.get("frame_count")
    entry["duration_seconds"] = metadata.get("duration_seconds") or bundle.get("clip_seconds")
    entry["available_media"] = sorted(k for k, p in _media_map(path).items() if p is not None)
    return entry


@app.get("/api/health")
def health() -> dict:
    demo_runs = 0
    if PROCESSED.exists():
        demo_runs = sum(
            1
            for path in PROCESSED.iterdir()
            if path.is_dir() and not path.name.startswith("_") and (path / "match_bundle.json").exists()
        )
    return {
        "status": "ok",
        "processed_root": "data/processed",
        "uploads_enabled": UPLOADS_ENABLED and uploads_disabled_reason() is None,
        "bundle_runs": demo_runs,
    }


@app.get("/api/matches")
def list_matches() -> dict:
    matches = []
    if PROCESSED.exists():
        for path in sorted(PROCESSED.iterdir()):
            if not path.is_dir() or path.name.startswith("_"):
                continue
            matches.append(_match_list_entry(path))
    return {"matches": matches}


@app.get("/api/matches/{match_id}")
def get_match(match_id: str) -> dict:
    root = _match_dir(match_id)
    bundle_path = root / "match_bundle.json"
    if not bundle_path.exists():
        raise HTTPException(404, "match_bundle.json missing — run scripts/run_analytics.py")
    return load_json(bundle_path)


@app.get("/api/matches/{match_id}/summary/{name}")
def get_summary(match_id: str, name: str) -> dict | list:
    root = _match_dir(match_id)
    mapping = {
        "possession": root / "metrics" / "possession_summary.json",
        "events": root / "metrics" / "events_summary.json",
        "tactics": root / "metrics" / "tactical_states_summary.json",
        "formation": root / "metrics" / "formation_summary.json",
        "ball": root / "metrics" / "ball_analytics_summary.json",
        "players": root / "metrics" / "player_metrics_summary.json",
        "team_shape": root / "metrics" / "team_shape_summary.json",
        "validation": root / "validation_report.json",
    }
    path = mapping.get(name)
    if path is None or not path.exists():
        raise HTTPException(404, f"summary not found: {name}")
    return load_json(path)


@app.get("/api/matches/{match_id}/possession")
def get_possession(match_id: str, limit: int = Query(2000, ge=1, le=20000)) -> dict:
    rows = load_csv(_match_dir(match_id) / "metrics" / "possession.csv")
    return {"count": len(rows), "rows": rows[:limit]}


@app.get("/api/matches/{match_id}/events")
def get_events(match_id: str) -> dict:
    rows = load_jsonl(_match_dir(match_id) / "metrics" / "events.jsonl")
    return {"count": len(rows), "rows": rows}


@app.get("/api/matches/{match_id}/tactics")
def get_tactics(match_id: str, limit: int = Query(2000, ge=1, le=20000)) -> dict:
    rows = load_jsonl(_match_dir(match_id) / "metrics" / "tactical_states.jsonl")
    return {"count": len(rows), "rows": rows[:limit]}


@app.get("/api/matches/{match_id}/team-shape")
def get_team_shape(match_id: str) -> dict:
    rows = load_csv(_match_dir(match_id) / "metrics" / "team_shape.csv")
    return {"count": len(rows), "rows": rows}


@app.get("/api/matches/{match_id}/players")
def get_players(match_id: str) -> dict:
    rows = load_csv(_match_dir(match_id) / "metrics" / "player_metrics.csv")
    return {"count": len(rows), "rows": rows}


@app.get("/api/matches/{match_id}/ball")
def get_ball(match_id: str, limit: int = Query(2000, ge=1, le=20000)) -> dict:
    rows = load_jsonl(_match_dir(match_id) / "ball" / "ball_pitch_trajectory_per_frame.jsonl")
    return {"count": len(rows), "rows": rows[:limit]}


@app.get("/api/matches/{match_id}/trajectories")
def get_trajectories(match_id: str, limit: int = Query(2000, ge=1, le=20000)) -> dict:
    rows = load_jsonl(_match_dir(match_id) / "pitch_trajectories.jsonl")
    return {"count": len(rows), "rows": rows[:limit]}


@app.post("/api/matches/{match_id}/ask")
def ask(match_id: str, body: AskBody) -> dict:
    root = _match_dir(match_id)
    if not (root / "match_bundle.json").exists():
        raise HTTPException(404, "match_bundle.json missing")
    context = load_context(root)
    return answer_question(body.question, context)


@app.get("/api/matches/{match_id}/media/{kind}")
def media(match_id: str, kind: str):
    root = _match_dir(match_id)
    path = _media_map(root).get(kind)
    if path is None:
        raise HTTPException(404, f"media not found: {kind}")
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/matches/{match_id}/viz")
def list_viz(match_id: str) -> dict:
    root = _match_dir(match_id) / "visualizations"
    files: list[str] = []
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4"}:
                files.append(str(path.relative_to(_match_dir(match_id))).replace("\\", "/"))
    return {"files": sorted(files)}


@app.get("/api/matches/{match_id}/file")
def get_file(match_id: str, path: str = Query(..., description="Relative path under match dir")):
    root = _match_dir(match_id).resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)) or not target.is_file():
        raise HTTPException(404, "file not found")
    if target.suffix.lower() not in {".png", ".jpg", ".jpeg", ".mp4", ".json", ".csv", ".jsonl"}:
        raise HTTPException(403, "file type not allowed")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, filename=target.name)


@app.post("/api/runs")
async def create_run_from_upload(
    video: UploadFile = File(...),
    max_seconds: str | None = Form(default=str(DEFAULT_MAX_SECONDS)),
):
    """Upload a local football clip and run the canonical match pipeline (blocking).

    Portfolio/demo only: the HTTP request waits until processing finishes.
    Disabled on cloud deploys without model weights (see TACTIVISION_UPLOADS_ENABLED).
    """
    try:
        original_name = sanitize_upload_filename(video.filename)
    except ValueError as exc:
        raise HTTPException(400, public_error_message(exc)) from exc

    if not extension_allowed(original_name):
        raise HTTPException(
            400,
            "Unsupported video type. Allowed: " + ", ".join(sorted(ALLOWED_EXTENSIONS)),
        )

    blocked = uploads_disabled_reason()
    if blocked:
        raise HTTPException(503, blocked)

    try:
        limit = parse_max_seconds(max_seconds)
    except ValueError as exc:
        raise HTTPException(400, public_error_message(exc)) from exc

    run_id = new_upload_run_id(processed_root=PROCESSED)
    try:
        ensure_run_slot_free(run_id, processed_root=PROCESSED)
        dest = upload_destination(run_id, original_name)
    except ValueError as exc:
        raise HTTPException(400, public_error_message(exc)) from exc

    size = 0
    try:
        with dest.open("wb") as handle:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    handle.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"Upload exceeds maximum size ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
                    )
                handle.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, public_error_message(exc)) from exc
    finally:
        await video.close()

    if size <= 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        with VideoReader(dest) as reader:
            meta = reader.metadata()
            if meta.width <= 0 or meta.height <= 0 or meta.fps <= 0:
                raise ValueError("Video has no usable width/height/fps")
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Unreadable video: {public_error_message(exc)}") from exc

    try:
        summary = run_match_pipeline(
            dest,
            run_id,
            processed_root=PROCESSED,
            max_seconds=limit,
            skip_detection=True,
            write_previews=True,
        )
    except Exception as exc:
        raise HTTPException(500, f"Processing failed: {public_error_message(exc)}") from exc

    return {
        "run_id": run_id,
        "status": "completed",
        "source_filename": original_name,
        "max_seconds": limit,
        "runtime_seconds": (summary or {}).get("runtime_seconds"),
        "match_bundle": (summary or {}).get("match_bundle"),
        "note": (
            "Local portfolio processing only — not cloud infrastructure. "
            "Long clips are slow; analytics remain model-derived / heuristic."
        ),
    }


# Mount UI last so /api routes take precedence.
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
