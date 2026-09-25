# Architecture

## Pipeline

Canonical CLI (arbitrary video → isolated run directory):

`python scripts/run_pipeline.py --video <path> --run-id <id> [--max-seconds N]`

Writes `data/processed/<run-id>/` including `match_bundle.json` for the API.

Local browser upload (portfolio only): `POST /api/runs` stores a file under
`data/raw/uploads/<run_id>/` and calls the same `run_match_pipeline` orchestrator.
The HTTP request blocks until processing finishes; long clips are slow.

```text
VIDEO
  video/reader.py + metadata
DETECTION
  detection/detector.py (YOLO11n)
TRACKING
  tracking/tracker.py (ByteTrack)
FILTER / TEAMS
  filtering/players.py
CALIBRATION (canonical: per-frame)
  calibration/*  (keypoints → homography → StatsBomb 120×80)
  → calibration/per_frame/calibration.jsonl
PLAYER PITCH (same per-frame H)
  → pitch_positions.jsonl
  spatial/clean.py, metrics.py, heatmap.py, team_shape.py, possession.py
BALL PITCH (same per-frame H)
  ball/detector.py + tracker.py
  → ball/ball_pitch_trajectory_per_frame.jsonl
ANALYTICS
  spatial/team_shape.py, spatial/possession.py
  analytics/ball_metrics.py, events.py, tactics.py, formation.py
  analytics/bundle.py → match_bundle.json
API + UI
  api/app.py + web/
  analytics/qa.py (structured Q&A)
```

## Package layout

```text
src/tactivision/
  video/          ingestion
  detection/      player YOLO
  tracking/       ByteTrack / shot-aware
  calibration/    pitch homography
  filtering/      keep + team kits
  ball/           ball detect/track
  spatial/        trajectories, metrics, heatmaps, team shape, possession
  analytics/      ball summary, events, tactics, formation, bundle, QA
  api/            FastAPI
  pipeline/       early-stage orchestration helpers
web/              static dashboard
scripts/          thin CLIs over library modules
configs/          reference YAML (not runtime-loaded; see file headers)
data/processed/   cached run outputs (do not invent fills)
data/evaluation/  SNMOT-101 benchmarks
docs/             consolidated project docs
```

## Data contracts

- Coordinates: StatsBomb **120 × 80 yards** when on-pitch.
- Every analytic row should carry `metric_type` / `provenance` and `ground_truth: false` when estimated.
- Missing ball / invalid calibration / insufficient players → leave fields empty or mark `unavailable`; **never interpolate** to invent values.
- **Canonical calibration path:** per-frame homographies in `calibration/per_frame/calibration.jsonl` drive **both** player and ball pitch projection. Homographies are not interpolated; invalid frames get no invented coordinates.
- Player projection: `scripts/project_players_pitch_per_frame.py` → `pitch_positions.jsonl`
- Ball projection: `scripts/project_ball_pitch_per_frame.py` → `ball/ball_pitch_trajectory_per_frame.jsonl`
- Primary hub for spatial joins: `pitch_trajectories.jsonl` + `player_filter.json` + `ball_pitch_trajectory_per_frame.jsonl`.

## Match bundle

`scripts/run_analytics.py` writes `match_bundle.json` listing artifact paths and nested summaries (possession, ball, events, tactics, formation, player metrics, team shape). The API serves this file and the underlying CSV/JSONL without recomputing CV.

## Dashboard sync

The UI plays cached MP4 previews (`visualizations/tracks/preview.mp4`, ball/trajectories/radar). Event and timeline clicks set `video.currentTime` from stored timestamps (25 fps clip clock). MKV source is not required in-browser.
