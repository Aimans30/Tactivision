# TactiVision

AI-powered football video intelligence platform (portfolio project).

TactiVision turns broadcast football video into structured, model-derived analytics: players, teams, ball, possession heuristics, event candidates, tactical states, and a dashboard that syncs those analytics back to the video timeline.

**This is not a commercial product.** Metrics are model-derived or heuristic unless labeled ground truth.

## What it does

```text
Match video
  → detection (YOLO11n) + ByteTrack
  → player filter / team kits
  → per-frame pitch calibration (StatsBomb 120×80)
  → player pitch trajectories + metrics + heatmaps
  → ball detection/tracking + pitch projection
  → possession / events / tactics / formation signatures
  → match_bundle.json + dashboard + structured Q&A
```

Development clip: SoccerNet `1_720p` (~30 s @ 1280×720 / 25 fps).

## Setup

Python 3.11 or 3.12 (not 3.14).

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate   # Windows CMD: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,web,detection]"
```

GPU detection/tracking needs the `detection` extra (torch + ultralytics) and weights at `models/detection/yolo11n.pt` (plus ball/calibration weights under `models/`). The dashboard and analytics reuse cached `data/processed/` outputs without re-running CV.

`configs/*.yaml` are **reference defaults only** (not loaded at runtime).

## Run the dashboard (cached clip)

If `data/processed/1_720p/` already contains tracks, calibration, ball, and possession:

```bash
python scripts/run_analytics.py --run-dir data/processed/1_720p
python scripts/run_app.py
```

Open http://127.0.0.1:8000

### Local browser upload (portfolio demo)

The dashboard can upload a short football video and process it **locally** via
`POST /api/runs`, which calls the same pipeline as
`python scripts/run_pipeline.py --video … --run-id …`.

- Processing is **synchronous** (the browser request waits until the run finishes).
- Default limit is **30 seconds**; “full” is capped (5 minutes) for safety.
- Long videos are slow; this is not cloud/job-queue infrastructure.
- Analytics remain model-derived / heuristic.
- Uploads land under `data/raw/uploads/` (Git-ignored); each run is isolated under `data/processed/<run_id>/`.

## Recompute analytics only

```bash
python scripts/run_possession.py
python scripts/run_analytics.py
```

## Full CV pipeline (expensive)

Stages are separate so you can resume from cache. **Canonical calibration** is per-frame and shared by player and ball pitch projection:

```bash
python scripts/run_pipeline.py --input path/to/clip.mp4
python scripts/run_detection.py --input path/to/clip.mp4
python scripts/run_tracking.py --input path/to/clip.mp4
python scripts/run_per_frame_calibration.py --input path/to/clip.mp4
python scripts/project_players_pitch_per_frame.py
python scripts/filter_players.py ...   # existing filter; reuse when present
python scripts/clean_trajectories.py
# ball tracking → project_ball_pitch_per_frame → metrics / team_shape /
# heatmaps → possession → analytics
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the stage map.

## Tracking benchmark (SoccerNet SNMOT-101)

Measured on the **SNMOT-101** tracking sequence with TrackEval — **not** on the
Real Madrid–Bayern `1_720p` demo clip.

| Metric | ByteTrack baseline |
|--------|-------------------:|
| HOTA | 47.058 |
| IDF1 | 53.138 |
| MOTA | 77.545 |
| ID switches | 103 |
| Fragmentation | 232 |

Details: [docs/EVALUATION.md](docs/EVALUATION.md).

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and data flow |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Tracking / calibration / ball evaluation |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Technical decisions |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | Known failure modes |

## Tests

```bash
python -m pytest
```

## Provenance

| Layer | Label |
|-------|--------|
| YOLO / ByteTrack / ball detector | model prediction |
| Pitch projection, distance, heatmaps, team shape | model-derived analytic |
| Possession, events, tactics, formation signatures | heuristic estimate |
| SoccerNet SNMOT GT (eval only) | ground truth |

The Ask panel answers from structured JSON only. It does not invent match statistics. Optional `OPENAI_API_KEY` can polish wording; numbers still come from the analytics bundle.
