# Repository Cleanup Plan

**Status:** Audit only at creation — **no files were deleted, moved, renamed, or rewritten** as part of the original audit.  
**Update (2026-09-24):** Player pitch projection was migrated to the per-frame calibration path (`scripts/project_players_pitch_per_frame.py`). Sections below that describe a split player/ball calibration path are **historical**; see `docs/ARCHITECTURE.md` for the current canonical path. Deletion/archiving still has **not** been performed.  
**Date:** 2026-09-24  
**Scope:** Entire TactiVision tree excluding `.venv/` contents (dependency install, not project source).

---

## 1. Current repository structure

```text
TactiVision/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .env / .env.example          # SoccerNet password key (local)
├── yolo11n.pt                   # root-level player weights (~5.6 MB)
├── configs/
│   ├── analytics.yaml           # NOT loaded by any code
│   ├── detection.yaml           # NOT loaded by any code
│   ├── pipeline.yaml            # NOT loaded by any code
│   └── tracking.yaml            # NOT loaded by any code
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EVALUATION.md
│   ├── DECISIONS.md
│   └── LIMITATIONS.md
├── src/tactivision/             # canonical library + API
├── web/                         # static dashboard (index.html, app.js, styles.css)
├── scripts/                     # 20 CLI entry scripts
├── tests/                       # 11 pytest modules
├── models/
│   ├── ball/                    # yolo-sn-ball.pt (gitignored pattern)
│   └── calibration/             # pitch YOLO + HF cache metadata
├── data/
│   ├── raw/                     # SoccerNet source video (~1.1G; gitignored)
│   ├── processed/
│   │   ├── 1_720p/              # canonical demo run (~219M; gitignored)
│   │   └── match/               # EMPTY directory
│   └── evaluation/
│       ├── SNMOT-101/           # benchmark results + scripts (~301M)
│       ├── snmot/SNMOT-101/     # GT / img1 / det for eval (~119M)
│       └── SNMOT_EVALUATION_CLIP.md
└── (no .git directory present in this workspace)
```

**Notes**

- There is **no Git repository** initialized here (`fatal: not a git repository`), though `.gitignore` exists. Cleanup recommendations involving “commit / ignore” assume Git will be initialized later.
- No `package.json`, Docker, or `requirements.txt` — Python deps live only in `pyproject.toml`.
- Frontend is static files under `web/`, served by FastAPI.

---

## 2. Canonical active components

### End-to-end runtime graph (current demo)

```text
scripts/run_analytics.py
  → tactivision.analytics.bundle.write_match_bundle
  → data/processed/<match>/match_bundle.json
       + metrics/{events,tactical_states,formation_*,ball_analytics_*} 

scripts/run_app.py
  → uvicorn tactivision.api.app:app
  → FastAPI reads cached data/processed/*
  → mounts web/ static UI
  → web/app.js calls /api/matches/...
  → POST /ask → tactivision.analytics.qa
```

**Prerequisite artifacts** (must already exist for demo; produced by CV scripts, not by `run_analytics`):

| Artifact | Typical producer |
|----------|------------------|
| `metrics/possession.csv` | `scripts/run_possession.py` → `spatial.possession` |
| `metrics/team_shape.csv` | `scripts/run_team_shape.py` → `spatial.team_shape` |
| `metrics/player_metrics.csv` | `scripts/run_metrics.py` → `spatial.metrics` |
| `pitch_trajectories.jsonl` | `scripts/clean_trajectories.py` → `spatial.clean` |
| `player_filter.json` | `scripts/filter_players.py` → `filtering.players` |
| `ball/ball_pitch_trajectory_per_frame.jsonl` | `scripts/project_ball_pitch_per_frame.py` |
| `visualizations/tracks/preview.mp4` | tracking stage |
| other media (ball/trajectories/radar/possession PNG) | respective scripts |

### Canonical library packages (`src/tactivision/`)

| Package | Role |
|---------|------|
| `video/` | ingest / metadata / frames |
| `detection/` | YOLO11n players |
| `tracking/` | ByteTrack (Ultralytics); shot-cut helpers |
| `filtering/` | off-pitch filter + kit teams |
| `calibration/` | pitch keypoints, H, radar |
| `ball/` | ball YOLO + greedy tracker |
| `spatial/` | trajectories, clean, metrics, heatmap, team_shape, possession |
| `analytics/` | ball summary, events, tactics, formation, bundle, QA |
| `api/` | FastAPI app |
| `pipeline/` | orchestration helpers used by early CV scripts |

### Canonical docs / config / UI

- Docs: `README.md`, `docs/{ARCHITECTURE,EVALUATION,DECISIONS,LIMITATIONS}.md`
- UI: `web/`
- Package config: `pyproject.toml`
- Demo data: `data/processed/1_720p/` (application depends on it)

---

## 3. Duplicate implementations found

**Finding:** There is **one** implementation of each core CV/analytics capability under `src/tactivision/`. Duplication is mainly **CLI scripts**, **generated outputs**, and **narrative reports**, not parallel libraries.

| Capability | Canonical code | Duplicate / superseded form | Recommendation |
|------------|----------------|-----------------------------|----------------|
| Player detection | `detection/detector.py` | None in src | Keep |
| Player tracking / ByteTrack | `tracking/tracker.py` (`bytetrack.yaml`) | BoT-SORT = same class + different Ultralytics YAML; results only under `data/processed/1_720p/botsort/` | Keep tracker; keep botsort **results** as evidence; no separate BoT-SORT codebase to delete |
| Shot-aware tracking | `tracking/shots.py` + `pipeline.runner.run_shot_tracking` | Experimental vs continuous ByteTrack; not required by dashboard | Keep code; demote script (see §3 scripts) |
| Player filter / teams | `filtering/players.py` | None | Keep |
| Calibration | `calibration/*` | **Two scripts / two output styles**: sparse `run_calibration.py` → `pitch_positions.jsonl` vs `run_per_frame_calibration.py` → `calibration/per_frame/` | Keep library; consolidate scripts (see below) |
| Trajectory smoothing | `spatial/clean.py` | None | Keep |
| Player metrics | `spatial/metrics.py` | None | Keep |
| Heatmaps | `spatial/heatmap.py` | None | Keep |
| Team shape | `spatial/team_shape.py` | Feature report `TEAM_SHAPE.md` + viz README overlap docs | Keep code; consolidate docs |
| Ball detect/track | `ball/*` | None | Keep |
| Ball pitch projection | `calibration.homography.transform_points` | **Two scripts**: `project_ball_pitch.py` (sparse H / `ball_pitch_trajectory.jsonl`) vs `project_ball_pitch_per_frame.py` (**canonical** for possession/analytics) | Keep per-frame script; archive/remove sparse projection script after confirming unused |
| Possession | `spatial/possession.py` | None | Keep |
| Events / tactics / formation | `analytics/{events,tactics,formation}.py` | None | Keep |
| Analytics bundle | `analytics/bundle.py` | None | Keep |
| API | `api/app.py` | None | Keep |
| Dashboard | `web/` | None | Keep |

**Sparse vs per-frame calibration (important dependency note)**

- Ball analytics use **`ball_pitch_trajectory_per_frame.jsonl`**.
- Player `pitch_trajectories.jsonl` is still produced via `clean_trajectories.py`, whose **default input is `pitch_positions.jsonl`** from **`run_calibration.py`** (sparse path).
- Therefore sparse calibration is **not dead** for a full player-path rebuild until player projection is migrated to per-frame H.

---

## 4. Obsolete / one-off scripts (`scripts/`)

| Script | Role | Verdict | Notes |
|--------|------|---------|-------|
| `run_app.py` | Start API + UI | **KEEP** | Canonical entry |
| `run_analytics.py` | Build bundle / events / tactics / formation | **KEEP** | Canonical entry |
| `run_possession.py` | Write possession CSV/summary/timeline | **KEEP** | Required before analytics |
| `run_team_shape.py` | Team shape CSV | **KEEP** | Required for tactics join + UI |
| `run_metrics.py` | Player metrics CSV | **KEEP** | UI players tab |
| `run_heatmaps.py` | Heatmap PNGs | **KEEP** (optional) | Artifact listed in bundle; UI does not deeply surface yet |
| `clean_trajectories.py` | Pitch trajectory clean | **KEEP** | Rebuild path |
| `filter_players.py` | Filter + teams | **KEEP** | Rebuild path |
| `run_ball_tracking.py` | Ball detect/track | **KEEP** | Rebuild path; hard-coded default video path |
| `project_ball_pitch_per_frame.py` | Canonical ball→pitch | **KEEP** | Used by possession/analytics |
| `run_per_frame_calibration.py` | Dense H | **KEEP** | Ball projection dependency |
| `run_tracking.py` | ByteTrack | **KEEP** | Rebuild + SNMOT docs reference |
| `run_detection.py` | Sampled detection | **KEEP** | Rebuild / early stage |
| `run_pipeline.py` | Ingestion only | **KEEP** | Thin wrapper |
| `run_trajectories.py` | Image-space trails | **KEEP** (optional) | Preview media for UI |
| `download_soccernet.py` | Download one game | **KEEP** | Dev utility; reads `.env` |
| `run_calibration.py` | Sparse player pitch positions | **CONSOLIDATE** | Still default feeder for `clean_trajectories`; migrate to per-frame then demote |
| `project_ball_pitch.py` | Sparse-H ball projection | **ARCHIVE** / later **REMOVE** | Superseded by per-frame; writes unused `ball_pitch_trajectory.jsonl` for current app |
| `run_shot_tracking.py` | Shot-aware ByteTrack experiment | **ARCHIVE** | Results under `bytetrack_shots/`; not on demo graph |
| `visualize_team_shape.py` | Offline PNGs + TEAM_SHAPE_VIZ notes | **CONSOLIDATE** | Useful regen tool; one-off viz README can fold into docs |

---

## 5. Files that should remain as evaluation evidence

**Do not delete these result/docs trees** (measured portfolio evidence):

| Path | Why |
|------|-----|
| `data/evaluation/SNMOT-101/BASELINE_RESULTS.md` | ByteTrack baseline narrative |
| `data/evaluation/SNMOT-101/tracking_metrics.json` + TrackEval outputs | Numeric HOTA/IDF1/MOTA |
| `data/evaluation/SNMOT-101/FAILURE_ANALYSIS.md` + `failure_analysis_*.json` | IDSW/Frag analysis |
| `data/evaluation/SNMOT-101/TRACK_PERSISTENCE_EXPERIMENT.md` + `persistence_sweep/*` | track_buffer 30/60/120 evidence |
| `data/evaluation/SNMOT-101/bytetrack*/tracking_summary.json` | Per-run summaries |
| `data/evaluation/SNMOT_EVALUATION_CLIP.md` | Clip pairing notes |
| `data/evaluation/snmot/SNMOT-101/` GT / images / det | Required to **reproduce** eval (large) |
| `data/processed/1_720p/botsort/tracking_summary.json` | BoT-SORT comparison evidence |
| `data/processed/1_720p/bytetrack_shots/` | Shot-aware experiment evidence |
| `data/processed/1_720p/ball/BALL_*.md` + summaries | Ball experiment evidence |
| `data/processed/1_720p/calibration/per_frame/CALIBRATION_COVERAGE.md` | Dense cal evidence |
| `data/processed/1_720p/calibration_summary.json` + `pitch_positions.jsonl` | Sparse cal historical outputs |
| `data/processed/1_720p/{POSSESSION,TEAM_SHAPE,VALIDATION}.md` | Clip-level evidence (content partially overlaps `docs/`) |
| `data/processed/1_720p/evaluation/` | Clip-local tracking eval notes |

Canonical **interpretation** of the above should live in `docs/EVALUATION.md` / `docs/DECISIONS.md`; raw numbers and run folders stay under `data/evaluation/` and `data/processed/...`.

---

## 6. Files / folders to archive (proposed)

Move later to something like `archive/` or `data/evaluation/experiments/` — **not done in this audit**.

| Path | Reason |
|------|--------|
| `scripts/project_ball_pitch.py` | Superseded ball projection CLI |
| `scripts/run_shot_tracking.py` | Experiment CLI (library can stay) |
| `data/evaluation/SNMOT-101/_analyze_failures.py` | One-off analysis script (already under eval) |
| `data/evaluation/SNMOT-101/persistence_sweep/run_persistence_sweep.py` | One-off sweep runner (keep results JSON/MD) |
| `data/processed/1_720p/ball/ball_pitch_trajectory.jsonl` + `BALL_PITCH_PROJECTION.md` + `ball_pitch_projection_summary.json` | Sparse-H ball outputs unused by app |
| `data/processed/match/` | Empty placeholder |

---

## 7. Proposed deletions (plan only — **not executed**)

For every candidate:

### Application / scripts

| PATH | REASON | EVIDENCE UNUSED/OBSOLETE | SAFE TO DELETE |
|------|--------|--------------------------|----------------|
| `scripts/project_ball_pitch.py` | Superseded by per-frame projector | App/analytics load `ball_pitch_trajectory_per_frame.jsonl` only; no import of this script | **YES** after confirming no external docs you care about still instruct it as primary (clip MD still references it — update docs first) |
| `data/processed/1_720p/ball/ball_pitch_trajectory.jsonl` | Old sparse projection | Not in `match_bundle` artifacts; possession uses per-frame file | **YES** (regenerable; keep MD summary if desired) |
| `data/processed/1_720p/ball/ball_pitch_projection_summary.json` | Paired with sparse projection | Same | **YES** |
| `data/processed/match/` | Empty dir | `find` shows no files; API lists it only as empty match without bundle | **YES** |
| Root `yolo11n.pt` | Duplicate weight location | Scripts default to `"yolo11n.pt"` CWD; also acceptable under `models/` — consolidate **one** location first | **UNCERTAIN** until defaults updated |
| `models/calibration/.cache/huggingface/**` | HF download cache | Cache metadata only; weights should be re-fetched or stored intentionally | **YES** for cache files; keep actual `.pt` if present |

### Evaluation one-off executables (optional)

| PATH | REASON | EVIDENCE | SAFE TO DELETE |
|------|--------|----------|----------------|
| `data/evaluation/SNMOT-101/_analyze_failures.py` | One-off; results already in MD/JSON | Docstring: “One-off SNMOT-101 identity failure analysis” | **NO** until archived; **YES** if MD/JSON retained and script moved to archive |
| `data/evaluation/SNMOT-101/persistence_sweep/run_persistence_sweep.py` | One-off sweep | Results JSON/MD exist; script re-runs expensive GPU tracking | **NO** delete blindly — **ARCHIVE** |

### Documentation duplicates (delete only after consolidation)

| PATH | REASON | EVIDENCE | SAFE TO DELETE |
|------|--------|----------|----------------|
| `data/processed/1_720p/visualizations/team_shape/README.md` | Feature mini-README | Overlaps `TEAM_SHAPE.md` / `docs/LIMITATIONS.md` | **UNCERTAIN** — prefer consolidate then remove |
| Multiple `data/processed/1_720p/**/*.md` | Feature reports | Valuable evidence; content overlaps `docs/*` | **NO** mass-delete — **CONSOLIDATE** pointers into docs, keep or archive reports |

### Config / secrets

| PATH | REASON | EVIDENCE | SAFE TO DELETE |
|------|--------|----------|----------------|
| `.env` | Local secret file | Contains `SOCCERNET_PASSWORD` key; `.gitignore` lists `.env` | **Do not commit**; keep locally or delete locally after rotating — **YES** remove from any future VCS | 
| `configs/*.yaml` as “unused” | Not loaded | No `yaml.safe_load` / path references in `src/` or `scripts/` | **NO** delete yet — **wire up or mark as reference**; deleting loses documented defaults |

### Do **not** delete

- `data/processed/1_720p/` demo run (app depends on it)
- `data/evaluation/SNMOT-101/` metrics & narratives
- `data/evaluation/snmot/` if you need reproducibility (or document “eval data not redistributed”)
- `data/raw/` SoccerNet video (local only; licensing)
- Any `src/tactivision/**` modules without a proven replacement

---

## 8. Proposed final structure

Current layout is already close to the target. Prefer **minimal moves**.

```text
TactiVision/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .env.example                 # no secrets
├── configs/                     # either load from code OR document as reference-only
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EVALUATION.md
│   ├── DECISIONS.md
│   ├── LIMITATIONS.md
│   └── REPOSITORY_CLEANUP_PLAN.md   # this file
├── src/tactivision/             # unchanged package layout
├── web/
├── scripts/                     # slim to KEEP list; optional scripts/dev/ for regenerate helpers
├── tests/
├── models/                      # all weights here (ball, calibration, yolo11n)
├── data/
│   ├── raw/                     # gitignored
│   ├── processed/1_720p/        # demo cache gitignored (or curated subset)
│   └── evaluation/              # SNMOT evidence (+ optional archive/ subfolder)
└── (optional) archive/          # superseded scripts + sparse ball outputs
```

**Do not** invent a second package tree or relocate `web/` / `src/` purely for aesthetics.

---

## 9. Dependencies / configuration cleanup recommendations

| Item | Finding | Recommendation |
|------|---------|----------------|
| `pyproject.toml` | Single source of deps; extras `dev`, `data`, `detection`, `web` | Keep; ensure `matplotlib` stays (possession/team-shape scripts) |
| `pyyaml` | Declared dependency but **project YAML configs never loaded** | Wire scripts to `configs/*.yaml` **or** drop unused dep later |
| `configs/analytics.yaml` | Written but unused by `analytics/*` (params hard-coded) | Load in bundle/events/tactics **or** delete after documenting constants in docs |
| No `requirements.txt` / Docker / npm | Fine for portfolio | Optional: export lockfile later; not required |
| Hard-coded absolute paths | `DEFAULT_VIDEO` in several scripts; absolute model paths in **generated** JSON summaries | Prefer relative paths from repo root; regenerate summaries |
| `DEFAULT_VIDEO` in `analytics/bundle.py` | Machine-specific SoccerNet relative path string in bundle | Keep relative; OK if under `data/raw/...` |
| `.env` / SoccerNet password | Present locally; example file OK | Never commit `.env`; ensure Git ignore before first commit |
| Root `yolo11n.pt` vs `models/` | Split weight locations | Standardize on `models/detection/yolo11n.pt` (or keep root if documented) |
| `.gitignore` | Ignores `data/processed/*`, `data/raw/*`, `*.pt`, `*.mp4`, `.env` | Good for licensing/size; decide whether a **curated** demo subset should be force-added for portfolio clones |
| No Git repo | Cannot use versioned cleanup | Initialize Git before executing delete/archive commits |
| SoccerNet raw video (~1.1G) | Copyrighted / ToS-bound | Do not publish; keep gitignored |
| Empty `data/processed/match` | Noise in `/api/matches` | Remove empty dir |

**Tests gap (not deletion — follow-up):** no tests for `api/app.py`, `spatial.possession`, `spatial.team_shape`, `ball/*`, `analytics.bundle`. Keep existing tests; add coverage later without removing current ones.

---

## 10. Python file classification (A–G)

Legend: **A** imported by app · **B** E2E CV/analytics pipeline · **C** tests · **D** eval utility · **E** useful standalone tool · **F** obsolete/duplicated · **G** unclear

### `src/tactivision/`

| File | Class | Notes |
|------|-------|-------|
| `api/app.py` | A | Dashboard backend |
| `api/__init__.py` | A | Re-exports app |
| `analytics/bundle.py` | A,B | `run_analytics` |
| `analytics/qa.py` | A | Ask endpoint |
| `analytics/io.py` | A | Shared loaders |
| `analytics/schema.py` | A | Provenance enums |
| `analytics/ball_metrics.py` | A,B,C | Bundle + tests |
| `analytics/events.py` | A,B,C | Bundle + tests |
| `analytics/tactics.py` | A,B,C | Bundle + tests |
| `analytics/formation.py` | A,B,C | Bundle + tests |
| `analytics/__init__.py` | A | Public exports |
| `spatial/possession.py` | B (A via CSV) | Script + `index_players` in bundle |
| `spatial/team_shape.py` | B | Script-produced CSV consumed by UI/tactics |
| `spatial/metrics.py` | B,C | |
| `spatial/heatmap.py` | B,C | |
| `spatial/clean.py` | B,C | |
| `spatial/trajectories.py` | B,C | Via pipeline |
| `spatial/__init__.py` | A/B | |
| `ball/detector.py` | B | Via `run_ball_tracking` |
| `ball/tracker.py` | B | |
| `ball/schemas.py` | B | |
| `ball/__init__.py` | B | |
| `calibration/*` | B,C | Homography tests; scripts |
| `filtering/players.py` | B,C | |
| `detection/*` | B,C | Schema tests; runner |
| `tracking/tracker.py` | B | |
| `tracking/schemas.py` | B,C | |
| `tracking/shots.py` | B,C,E | Shot experiment + tests |
| `pipeline/runner.py` | B,C | Ingestion/detect/track/shot/traj |
| `video/*` | B,C | |
| `__init__.py` | A | Version |

### `scripts/`

| File | Class | Verdict |
|------|-------|---------|
| `run_app.py` | A | KEEP |
| `run_analytics.py` | A,B | KEEP |
| `run_possession.py` | B | KEEP |
| `run_team_shape.py` | B | KEEP |
| `run_metrics.py` | B | KEEP |
| `run_heatmaps.py` | B,E | KEEP optional |
| `clean_trajectories.py` | B | KEEP |
| `filter_players.py` | B | KEEP |
| `run_ball_tracking.py` | B | KEEP |
| `project_ball_pitch_per_frame.py` | B | KEEP |
| `run_per_frame_calibration.py` | B | KEEP |
| `run_tracking.py` | B | KEEP |
| `run_detection.py` | B | KEEP |
| `run_pipeline.py` | B | KEEP |
| `run_trajectories.py` | B,E | KEEP optional |
| `download_soccernet.py` | E | KEEP |
| `run_calibration.py` | B (legacy path) | CONSOLIDATE |
| `project_ball_pitch.py` | F | ARCHIVE/REMOVE |
| `run_shot_tracking.py` | D/E | ARCHIVE |
| `visualize_team_shape.py` | E | CONSOLIDATE |

### Eval Python

| File | Class | Verdict |
|------|-------|---------|
| `data/evaluation/SNMOT-101/_analyze_failures.py` | D | ARCHIVE (keep outputs) |
| `data/evaluation/SNMOT-101/persistence_sweep/run_persistence_sweep.py` | D | ARCHIVE (keep results) |

### `tests/`

All **C** (active). None identified as duplicate/stale by import graph. Gaps: API, possession, team_shape, ball, bundle (missing coverage, not obsolete files).

| File | Imports canonical `src/`? |
|------|---------------------------|
| `test_analytics.py` | Yes |
| `test_detection.py` | Yes |
| `test_heatmap.py` | Yes |
| `test_homography.py` | Yes |
| `test_metrics.py` | Yes |
| `test_player_filter.py` | Yes |
| `test_shots.py` | Yes |
| `test_tracking.py` | Yes |
| `test_trajectories.py` | Yes |
| `test_trajectory_clean.py` | Yes |
| `test_video.py` | Yes |

---

## 11. Data review summary

| Class | Location | Action |
|-------|----------|--------|
| A. Raw/source | `data/raw/soccernet/.../1_720p.mkv` | Keep local; **do not commit** (ToS/copyright) |
| B. Processed demo | `data/processed/1_720p/**` | **Keep** for app; gitignore bulk; optional curated publish set |
| C. Evaluation | `data/evaluation/**` | Keep evidence; GT images large — document redistribution limits |
| D. Temporary/generated | MP4/PNG/JSONL under processed; HF caches | Prefer gitignore over delete |

---

## 12. Documentation review

**Canonical (keep):** `README.md`, `docs/ARCHITECTURE.md`, `docs/EVALUATION.md`, `docs/DECISIONS.md`, `docs/LIMITATIONS.md`, this plan.

**Clip/experiment reports (evidence — consolidate, don’t mass-delete):**

- `data/processed/1_720p/POSSESSION.md`, `TEAM_SHAPE.md`, `VALIDATION.md`
- `data/processed/1_720p/ball/BALL_*.md`
- `data/processed/1_720p/calibration/per_frame/CALIBRATION_COVERAGE.md`
- `data/processed/1_720p/evaluation/TRACKING_EVALUATION.md`
- `data/processed/1_720p/visualizations/team_shape/README.md`
- `data/evaluation/SNMOT-101/*.md`, `SNMOT_EVALUATION_CLIP.md`

**Proposed doc cleanup (later):** add short “Evidence index” sections in `docs/EVALUATION.md` linking to those paths; then optional archive of redundant prose.

---

## 13. Uncertainty (usage not fully established)

1. **Whether sparse `pitch_positions.jsonl` still matches current player analytics quality** vs an unpublished per-frame player projection — app runs, but rebuild path may be inconsistent with ball’s per-frame H.
2. **Whether portfolio Git should ship any of `data/processed/1_720p`** (demos without SoccerNet download) despite `.gitignore`.
3. **Whether `data/evaluation/snmot/` must remain** for third parties (size + SoccerNet license) vs “results only”.
4. **Root `yolo11n.pt` vs Ultralytics auto-download** — deleting root weights may break offline scripts that expect CWD default.
5. **`configs/*.yaml` intent** — written as milestone config but never wired; unknown if intentional “documentation as YAML”.
6. **Heatmap / team-shape PNGs** — in bundle artifact list but weakly used by current UI; keep generators until UI decision.
7. **No Git history** — cannot verify “last used” via blame; classifications use import/entry-point evidence only.

---

## 14. Suggested execution order (historical — superseded by §16)

1. Initialize Git; verify `.gitignore`; ensure `.env` never staged.
2. Consolidate evaluation narrative into `docs/EVALUATION.md` (links only).
3. Wire or formally mark `configs/` as reference-only.
4. Migrate player pitch rebuild to per-frame calibration; then demote `run_calibration.py` / sparse ball projector.
5. Archive superseded scripts + sparse ball JSONL.
6. Remove empty `data/processed/match/`.
7. Standardize model weight paths.
8. Add API/possession tests.
9. Only then delete archived files from the working tree.

---

## 15. Explicit non-actions of the original audit

- At audit time: no deletions, moves, renames, or behavior changes were performed.
- This file was the sole audit deliverable; cleanup execution is recorded in §16.

---

## 16. Cleanup execution log (2026-09-24)

Per-frame calibration migration was already complete before this cleanup.

### Deleted
| Path | Reason |
|------|--------|
| `scripts/project_ball_pitch.py` | Superseded sparse ball projector |
| `scripts/run_calibration.py` | Superseded sparse player calibration CLI |
| `data/processed/1_720p/ball/ball_pitch_trajectory.jsonl` | Obsolete sparse ball pitch output |
| `data/processed/1_720p/ball/ball_pitch_projection_summary.json` | Paired sparse summary |
| `data/processed/match/` | Empty placeholder |
| `models/calibration/.cache/huggingface/` | HF cache only; weight kept at `models/calibration/yolo-football-pitch-detection.pt` |

### Archived (evidence retained in place)
| From | To |
|------|----|
| `scripts/run_shot_tracking.py` | `archive/experiments/run_shot_tracking.py` |
| `data/evaluation/SNMOT-101/_analyze_failures.py` | `archive/experiments/SNMOT-101/_analyze_failures.py` |
| `data/evaluation/SNMOT-101/persistence_sweep/run_persistence_sweep.py` | `archive/experiments/SNMOT-101/run_persistence_sweep.py` |

### Moved
| From | To |
|------|----|
| `yolo11n.pt` (repo root) | `models/detection/yolo11n.pt` |

### Kept (confirmed useful)
- `scripts/visualize_team_shape.py` — reproducible viz generator; not replaced by API
- All SNMOT-101 result MD/JSON, persistence sweep results, BoT-SORT summaries
- `data/processed/1_720p/` demo run
- `data/evaluation/snmot/`, `data/raw/` (local; gitignored)
- Calibration library under `src/tactivision/calibration/`
- `BALL_PITCH_PROJECTION.md` (historical evidence)

### Configuration
- `configs/*.yaml` marked **REFERENCE DEFAULTS — not loaded at runtime**
- PyYAML retained (declared dependency; Ultralytics/other tooling may use YAML)
- Defaults for player weights updated to `models/detection/yolo11n.pt` in detector/tracker/runner/scripts

### Git
- Repository initialized
- `.gitignore` updated: `.env`, `data/raw`, `data/processed`, `data/evaluation/snmot`, `models/**/*.pt`, `models/**/.cache/`, media extensions
- No secrets committed; `.env` ignored
