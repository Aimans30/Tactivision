# Decisions

| Decision | Rationale |
|----------|-----------|
| YOLO11n for players | Working GPU baseline; no evidence a heavier detector was needed for the portfolio clip. |
| ByteTrack over BoT-SORT | BoT-SORT did not improve the development clip; SNMOT baseline reported for ByteTrack. |
| Keep HSV appearance remapping experimental | On SNMOT-101, post-ByteTrack torso-histogram ID remapping raised IDF1/HOTA and cut IDSW, but Frag unchanged and kit-color confusion risk remains. Do not replace demo ByteTrack until multi-sequence evidence and a safer descriptor exist. |
| One CLI for arbitrary videos | `scripts/run_pipeline.py --video … --run-id …` writes isolated `data/processed/<run-id>/`; API discovers any run with `match_bundle.json`. Demo `1_720p` remains a separate run. |
| Local browser upload via same pipeline | `POST /api/runs` stores under `data/raw/uploads/` and calls `run_match_pipeline` synchronously; portfolio/demo only — no queue/cloud. Default 30s cap; “full” capped at 5 minutes. |
| Do not keep raising `track_buffer` | Sweep 30/60/120: Frag unchanged; IDSW worse at 60. Root issue is misses + re-association, not short buffer alone. |
| Separate ball detector (SoccerNet-v3D) | Generic person models are unreliable on the ball; dedicated weights give high coverage. |
| Greedy single-hypothesis ball tracker | Transparent, no invented tracks across long gaps; gaps stay unobserved. |
| Per-frame pitch calibration (canonical for players **and** ball) | Sparse 2 Hz H left most ball frames unprojected and split player/ball calibration paths. Dense per-frame H is stored once and reused for both projections. Sparse CLIs removed after migration. |
| Player weights under `models/detection/yolo11n.pt` | Single offline weight location; no root-level duplicate. |
| `configs/*.yaml` are reference-only | Not runtime-loaded; argparse/module constants are authoritative. Marked in-file to avoid false control surface. |
| Experiment CLIs under `archive/experiments/` | Preserve reproducibility of SNMOT sweeps / shot tracking without cluttering `scripts/`. |
| No homography / coordinate interpolation | Prefer honest gaps over smooth fiction. Invalid calibration → no player/ball pitch coordinates. |
| StatsBomb 120×80 | Common analytics frame; future event joins stay aligned. |
| Proximity possession (5 yd + 2-frame debounce) | Transparent baseline; distance beyond radius → `unknown`, not nearest-team assignment. |
| Heuristic events after possession | Pass/carry/progression/recovery need ball+player; sparse joins limit recall — still useful candidates. |
| Formation as line signature, not “4-3-3” | Visible player counts ≪ 11; named formations would overclaim. |
| FastAPI + static web/ | Thin API over cached artifacts; no microservice sprawl. |
| Deterministic QA (+ optional LLM wording) | LLM must not invent numbers; answers grounded in `match_bundle` / summaries. |
| Keep experimental markdown under `data/processed/...` | Clip-specific reports stay with outputs; project docs live in `docs/`. |
