# Evaluation

This document separates **quantitative benchmarks** from **demo-clip coverage
checks**. The SoccerNet SNMOT-101 tracking evaluation is **not** measured on the
Real Madrid–Bayern `1_720p` development clip.

## Player tracking — SoccerNet SNMOT-101 (separate benchmark)

Genuine matching video + GT pair evaluated with TrackEval (MotChallenge).
This sequence is independent of the `1_720p` portfolio demo.

| Metric | Value |
|--------|------:|
| HOTA | 47.058 |
| IDF1 | 53.138 |
| MOTA | 77.545 |
| ID switches | 103 |
| Fragmentation | 232 |

Artifacts: `data/evaluation/SNMOT-101/`.

### Failure analysis (summary)

- 57/103 ID switches follow CLEAR misses then rematch with a new ID.
- 46/103 are identity flips while GT remains matched.
- All 232 fragmentations are restarts after unmatched gaps (mostly 1–5 frames).
- Camera motion ≈ 10 ID switches (secondary).
- Clean crossing swaps and FP-driven switches are rare.

### Persistence sweep

`track_buffer` 30 / 60 / 120 did **not** reduce fragmentation; buffer 60 increased ID switches. Do not tune buffer blindly.

### Appearance-assisted ID remapping (experimental, SNMOT-101)

Offline post-ByteTrack HSV torso-histogram remapping of **newly appeared** IDs onto recently lost IDs (spatial + temporal + similarity gates). Detector and ByteTrack association unchanged. **Not** the demo default.

Reproduce: `archive/experiments/SNMOT-101/run_appearance_assist.py` → `data/evaluation/SNMOT-101/appearance_assist/appearance_assist_results.json`.

| Config | HOTA | IDF1 | MOTA | IDSW | Frag | pred tracks | remaps |
|--------|-----:|-----:|-----:|-----:|-----:|------------:|-------:|
| ByteTrack baseline (reproduced) | 47.058 | 53.138 | 77.545 | 103 | 232 | 113 | — |
| weak (thr 0.88, gap 8, dist 80) | 49.284 | 58.020 | 77.645 | 95 | 232 | 104 | 15 |
| medium (thr 0.75, gap 15, dist 120) | 49.701 | 59.433 | 77.682 | 92 | 232 | 93 | 30 |
| strong (thr 0.62, gap 30, dist 200) | 50.942 | 62.051 | 77.782 | 84 | 232 | 77 | 45 |

Primary identity metrics (IDF1, HOTA, IDSW) improved on this sequence; **fragmentation unchanged** (remapping does not recover CLEAR misses). Remain experimental — kit-color histograms can confuse similar shirts; single-sequence evidence only.

### Tracker comparison on the `1_720p` demo (qualitative / summary only)

BoT-SORT did not improve the working 30 s development clip versus ByteTrack.
That comparison is **not** an SNMOT HOTA/IDF1/MOTA result.

## Calibration (clip `1_720p`)

**Canonical path:** per-frame homographies (`calibration/per_frame/calibration.jsonl`) for **both** player and ball pitch projection.

| Metric | Value |
|--------|------:|
| Valid per-frame homographies | 750 / 750 |
| Median reprojection error | ~1.25 yd |
| Player pitch frames (per-frame H) | 750 calibrated / 750 |
| Player coordinate samples (raw on-pitch) | ~10.2k (was ~873 on sparse 2 Hz path) |

Invalid calibration is left invalid (no homography interpolation).  
Sparse 2 Hz calibration is historical/superseded and is not the active pipeline.

## Ball (clip `1_720p`) — coverage, not accuracy

There is **no ball detection/tracking ground truth** on this clip. Figures below are
**observed-frame / continuity coverage** for the model-derived trajectory.

| Metric | Value |
|--------|------:|
| Observed-frame detection coverage | 644 / 750 = 85.87% |
| Pitch projection of observed frames | 644 / 644 |
| Temporal trajectory segments | 33–35 |
| Longest continuous observed segment | 210 frames / 8.4 s |

Missing detections remain missing. Do not read these as precision, recall, or
tracking accuracy.

## Analytics coverage (heuristic, clip `1_720p`)

After per-frame player+ball projection + possession + analytics bundle:

- Player and ball pitch coordinates share the same per-frame homography source.
- Possession / events / tactics / formation remain **heuristic estimates**; there is no possession/event GT on this clip.
- Validate visually via dashboard + summaries under `data/processed/1_720p/metrics/`.

## What is not evaluated as GT

Possession %, pass counts, formation names, tactical states, xG, ball precision/recall —
**no ground truth** in this project. Do not invent it.
