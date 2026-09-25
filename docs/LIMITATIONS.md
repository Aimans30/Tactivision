# Limitations

## Tracking

- ByteTrack IDs fragment after short unmatched gaps; IDs are **not** player identities.
- SNMOT-101 (separate benchmark): HOTA ~47, 103 ID switches, 232 fragmentations — usable baseline, not professional tracking. Those metrics do **not** apply to the `1_720p` demo clip.
- No jersey OCR / ReID in this project.

## Calibration & projection

- Homography assumes a mostly planar pitch view; extreme zoom/occlusion can degrade keypoints.
- Median ~1.25 yd reprojection error still moves players/ball on the radar.
- Player and ball pitch coordinates both use the **same** per-frame homography file; neither path interpolates missing H.

## Player analytics

- Pitch trajectories now use per-frame homographies (aligned with the ball path).
- Distance/speed on short ByteTrack fragments remain exploratory; do not treat as wearable-grade.
- Heatmaps inherit ID fragmentation and visibility limits.

## Ball

- ~14% of frames lack a ball observation (**observed-frame coverage**, not accuracy); those frames never get invented positions.
- Image-space association can restart track IDs after gaps (segments).
- No ball GT → do not claim precision/recall or “ball tracking accuracy.”

## Possession / events / tactics / formation

- Proximity ≠ control (duels, loose balls, keeper actions).
- Assigned possession still requires ball observed **and** nearest player within radius; ball gaps and distance threshold keep many frames `unknown`/`unavailable`.
- Pass/carry/progression/recovery are **candidates** with confidence scores, not Opta events.
- Tactical states are thresholded proxies (ball thirds, compactness).
- Formation signatures are spatial band counts on **visible** players only.

## Product scope

- No billing, auth, or multi-tenant SaaS features.
- Dashboard expects a preprocessed match directory; live full-match GPU processing is out of scope for the default demo path.
- Optional LLM polish requires an API key and still cannot add unseen statistics.
