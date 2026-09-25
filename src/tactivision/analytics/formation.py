"""Approximate team line structure from spatial player distributions.

Does NOT claim exact formations (e.g. 4-3-3) as fact. Emits estimated
line counts with confidence based on visible player coverage.
"""

from __future__ import annotations

from tactivision.analytics.schema import COORDINATE_SYSTEM, HEURISTIC


def _lines_from_points(points: list[tuple[float, float]], *, along_x: bool = True) -> dict:
    """Split players into three bands relative to the team centroid axis."""
    if len(points) < 5:
        return {
            "valid": False,
            "n_players": len(points),
            "line_counts": None,
            "shape_signature": None,
            "confidence": 0.0,
            "reason": "insufficient_players",
        }
    axis = [p[0] for p in points] if along_x else [p[1] for p in points]
    mean = sum(axis) / len(axis)
    # Relative thirds around the mean using std-ish spread.
    spread = (sum((v - mean) ** 2 for v in axis) / len(axis)) ** 0.5
    if spread < 1e-6:
        return {
            "valid": False,
            "n_players": len(points),
            "line_counts": None,
            "shape_signature": None,
            "confidence": 0.0,
            "reason": "degenerate_spread",
        }
    low = mean - 0.55 * spread
    high = mean + 0.55 * spread
    back = mid = fwd = 0
    for v in axis:
        if v < low:
            back += 1
        elif v > high:
            fwd += 1
        else:
            mid += 1
    counts = [back, mid, fwd]
    # Confidence rises with player count toward 11, capped.
    coverage = min(1.0, len(points) / 11.0)
    confidence = round(0.35 + 0.45 * coverage, 3)
    signature = "-".join(str(c) for c in counts)
    return {
        "valid": True,
        "n_players": len(points),
        "line_counts": counts,
        "shape_signature": signature,
        "confidence": confidence,
        "centroid_axis": round(mean, 3),
        "spread_axis": round(spread, 3),
        "reason": None,
        "disclaimer": (
            f"Estimated line signature {signature} from {len(points)} visible tracks. "
            "Not an official formation label; missing players and fragmented IDs bias counts."
        ),
    }


def estimate_formation_at_frame(
    players: list[tuple[int, int, float, float]],
) -> dict:
    """players: (track_id, team, x, y)."""
    by_team: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
    for _tid, team, x, y in players:
        if team in by_team:
            by_team[team].append((x, y))
    out = {
        "provenance": HEURISTIC,
        "ground_truth": False,
        "coordinate_system": COORDINATE_SYSTEM,
        "axis": "pitch_x_relative_to_team_centroid",
        "teams": {},
    }
    for team_id, pts in by_team.items():
        out["teams"][str(team_id)] = _lines_from_points(pts, along_x=True)
    return out


def summarize_formation_over_time(
    players_by_frame: dict[int, list[tuple[int, int, float, float]]],
    timestamps: dict[int, float],
) -> dict:
    """Aggregate modal shape signatures across frames with enough players."""
    frames_out: list[dict] = []
    sig_counts: dict[str, dict[str, int]] = {"0": {}, "1": {}}
    for frame in sorted(players_by_frame):
        estimate = estimate_formation_at_frame(players_by_frame[frame])
        row = {
            "frame": frame,
            "timestamp": round(timestamps.get(frame, frame / 25.0), 3),
            **estimate,
        }
        frames_out.append(row)
        for team_id, payload in estimate["teams"].items():
            if not payload.get("valid"):
                continue
            sig = payload["shape_signature"]
            bucket = sig_counts[team_id]
            bucket[sig] = bucket.get(sig, 0) + 1

    modal = {}
    for team_id, bucket in sig_counts.items():
        if not bucket:
            modal[team_id] = None
            continue
        best = max(bucket.items(), key=lambda kv: kv[1])
        modal[team_id] = {
            "shape_signature": best[0],
            "frames": best[1],
            "share_of_valid_frames": round(best[1] / sum(bucket.values()), 3),
        }

    return {
        "metric_type": HEURISTIC,
        "provenance": HEURISTIC,
        "ground_truth": False,
        "frames_evaluated": len(frames_out),
        "modal_shape_by_team": modal,
        "note": (
            "Shape signatures are spatial line-count estimates (back-mid-forward bands), "
            "not named formations. Do not report as 4-3-3 unless evidence and confidence "
            "explicitly support it — this pipeline does not claim named formations."
        ),
        "per_frame": frames_out,
    }
