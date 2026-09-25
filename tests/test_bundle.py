"""Match bundle structure contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactivision.analytics.bundle import build_match_bundle

RUN = Path("data/processed/1_720p")


@pytest.mark.skipif(not (RUN / "match_bundle.json").exists() and not (RUN / "metrics" / "possession.csv").exists(), reason="1_720p demo not present")
def test_build_match_bundle_required_fields():
    computed = build_match_bundle(RUN, clip_seconds=30.0)
    bundle = computed["bundle"]
    for key in (
        "match_id",
        "clip_seconds",
        "coordinate_system",
        "provenance_policy",
        "video",
        "artifacts",
        "summaries",
        "limitations",
    ):
        assert key in bundle
    assert bundle["match_id"] == "1_720p"
    assert bundle["coordinate_system"] == "statsbomb_120x80_yards"
    assert bundle["provenance_policy"]["possession"] == "heuristic_estimate"
    summaries = bundle["summaries"]
    for name in ("possession", "ball", "events", "tactics", "formation", "player_metrics", "team_shape"):
        assert name in summaries
    # Artifact paths that the dashboard depends on should exist when listed.
    for key in ("possession", "ball_pitch", "tracking_preview", "possession_summary"):
        rel = bundle["artifacts"].get(key)
        if rel:
            assert (RUN / rel).exists(), f"missing artifact {key}: {rel}"
    poss = summaries["possession"]
    assert poss.get("ground_truth") is False
    assert 0.0 <= float(poss.get("possession_coverage_percent", 0)) <= 100.0
    ball = summaries["ball"]
    assert 0.0 <= float(ball.get("pitch_coverage_percent", 0)) <= 100.0
