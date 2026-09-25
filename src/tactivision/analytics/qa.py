"""Natural-language answers over structured match analytics only.

Never invent statistics. Optional LLM polish; deterministic replies by default.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from tactivision.analytics.io import load_json, load_jsonl, load_csv


def load_context(run_dir: Path) -> dict[str, Any]:
    root = Path(run_dir)
    metrics = root / "metrics"
    bundle = load_json(root / "match_bundle.json") if (root / "match_bundle.json").exists() else {}
    return {
        "bundle": bundle,
        "possession_summary": _maybe_json(metrics / "possession_summary.json"),
        "events_summary": _maybe_json(metrics / "events_summary.json"),
        "events": load_jsonl(metrics / "events.jsonl")[:200],
        "tactics_summary": _maybe_json(metrics / "tactical_states_summary.json"),
        "formation_summary": _maybe_json(metrics / "formation_summary.json"),
        "ball_summary": _maybe_json(metrics / "ball_analytics_summary.json"),
        "player_metrics_summary": _maybe_json(metrics / "player_metrics_summary.json"),
        "player_metrics": load_csv(metrics / "player_metrics.csv")[:100],
        "team_shape_summary": _maybe_json(metrics / "team_shape_summary.json"),
        "validation": _maybe_json(root / "validation_report.json"),
    }


def _maybe_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return load_json(path)


def answer_question(question: str, context: dict[str, Any]) -> dict:
    """Return structured answer with fact vs interpretation separation."""
    q = question.strip().lower()
    facts: list[str] = []
    interpretation: list[str] = []
    uncertainty: list[str] = [
        "All figures are model-derived or heuristic unless labeled ground truth.",
        "Player track IDs are fragmented; they are not named player identities.",
    ]

    poss = context.get("possession_summary") or {}
    events_sum = context.get("events_summary") or {}
    ball = context.get("ball_summary") or {}
    players = context.get("player_metrics_summary") or {}
    formation = context.get("formation_summary") or {}
    tactics = context.get("tactics_summary") or {}
    team_shape = context.get("team_shape_summary") or {}

    if any(k in q for k in ("possession", "who had the ball", "ball control")):
        facts.append(
            f"Assigned possession coverage: {poss.get('possession_coverage_percent')}% of frames "
            f"(team0={poss.get('team0_possession_percent_of_all_frames')}%, "
            f"team1={poss.get('team1_possession_percent_of_all_frames')}%, "
            f"unknown={poss.get('unknown_percent_of_all_frames')}%)."
        )
        facts.append(
            f"Possession transitions (assigned timeline): {poss.get('possession_transitions')}."
        )
        facts.append(
            f"Method: nearest player within {poss.get('possession_radius_yards')} yd + "
            f"{poss.get('debounce_frames')}-frame debounce."
        )
        interpretation.append(
            "These are proximity heuristic labels on model-derived pitch coordinates, not official match possession."
        )

    if any(k in q for k in ("distance", "covered", "speed", "player metric")):
        dist = (players or {}).get("distance_yards") or {}
        spd = (players or {}).get("mean_speed_yards_per_second") or {}
        facts.append(
            f"Tracks measured: {players.get('tracks_measured')}. "
            f"Distance yards min/median/max: {dist.get('minimum')}/{dist.get('median')}/{dist.get('maximum')}."
        )
        facts.append(
            f"Mean speed (yd/s) min/median/max: {spd.get('minimum')}/{spd.get('median')}/{spd.get('maximum')}."
        )
        rows = context.get("player_metrics") or []
        if rows:
            best = max(rows, key=lambda r: float(r.get("distance_yards") or 0))
            facts.append(
                f"Highest distance among measured tracks: track_id={best.get('track_id')} "
                f"at {best.get('distance_yards')} yards (model-derived)."
            )
        uncertainty.append("Short fragmented tracks inflate or truncate distance totals.")

    if any(k in q for k in ("shape", "compact", "width", "length", "centroid")):
        facts.append(f"Team-shape summary available: {bool(team_shape)}.")
        if team_shape:
            facts.append(json.dumps({k: team_shape[k] for k in list(team_shape)[:8]}, default=str))
        interpretation.append("Shape uses visible filtered players only; not full 11-a-side geometry.")

    if any(k in q for k in ("event", "pass", "carry", "recovery", "progress")):
        facts.append(f"Heuristic events total: {events_sum.get('total_events')}; by type: {events_sum.get('by_type')}.")
        sample = (context.get("events") or [])[:5]
        if sample:
            facts.append(
                "Earliest events: "
                + "; ".join(
                    f"{e['timestamp']}s {e['event_type']} team={e.get('team')} conf={e.get('confidence')}"
                    for e in sample
                )
            )
        interpretation.append("Pass/carry candidates are proximity heuristics, not tagged match events.")

    if any(k in q for k in ("ball", "missing")):
        facts.append(
            f"Ball pitch coverage: {ball.get('pitch_coverage_percent')}% "
            f"({ball.get('frames_with_pitch_ball')}/{ball.get('total_frames')} frames); "
            f"segments={ball.get('segments')}; "
            f"median speed={ball.get('median_speed_yards_per_second')} yd/s."
        )

    if any(k in q for k in ("formation", "4-3-3", "line")):
        modal = (formation or {}).get("modal_shape_by_team")
        facts.append(f"Modal estimated line signatures by team: {modal}.")
        interpretation.append(
            "Line signatures are spatial band counts, not named formations like 4-3-3."
        )

    if any(k in q for k in ("tactic", "attack", "defend", "transition", "phase")):
        facts.append(f"Tactical state counts: {(tactics or {}).get('state_counts')}.")
        facts.append(f"Lateral bias counts: {(tactics or {}).get('lateral_bias_counts')}.")
        interpretation.append("Tactical states are thresholded proxies from ball thirds and compactness.")

    if any(k in q for k in ("summar", "overview", "what happened")):
        facts.append(
            f"Clip analytics bundle match_id={(context.get('bundle') or {}).get('match_id')}; "
            f"possession coverage={poss.get('possession_coverage_percent')}%; "
            f"events={events_sum.get('total_events')}; "
            f"ball coverage={ball.get('pitch_coverage_percent')}%."
        )
        interpretation.append(
            "Overall: a short broadcast clip with strong ball detection, sparse player–pitch joins, "
            "and heuristic possession/events rather than official stats."
        )

    if not facts:
        facts.append(
            "No specific metric matched the question. Ask about possession, distance, team shape, "
            "events, ball coverage, formation signature, or tactical states."
        )

    answer = {
        "question": question,
        "facts": facts,
        "interpretation": interpretation,
        "uncertainty": uncertainty,
        "source": "deterministic_structured_analytics",
        "ground_truth": False,
    }

    # Optional LLM polish — only if key present; still grounded in facts above.
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TACTIVISION_LLM_API_KEY")
    if api_key and facts:
        polished = _llm_polish(question, answer, api_key)
        if polished:
            answer["narrative"] = polished
            answer["source"] = "deterministic_facts_plus_optional_llm_wording"
    else:
        answer["narrative"] = " ".join(facts[:3]) + (
            " " + interpretation[0] if interpretation else ""
        )
    return answer


def _llm_polish(question: str, answer: dict, api_key: str) -> str | None:
    """Best-effort OpenAI chat; fail soft. Does not add new numbers."""
    try:
        import urllib.request

        body = {
            "model": os.environ.get("TACTIVISION_LLM_MODEL", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You explain TactiVision football analytics. "
                        "ONLY use the provided facts. Do not invent statistics. "
                        "Clearly separate measured/model-derived facts from interpretation. "
                        "Mention uncertainty."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"question": question, "structured": answer}),
                },
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"]
        # Soft guard: reject replies that invent large new integers not in facts.
        fact_blob = " ".join(answer["facts"])
        for token in re.findall(r"\b\d{2,}\b", text):
            if token not in fact_blob and token not in question:
                return None
        return text
    except Exception:
        return None
