"""SNMOT-101 appearance-assisted ID remapping experiment (offline on ByteTrack tracks).

Does not change the default demo tracker. Reuses existing bytetrack/tracks.jsonl
detections/IDs; only remaps identities via HSV torso appearance + spatial gates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tactivision.tracking.appearance import PRESETS, AppearanceParams  # noqa: E402
from tactivision.tracking.appearance_remap import AppearanceRemapper  # noqa: E402
from tactivision.tracking.schemas import Track  # noqa: E402
from tactivision.tracking.snmot_eval import (  # noqa: E402
    EVAL_ROOT,
    evaluate_mot,
    tracks_jsonl_to_mot,
)
from tactivision.video.reader import VideoReader  # noqa: E402

VIDEO = EVAL_ROOT / "SNMOT-101.mp4"
BASE_TRACKS = EVAL_ROOT / "bytetrack" / "tracks.jsonl"
BASE_MOT = EVAL_ROOT / "predictions_mot.txt"
OUT = EVAL_ROOT / "appearance_assist"


def load_tracks_by_frame(path: Path) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_frame[int(row["frame_id"])] = row.get("tracks") or []
    return by_frame


def remap_tracks_jsonl(
    tracks_by_frame: dict[int, list[dict]],
    video: Path,
    out_jsonl: Path,
    params: AppearanceParams,
) -> dict:
    remapper = AppearanceRemapper(params)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    written = 0
    with VideoReader(video) as reader, out_jsonl.open("w", encoding="utf-8") as handle:
        for frame_id, timestamp, frame in reader.iter_frames():
            if frame_id not in tracks_by_frame and frame_id > max(tracks_by_frame, default=-1):
                break
            raw = tracks_by_frame.get(frame_id, [])
            tracks = tuple(
                Track(
                    track_id=None if tr.get("track_id") is None else int(tr["track_id"]),
                    class_name=str(tr.get("class", "player")),
                    confidence=float(tr.get("confidence", 0.0)),
                    bbox=tuple(float(x) for x in tr["bbox"]),  # type: ignore[arg-type]
                )
                for tr in raw
            )
            remapped = remapper.apply(frame, frame_id, tracks)
            handle.write(
                json.dumps(
                    {
                        "frame_id": frame_id,
                        "timestamp": round(float(timestamp), 3),
                        "shot_id": 0,
                        "tracks": [t.to_dict() for t in remapped if t.track_id is not None],
                    }
                )
                + "\n"
            )
            written += 1
            if (frame_id + 1) % 100 == 0:
                print(f"  remapped {frame_id + 1} frames...", flush=True)
    runtime = time.perf_counter() - t0
    return {
        "frames_written": written,
        "remap_events": len(remapper.remap_events),
        "runtime_seconds": round(runtime, 2),
        "sample_remaps": remapper.remap_events[:20],
    }


def pct(metrics: dict) -> dict:
    return {
        "HOTA": round(100.0 * metrics["HOTA"], 3)
        if metrics.get("HOTA_percent") is None
        else round(float(metrics["HOTA_percent"]), 3),
        "IDF1": round(100.0 * metrics["IDF1"], 3)
        if metrics.get("IDF1_percent") is None
        else round(float(metrics["IDF1_percent"]), 3),
        "MOTA": round(100.0 * metrics["MOTA"], 3)
        if metrics.get("MOTA_percent") is None
        else round(float(metrics["MOTA_percent"]), 3),
        "IDSW": int(metrics["ID_switches"]),
        "Frag": int(metrics["fragmentation"]),
        "pred_tracks": int(metrics["evaluator_IDs"]),
        "pred_boxes": int(metrics["evaluator_Dets"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Appearance-assist SNMOT experiment")
    parser.add_argument("--skip-remap", action="store_true", help="Only re-eval baseline MOT")
    args = parser.parse_args()

    if not VIDEO.is_file():
        print(f"error: missing {VIDEO}", file=sys.stderr)
        return 2
    if not BASE_TRACKS.is_file():
        print(f"error: missing {BASE_TRACKS}", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "sequence": "SNMOT-101",
        "method": "post_bytetrack_hsv_torso_histogram_id_remap",
        "note": (
            "Appearance remapping is applied offline to existing ByteTrack tracks. "
            "Detector/ByteTrack association unchanged. Experimental — not demo default."
        ),
        "baseline_target": {
            "HOTA": 47.058,
            "IDF1": 53.138,
            "MOTA": 77.545,
            "IDSW": 103,
            "Frag": 232,
        },
        "runs": [],
    }

    print("=== baseline TrackEval on existing predictions_mot.txt ===", flush=True)
    base_metrics = evaluate_mot(BASE_MOT, "baseline_reproduce", OUT)
    base_pct = pct(base_metrics)
    results["baseline_reproduction"] = {
        **base_pct,
        "matches_published": (
            abs(base_pct["HOTA"] - 47.058) < 0.01
            and abs(base_pct["IDF1"] - 53.138) < 0.01
            and abs(base_pct["MOTA"] - 77.545) < 0.01
            and base_pct["IDSW"] == 103
            and base_pct["Frag"] == 232
        ),
        "summary_path": base_metrics["summary_path"],
    }
    print(results["baseline_reproduction"], flush=True)
    if not results["baseline_reproduction"]["matches_published"]:
        print("STOP: baseline reproduction mismatch — investigate before appearance runs.", file=sys.stderr)
        (OUT / "appearance_assist_results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        return 1

    if args.skip_remap:
        return 0

    tracks_by_frame = load_tracks_by_frame(BASE_TRACKS)
    for name, params in PRESETS.items():
        print(f"=== appearance preset: {name} ===", flush=True)
        out_jsonl = OUT / f"tracks_{name}.jsonl"
        out_mot = OUT / f"predictions_{name}.txt"
        info = remap_tracks_jsonl(tracks_by_frame, VIDEO, out_jsonl, params)
        n_ids, n_boxes = tracks_jsonl_to_mot(out_jsonl, out_mot)
        metrics = evaluate_mot(out_mot, f"appearance_{name}", OUT)
        run_pct = pct(metrics)
        delta = {
            "HOTA": round(run_pct["HOTA"] - base_pct["HOTA"], 3),
            "IDF1": round(run_pct["IDF1"] - base_pct["IDF1"], 3),
            "MOTA": round(run_pct["MOTA"] - base_pct["MOTA"], 3),
            "IDSW": run_pct["IDSW"] - base_pct["IDSW"],
            "Frag": run_pct["Frag"] - base_pct["Frag"],
        }
        row = {
            "preset": name,
            "params": {
                "appearance_threshold": params.appearance_threshold,
                "max_gap_frames": params.max_gap_frames,
                "max_center_distance_px": params.max_center_distance_px,
                "ema_alpha": params.ema_alpha,
                "hist_h_bins": params.hist_h_bins,
                "hist_s_bins": params.hist_s_bins,
            },
            "metrics": run_pct,
            "delta_vs_baseline": delta,
            "unique_ids_in_mot_file": n_ids,
            "boxes_in_mot_file": n_boxes,
            "remap_info": info,
            "summary_path": metrics["summary_path"],
        }
        results["runs"].append(row)
        print(row["metrics"], "delta", delta, "remaps", info["remap_events"], flush=True)

    (OUT / "appearance_assist_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(OUT / "appearance_assist_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
