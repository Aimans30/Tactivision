"""Controlled ByteTrack track_buffer sweep on SNMOT-101. Experiment-only; does not change defaults."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

# TrackEval NumPy 2.x shim
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

from trackeval import Evaluator
from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import CLEAR, HOTA, Identity

from tactivision.pipeline.runner import run_tracking

EXP = Path("data/evaluation/SNMOT-101/persistence_sweep")
VIDEO = Path("data/evaluation/SNMOT-101/SNMOT-101.mp4")
GT_PERSONS = Path("data/evaluation/SNMOT-101/gt_persons.txt")
SEQINFO = Path("data/evaluation/snmot/SNMOT-101/seqinfo.ini")
OUT_ROOT = Path("data/evaluation")
BUFFERS = (30, 60, 120)


def tracks_jsonl_to_mot(tracks_path: Path, mot_path: Path) -> tuple[int, int]:
    ids: set[int] = set()
    boxes = 0
    lines: list[str] = []
    with tracks_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            mot_frame = int(row["frame_id"]) + 1
            for tr in row["tracks"]:
                tid = tr.get("track_id")
                if tid is None:
                    continue
                x1, y1, x2, y2 = tr["bbox"]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                conf = float(tr.get("confidence", 1.0))
                lines.append(
                    f"{mot_frame},{int(tid)},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{conf:.4f},1,1"
                )
                ids.add(int(tid))
                boxes += 1
    mot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(ids), boxes


def evaluate_mot(mot_path: Path, run_tag: str) -> dict:
    te = EXP / "trackeval" / run_tag
    if te.exists():
        shutil.rmtree(te)
    gt_seq = te / "gt" / "SNMOT-train" / "SNMOT-101"
    trk_data = te / "trackers" / "SNMOT-train" / run_tag / "data"
    seqmap = te / "gt" / "seqmaps" / "SNMOT-train.txt"
    out_eval = te / "output"
    (gt_seq / "gt").mkdir(parents=True)
    trk_data.mkdir(parents=True)
    seqmap.parent.mkdir(parents=True)
    out_eval.mkdir(parents=True)

    shutil.copy2(SEQINFO, gt_seq / "seqinfo.ini")
    shutil.copy2(GT_PERSONS, gt_seq / "gt" / "gt.txt")
    seqmap.write_text("name\nSNMOT-101\n", encoding="utf-8")
    shutil.copy2(mot_path, trk_data / "SNMOT-101.txt")

    eval_config = Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "DISPLAY_LESS_PROGRESS": True,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
        }
    )
    dataset_config = MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(te / "gt"),
            "TRACKERS_FOLDER": str(te / "trackers"),
            "OUTPUT_FOLDER": str(out_eval),
            "TRACKERS_TO_EVAL": [run_tag],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "SNMOT",
            "SPLIT_TO_EVAL": "train",
            "DO_PREPROC": False,
            "SEQMAP_FILE": str(seqmap),
            "TRACKER_SUB_FOLDER": "data",
            "PRINT_CONFIG": False,
        }
    )
    metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
        "PRINT_CONFIG": False,
    }
    output_res, output_msg = Evaluator(eval_config).evaluate(
        [MotChallenge2DBox(dataset_config)],
        [HOTA(metrics_config), CLEAR(metrics_config), Identity(metrics_config)],
    )
    seq = output_res["MotChallenge2DBox"][run_tag]["SNMOT-101"]["pedestrian"]

    def hota_mean(arr) -> float:
        return float(np.mean(np.asarray(arr, dtype=float)))

    summary_path = out_eval / run_tag / "pedestrian_summary.txt"
    summary_pct: dict[str, float] = {}
    if summary_path.exists():
        hdr, vals = summary_path.read_text(encoding="utf-8").strip().splitlines()[:2]
        keys = hdr.split()
        nums = [float(x) for x in vals.split()]
        summary_pct = dict(zip(keys, nums, strict=False))

    return {
        "HOTA": hota_mean(seq["HOTA"]["HOTA"]),
        "IDF1": float(seq["Identity"]["IDF1"]),
        "MOTA": float(seq["CLEAR"]["MOTA"]),
        "ID_switches": int(seq["CLEAR"]["IDSW"]),
        "fragmentation": int(seq["CLEAR"]["Frag"]),
        "CLR_TP": int(seq["CLEAR"]["CLR_TP"]),
        "CLR_FP": int(seq["CLEAR"]["CLR_FP"]),
        "CLR_FN": int(seq["CLEAR"]["CLR_FN"]),
        "evaluator_Dets": int(seq["Count"]["Dets"]),
        "evaluator_GT_Dets": int(seq["Count"]["GT_Dets"]),
        "evaluator_IDs": int(seq["Count"]["IDs"]),
        "evaluator_GT_IDs": int(seq["Count"]["GT_IDs"]),
        "HOTA_percent": summary_pct.get("HOTA"),
        "IDF1_percent": summary_pct.get("IDF1"),
        "MOTA_percent": summary_pct.get("MOTA"),
        "output_msg": {str(k): str(v) for k, v in (output_msg or {}).items()}
        if isinstance(output_msg, dict)
        else str(output_msg),
        "summary_path": str(summary_path),
    }


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    results = {
        "sequence": "SNMOT-101",
        "video": str(VIDEO),
        "varied_parameter": "track_buffer",
        "held_constant": [
            "models/detection/yolo11n.pt",
            "confidence=0.35",
            "imgsz=1280",
            "track_high_thresh",
            "track_low_thresh",
            "new_track_thresh",
            "match_thresh",
            "fuse_score",
            "evaluation protocol (TrackEval MotChallenge2DBox, DO_PREPROC=False, IoU 0.5 CLEAR, HOTA 0.05-0.95)",
            "gt_persons.txt",
        ],
        "runs": [],
    }

    for buf in BUFFERS:
        tracker_yaml = EXP / f"bytetrack_tb{buf}.yaml"
        run_tag = f"bytetrack_tb{buf}"
        print(f"\n=== track_buffer={buf} ===", flush=True)
        t0 = time.perf_counter()
        tracker_dir = run_tracking(
            VIDEO,
            OUT_ROOT,
            model_path="models/detection/yolo11n.pt",
            tracker_name=str(tracker_yaml.resolve()),
            confidence=0.35,
            image_size=1280,
            max_seconds=30.0,
            device="auto",
        )
        runtime_s = time.perf_counter() - t0
        # run_tracking puts outputs under SNMOT-101/<yaml_stem>/
        # yaml stem may be absolute-path stem on Windows - check
        tracks = tracker_dir / "tracks.jsonl"
        if not tracks.exists():
            # fallback: find by buffer tag
            cand = Path("data/evaluation/SNMOT-101") / run_tag / "tracks.jsonl"
            if cand.exists():
                tracker_dir = cand.parent
                tracks = cand
            else:
                raise FileNotFoundError(f"tracks not found after run; tracker_dir={tracker_dir}")

        mot_path = EXP / f"predictions_tb{buf}.txt"
        n_tracks, n_boxes = tracks_jsonl_to_mot(tracks, mot_path)
        metrics = evaluate_mot(mot_path, run_tag)
        row = {
            "track_buffer": buf,
            "multiplier_vs_baseline": buf / 30,
            "tracker_yaml": str(tracker_yaml),
            "tracks_jsonl": str(tracks),
            "predictions_mot": str(mot_path),
            "runtime_seconds": round(runtime_s, 2),
            "predicted_tracks": n_tracks,
            "predicted_boxes": n_boxes,
            **metrics,
        }
        results["runs"].append(row)
        print(
            f"tb={buf} HOTA%={row['HOTA_percent']} IDF1%={row['IDF1_percent']} "
            f"MOTA%={row['MOTA_percent']} IDSW={row['ID_switches']} Frag={row['fragmentation']} "
            f"tracks={n_tracks} boxes={n_boxes} runtime={row['runtime_seconds']}s",
            flush=True,
        )

    baseline = results["runs"][0]
    for row in results["runs"]:
        row["delta_vs_baseline"] = {
            "ID_switches": row["ID_switches"] - baseline["ID_switches"],
            "fragmentation": row["fragmentation"] - baseline["fragmentation"],
            "HOTA": row["HOTA"] - baseline["HOTA"],
            "IDF1": row["IDF1"] - baseline["IDF1"],
            "MOTA": row["MOTA"] - baseline["MOTA"],
            "predicted_tracks": row["predicted_tracks"] - baseline["predicted_tracks"],
            "predicted_boxes": row["predicted_boxes"] - baseline["predicted_boxes"],
        }

    out_json = EXP / "persistence_sweep_results.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_json)


if __name__ == "__main__":
    main()
