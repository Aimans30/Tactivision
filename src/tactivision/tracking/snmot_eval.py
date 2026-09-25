"""Shared SNMOT-101 TrackEval helpers (MotChallenge2DBox, DO_PREPROC=False)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

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

EVAL_ROOT = Path("data/evaluation/SNMOT-101")
GT_PERSONS = EVAL_ROOT / "gt_persons.txt"
SEQINFO = Path("data/evaluation/snmot/SNMOT-101/seqinfo.ini")


def tracks_jsonl_to_mot(tracks_path: Path, mot_path: Path) -> tuple[int, int]:
    ids: set[int] = set()
    boxes = 0
    lines: list[str] = []
    with tracks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            mot_frame = int(row["frame_id"]) + 1
            for tr in row["tracks"]:
                tid = tr.get("track_id")
                if tid is None:
                    continue
                x1, y1, x2, y2 = tr["bbox"]
                w = max(0.0, float(x2) - float(x1))
                h = max(0.0, float(y2) - float(y1))
                conf = float(tr.get("confidence", 1.0))
                lines.append(
                    f"{mot_frame},{int(tid)},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{conf:.4f},1,1"
                )
                ids.add(int(tid))
                boxes += 1
    mot_path.parent.mkdir(parents=True, exist_ok=True)
    mot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(ids), boxes


def evaluate_mot(mot_path: Path, run_tag: str, work_root: Path) -> dict:
    te = work_root / "trackeval" / run_tag
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
        summary_pct = dict(zip(hdr.split(), [float(x) for x in vals.split()], strict=False))

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
        "evaluator_IDs": int(seq["Count"]["IDs"]),
        "HOTA_percent": summary_pct.get("HOTA"),
        "IDF1_percent": summary_pct.get("IDF1"),
        "MOTA_percent": summary_pct.get("MOTA"),
        "IDSW_summary": summary_pct.get("IDSW"),
        "Frag_summary": summary_pct.get("Frag"),
        "output_msg": {str(k): str(v) for k, v in (output_msg or {}).items()}
        if isinstance(output_msg, dict)
        else str(output_msg),
        "summary_path": str(summary_path),
    }
