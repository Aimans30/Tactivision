"""One-off SNMOT-101 identity failure analysis. Does not modify the tracker."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Archived experiment script. Evidence outputs remain under data/evaluation/SNMOT-101/.
EVAL = Path("data/evaluation/SNMOT-101")
GT_PATH = EVAL / "gt_persons.txt"
PRED_PATH = EVAL / "predictions_mot.txt"
GAMEINFO = Path("data/evaluation/snmot/SNMOT-101/gameinfo.ini")
IMG_DIR = Path("data/evaluation/snmot/SNMOT-101/img1")
OUT_DIR = EVAL  # write analysis JSON next to existing evidence
W, H = 1920, 1080
IOU_THR = 0.5
N_FRAMES = 750


def load_labels() -> dict[int, str]:
    labels: dict[int, str] = {}
    for ln in GAMEINFO.read_text(encoding="utf-8").splitlines():
        if ln.startswith("trackletID_"):
            k, v = ln.split("=", 1)
            labels[int(k.split("_")[1])] = v.strip()
    return labels


def load_mot(path: Path) -> dict[int, list[dict]]:
    by_f: dict[int, list[dict]] = defaultdict(list)
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        p = ln.split(",")
        f = int(float(p[0]))
        tid = int(float(p[1]))
        x, y, w, h = map(float, p[2:6])
        by_f[f].append(
            {
                "id": tid,
                "xywh": (x, y, w, h),
                "cx": x + w / 2,
                "cy": y + h / 2,
                "area": w * h,
            }
        )
    return by_f


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def dist(a, b) -> float:
    return float(np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]))


def near_border(box, margin=40) -> bool:
    x, y, w, h = box["xywh"]
    return x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin)


def phase_shift(f0: int, f1: int):
    a = cv2.imread(str(IMG_DIR / f"{f0:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(IMG_DIR / f"{f1:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    a = cv2.resize(a, (480, 270)).astype(np.float32)
    b = cv2.resize(b, (480, 270)).astype(np.float32)
    shift, response = cv2.phaseCorrelate(a, b)
    return shift, float(response)


def classify_idsw(e: dict) -> tuple[str, list[str]]:
    tags: list[str] = []
    if e["swap_partner"] is not None and e["fn_in_gap"] == 0:
        tags.append("crossing_id_swap")
    if e["fn_in_gap"] >= 1:
        tags.append("missed_detection_gap")
    if e["occlusion_proxy"] and e["fn_in_gap"] >= 1:
        tags.append("short_occlusion_proxy")
    diag = e["gt_box_diag"] or 1.0
    if e["gt_disp_px"] is not None and e["gt_disp_px"] > 0.5 * diag and e["gap_frames"] <= 2:
        tags.append("large_gt_displacement")
    if e["near_border"] and (e["fn_in_gap"] >= 1 or e["gap_frames"] <= 2):
        tags.append("enter_leave_border")
    if e["fn_in_gap"] == 0 and e["gap_frames"] <= 1 and e["swap_partner"] is None:
        if e["old_pred_still_present"]:
            tags.append("association_failure_both_preds_present")
        else:
            tags.append("association_failure_old_pred_gone")
    if e["other_pred_in_gap"] >= 1 and e["fn_in_gap"] >= 1:
        tags.append("identity_changed_during_gap")
    if not tags:
        tags.append("other_unclassified")

    if "crossing_id_swap" in tags:
        primary = "crossing_id_swap"
    elif "missed_detection_gap" in tags and e["fn_in_gap"] >= 3:
        primary = "missed_detection_gap"
    elif "short_occlusion_proxy" in tags:
        primary = "short_occlusion_proxy"
    elif "missed_detection_gap" in tags:
        primary = "missed_detection_gap"
    elif "large_gt_displacement" in tags:
        primary = "large_gt_displacement"
    elif "enter_leave_border" in tags:
        primary = "enter_leave_border"
    elif "association_failure_both_preds_present" in tags:
        primary = "association_failure_both_preds_present"
    elif "association_failure_old_pred_gone" in tags:
        primary = "association_failure_old_pred_gone"
    else:
        primary = tags[0]
    return primary, tags


def main() -> None:
    labels = load_labels()
    gt = load_mot(GT_PATH)
    pr = load_mot(PRED_PATH)

    all_gt_ids = sorted({d["id"] for f in gt.values() for d in f})
    all_pr_ids = sorted({d["id"] for f in pr.values() for d in f})
    gt_id_map = {g: i for i, g in enumerate(all_gt_ids)}
    pr_id_map = {p: i for i, p in enumerate(all_pr_ids)}
    num_gt_ids = len(all_gt_ids)

    frames_gt_ids = []
    frames_pr_ids = []
    frames_sim = []
    frames_gt_objs = []
    frames_pr_objs = []
    for t in range(1, N_FRAMES + 1):
        gts = gt.get(t, [])
        prs = pr.get(t, [])
        frames_gt_objs.append(gts)
        frames_pr_objs.append(prs)
        gids = np.array([gt_id_map[d["id"]] for d in gts], dtype=int)
        pids = np.array([pr_id_map[d["id"]] for d in prs], dtype=int)
        frames_gt_ids.append(gids)
        frames_pr_ids.append(pids)
        sim = np.zeros((len(gts), len(prs)), dtype=float)
        for i, g in enumerate(gts):
            for j, p in enumerate(prs):
                sim[i, j] = iou_xywh(g["xywh"], p["xywh"])
        frames_sim.append(sim)

    prev_tracker_id = np.full(num_gt_ids, np.nan)
    prev_timestep_tracker_id = np.full(num_gt_ids, np.nan)
    gt_frag_count = np.zeros(num_gt_ids)
    idsw_events: list[dict] = []
    frag_events: list[dict] = []
    match_history: dict[int, list[tuple[int, int | None, float]]] = defaultdict(list)
    last_match_frame: dict[int, int] = {}
    last_match_gt_box: dict[int, dict] = {}

    idsw_total = 0
    tp = fp = fn = 0

    for t_idx in range(N_FRAMES):
        frame = t_idx + 1
        gids = frames_gt_ids[t_idx]
        pids = frames_pr_ids[t_idx]
        sim = frames_sim[t_idx]
        gts = frames_gt_objs[t_idx]
        prs = frames_pr_objs[t_idx]

        if len(gids) == 0:
            fp += len(pids)
            continue
        if len(pids) == 0:
            fn += len(gids)
            for gi in gids:
                match_history[all_gt_ids[int(gi)]].append((frame, None, 0.0))
            prev_timestep_tracker_id[:] = np.nan
            continue

        score_mat = pids[np.newaxis, :] == prev_timestep_tracker_id[gids[:, np.newaxis]]
        score_mat = 1000 * score_mat + sim
        score_mat[sim < IOU_THR - np.finfo("float").eps] = 0
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        actually = score_mat[match_rows, match_cols] > 0 + np.finfo("float").eps
        match_rows = match_rows[actually]
        match_cols = match_cols[actually]

        matched_gt_ids = gids[match_rows]
        matched_pr_ids = pids[match_cols]
        matched_ious = sim[match_rows, match_cols]

        prev_matched = prev_tracker_id[matched_gt_ids]
        is_idsw = (~np.isnan(prev_matched)) & (matched_pr_ids != prev_matched)
        idsw_total += int(np.sum(is_idsw))

        matched_gi_set = {int(x) for x in matched_gt_ids}
        matched_gi_to_pos = {int(gi): mpos for mpos, gi in enumerate(matched_gt_ids)}

        for i, gobj in enumerate(gts):
            gi = int(gids[i])
            raw = all_gt_ids[gi]
            if gi in matched_gi_set:
                mpos = matched_gi_to_pos[gi]
                pid_r = all_pr_ids[int(matched_pr_ids[mpos])]
                iou = float(matched_ious[mpos])
                match_history[raw].append((frame, pid_r, iou))
            else:
                match_history[raw].append((frame, None, 0.0))

        for mpos, gi in enumerate(matched_gt_ids):
            gi = int(gi)
            raw_gt = all_gt_ids[gi]
            new_pr = all_pr_ids[int(matched_pr_ids[mpos])]
            iou = float(matched_ious[mpos])
            gobj = next(g for g in gts if g["id"] == raw_gt)
            pobj = next(p for p in prs if p["id"] == new_pr)

            if is_idsw[mpos]:
                old_pr = all_pr_ids[int(prev_matched[mpos])]
                hist = match_history[raw_gt]
                gap_frames = []
                for fr, pid, iou_h in reversed(hist[:-1]):
                    if pid == old_pr:
                        break
                    gap_frames.append((fr, pid, iou_h))
                gap_frames = list(reversed(gap_frames))
                fn_in_gap = sum(1 for _, pid, _ in gap_frames if pid is None)
                other_pred_in_gap = sum(
                    1 for _, pid, _ in gap_frames if pid is not None and pid != old_pr
                )

                near_gts = []
                for og in gts:
                    if og["id"] == raw_gt:
                        continue
                    d = dist(gobj, og)
                    if d < 120:
                        near_gts.append((og["id"], d, iou_xywh(gobj["xywh"], og["xywh"])))

                swap_partner = None
                for og_id, d, oi in near_gts:
                    og_hist = match_history[og_id]
                    if og_hist and og_hist[-1][0] == frame and og_hist[-1][1] == old_pr:
                        swap_partner = og_id
                        break

                prev_box = last_match_gt_box.get(gi)
                gt_disp = dist(gobj, prev_box) if prev_box else None
                old_pr_now = next((p for p in prs if p["id"] == old_pr), None)
                new_vs_old_iou = (
                    iou_xywh(pobj["xywh"], old_pr_now["xywh"]) if old_pr_now else None
                )

                occl = False
                for fr, pid, _ in gap_frames:
                    if pid is not None or fr not in gt:
                        continue
                    self_box = next(g["xywh"] for g in gt[fr] if g["id"] == raw_gt)
                    for og in gt[fr]:
                        if og["id"] == raw_gt:
                            continue
                        if iou_xywh(self_box, og["xywh"]) >= 0.3:
                            occl = True
                            break
                    if occl:
                        break

                idsw_events.append(
                    {
                        "frame": frame,
                        "gt_id": raw_gt,
                        "gt_label": labels.get(raw_gt, "?"),
                        "old_pred": old_pr,
                        "new_pred": new_pr,
                        "match_iou": round(iou, 3),
                        "gap_frames": frame - last_match_frame.get(gi, frame),
                        "fn_in_gap": fn_in_gap,
                        "other_pred_in_gap": other_pred_in_gap,
                        "gt_disp_px": None if gt_disp is None else round(gt_disp, 1),
                        "gt_box_diag": round(
                            float(np.hypot(gobj["xywh"][2], gobj["xywh"][3])), 1
                        ),
                        "near_gts": [
                            (a, round(b, 1), round(c, 3)) for a, b, c in near_gts[:5]
                        ],
                        "swap_partner": swap_partner,
                        "near_border": near_border(gobj),
                        "occlusion_proxy": occl,
                        "old_pred_still_present": old_pr_now is not None,
                        "new_vs_old_pred_iou": None
                        if new_vs_old_iou is None
                        else round(new_vs_old_iou, 3),
                        "gt_xywh": [round(x, 1) for x in gobj["xywh"]],
                        "new_pred_xywh": [round(x, 1) for x in pobj["xywh"]],
                    }
                )

            last_match_frame[gi] = frame
            last_match_gt_box[gi] = gobj

        not_previously_tracked = np.isnan(prev_timestep_tracker_id)
        prev_tracker_id[matched_gt_ids] = matched_pr_ids
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_pr_ids
        currently_tracked = ~np.isnan(prev_timestep_tracker_id)
        became = np.logical_and(not_previously_tracked, currently_tracked)

        for gi in np.where(became)[0]:
            gi = int(gi)
            raw_gt = all_gt_ids[gi]
            mpos = matched_gi_to_pos.get(gi)
            if mpos is None:
                continue
            new_pr = all_pr_ids[int(matched_pr_ids[mpos])]
            iou = float(matched_ious[mpos])
            hist = match_history[raw_gt]
            gap_len = 0
            for fr, pid, _ in reversed(hist[:-1]):
                if pid is None:
                    gap_len += 1
                else:
                    break
            gobj = next(g for g in gts if g["id"] == raw_gt)
            frag_events.append(
                {
                    "frame": frame,
                    "gt_id": raw_gt,
                    "gt_label": labels.get(raw_gt, "?"),
                    "pred_id": new_pr,
                    "match_iou": round(iou, 3),
                    "unmatched_gap": gap_len,
                    "near_border": near_border(gobj),
                }
            )
            gt_frag_count[gi] += 1

        tp += len(matched_gt_ids)
        fn += len(gids) - len(matched_gt_ids)
        fp += len(pids) - len(matched_gt_ids)

    frag_total = int(np.sum(np.maximum(gt_frag_count[gt_frag_count > 0] - 1, 0)))

    for e in idsw_events:
        primary, tags = classify_idsw(e)
        e["primary"] = primary
        e["tags"] = tags

    frag_by_gt: dict[int, list[dict]] = defaultdict(list)
    for e in frag_events:
        frag_by_gt[e["gt_id"]].append(e)
    counted_frags: list[dict] = []
    for gid, evs in frag_by_gt.items():
        for i, e in enumerate(sorted(evs, key=lambda x: x["frame"])):
            e["counted_in_frag_metric"] = i > 0
            e["restart_index"] = i
            if i == 0:
                continue
            if e["unmatched_gap"] >= 1:
                e["primary"] = "restart_after_missed_detections"
            elif e["near_border"]:
                e["primary"] = "restart_near_border"
            else:
                e["primary"] = "restart_other"
            counted_frags.append(e)

    idsw_primary = Counter(e["primary"] for e in idsw_events)
    frag_primary = Counter(e["primary"] for e in counted_frags)

    gt_med_disp = []
    for t in range(2, N_FRAMES + 1):
        prev = {d["id"]: d for d in gt.get(t - 1, [])}
        cur = {d["id"]: d for d in gt.get(t, [])}
        common = set(prev) & set(cur)
        if not common:
            continue
        disps = [dist(prev[i], cur[i]) for i in common]
        gt_med_disp.append((t, float(np.median(disps))))
    disps_only = [d for _, d in gt_med_disp]
    p95 = float(np.percentile(disps_only, 95))
    high_cam_frames = {t for t, d in gt_med_disp if d >= p95}
    idsw_in_high_cam = sum(1 for e in idsw_events if e["frame"] in high_cam_frames)

    sorted_disp = sorted(gt_med_disp, key=lambda x: -x[1])
    cam_samples = []
    for t, d in sorted_disp[:8] + sorted_disp[len(sorted_disp) // 2 : len(sorted_disp) // 2 + 3]:
        shift, resp = phase_shift(t - 1, t)
        cam_samples.append(
            {
                "frame": t,
                "gt_med_disp": round(d, 2),
                "phase_shift_px_at_480w": [round(shift[0], 2), round(shift[1], 2)],
                "response": round(resp, 3),
            }
        )

    area_ratios = []
    for t in range(2, N_FRAMES + 1):
        prev = {d["id"]: d for d in gt.get(t - 1, [])}
        cur = {d["id"]: d for d in gt.get(t, [])}
        common = set(prev) & set(cur)
        if len(common) < 5:
            continue
        ratios = [cur[i]["area"] / max(prev[i]["area"], 1.0) for i in common]
        area_ratios.append(float(np.median(ratios)))

    pred_match_frames: dict[int, int] = defaultdict(int)
    pred_fp_frames: dict[int, int] = defaultdict(int)
    for t_idx in range(N_FRAMES):
        sim = frames_sim[t_idx]
        prs = frames_pr_objs[t_idx]
        if not prs:
            continue
        if sim.size == 0:
            for p in prs:
                pred_fp_frames[p["id"]] += 1
            continue
        max_iou = sim.max(axis=0)
        for j, p in enumerate(prs):
            if max_iou[j] >= IOU_THR:
                pred_match_frames[p["id"]] += 1
            else:
                pred_fp_frames[p["id"]] += 1
    mostly_fp = [
        pid
        for pid in all_pr_ids
        if pred_fp_frames[pid] > pred_match_frames[pid] and pred_fp_frames[pid] >= 5
    ]
    idsw_new_is_fpish = sum(1 for e in idsw_events if e["new_pred"] in mostly_fp)

    det_related = sum(1 for e in idsw_events if e["fn_in_gap"] >= 1)
    assoc_related = sum(1 for e in idsw_events if e["fn_in_gap"] == 0)

    gap_stats: Counter[str] = Counter()
    for e in counted_frags:
        g = e["unmatched_gap"]
        if g <= 0:
            gap_stats["0"] += 1
        elif g == 1:
            gap_stats["1"] += 1
        elif g <= 5:
            gap_stats["2-5"] += 1
        elif g <= 25:
            gap_stats["6-25"] += 1
        else:
            gap_stats["26+"] += 1

    # Per-GT IDSW / Frag load
    idsw_by_gt = Counter(e["gt_id"] for e in idsw_events)
    frag_by_gt_c = Counter(e["gt_id"] for e in counted_frags)

    out = {
        "verification": {
            "IDSW": idsw_total,
            "Frag": frag_total,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "expected_IDSW": 103,
            "expected_Frag": 232,
        },
        "idsw_primary_counts": dict(idsw_primary),
        "frag_primary_counts": dict(frag_primary),
        "idsw_detection_linked": det_related,
        "idsw_association_linked": assoc_related,
        "frag_gap_buckets": dict(gap_stats),
        "idsw_by_gt": dict(idsw_by_gt),
        "frag_by_gt": dict(frag_by_gt_c),
        "camera_proxy": {
            "gt_centroid_med_disp_mean": round(float(np.mean(disps_only)), 2),
            "gt_centroid_med_disp_p50": round(float(np.percentile(disps_only, 50)), 2),
            "gt_centroid_med_disp_p90": round(float(np.percentile(disps_only, 90)), 2),
            "gt_centroid_med_disp_p95": round(p95, 2),
            "gt_centroid_med_disp_max": round(float(max(disps_only)), 2),
            "idsw_in_high_cam_frames": idsw_in_high_cam,
            "idsw_total": len(idsw_events),
            "phase_correlation_samples": cam_samples,
            "median_gt_area_ratio_p1_p50_p99": [
                round(float(np.percentile(area_ratios, 1)), 4),
                round(float(np.percentile(area_ratios, 50)), 4),
                round(float(np.percentile(area_ratios, 99)), 4),
            ],
        },
        "mostly_fp_pred_tracks": len(mostly_fp),
        "mostly_fp_pred_track_ids": mostly_fp[:30],
        "idsw_new_pred_mostly_fp": idsw_new_is_fpish,
        "idsw_events": idsw_events,
        "counted_frag_events": counted_frags,
    }
    (OUT_DIR / "failure_analysis_data.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )

    print("IDSW", idsw_total, "Frag", frag_total, "TP/FP/FN", tp, fp, fn)
    print("idsw_primary", dict(idsw_primary))
    print("frag_primary", dict(frag_primary))
    print("det_linked", det_related, "assoc_linked", assoc_related)
    print("frag gaps", dict(gap_stats))
    print("cam", out["camera_proxy"])
    print("mostly_fp", len(mostly_fp), "idsw_new_fpish", idsw_new_is_fpish)


if __name__ == "__main__":
    main()
