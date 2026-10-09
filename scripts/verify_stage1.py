"""Stage 1: check both ONNX models run on the CUDA EP and give sane output.

For each of the first N frames this:
  * runs both models on the CUDA EP and, separately, on the CPU EP, and
    reports GPU-vs-CPU numerical agreement (a correctness check on the GPU path),
  * records detection counts/classes and segmentation class fractions,
  * if a BDD100K sem-seg mask exists for the frame, computes pixel accuracy
    as a quick check that the class-id mapping is right (not Stage-5 mIoU),
  * saves overlays for the first few frames.

No latency is measured or reported here; that is Stage 2.

Usage:  python scripts/verify_stage1.py [--config config.toml] [--frames-glob GLOB]
"""
from __future__ import annotations

import argparse
import glob
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort

from bsched.config import load_config, relpath, resolve
from bsched.models import (CITYSCAPES_NAMES, COCO_NAMES, det_postprocess, det_preprocess,
                           make_session, node_placement, seg_postprocess, seg_preprocess)
from bsched.vis import draw_detections, draw_segmentation, side_by_side


def gpu_info() -> dict:
    q = "name,memory.total,driver_version,clocks.max.sm,power.limit"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return dict(zip(q.split(","), [s.strip() for s in out.split(",")]))
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def match_by_iou(a: np.ndarray, b: np.ndarray, thr: float = 0.5):
    """Greedy one-to-one matching of [N,6] boxes by IoU. Returns (pairs, n_unmatched)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area = lambda d: (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])  # noqa: E731
    iou = inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)
    # YOLOv10 can emit the same box twice with different classes; match within class.
    iou = np.where(a[:, None, 5] == b[None, :, 5], iou, 0.0)
    pairs, used_a, used_b = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
        if iou[i, j] < thr:
            break
        if i not in used_a and j not in used_b:
            pairs.append((int(i), int(j), float(iou[i, j])))
            used_a.add(i); used_b.add(j)
    return pairs, (len(a) - len(pairs)) + (len(b) - len(pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--frames-glob", default=None, help="override data.frames_glob")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s1, rt = cfg["stage1"], cfg["runtime"]
    det_cfg, seg_cfg = cfg["models"]["detector"], cfg["models"]["segmenter"]

    pattern = args.frames_glob or str(resolve(cfg, cfg["data"]["frames_glob"]))
    frames = sorted(glob.glob(pattern))[: s1["n_frames"]]
    if not frames:
        sys.exit(f"No frames match {pattern}")
    out_dir = resolve(cfg, s1["out_dir"])
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    det_onnx = str(resolve(cfg, det_cfg["onnx"]))
    seg_onnx = str(resolve(cfg, seg_cfg["onnx"]))
    det_gpu, seg_gpu = make_session(det_onnx, rt, "cuda"), make_session(seg_onnx, rt, "cuda")
    det_cpu, seg_cpu = make_session(det_onnx, rt, "cpu"), make_session(seg_onnx, rt, "cpu")
    det_in, seg_in = det_gpu.get_inputs()[0].name, seg_gpu.get_inputs()[0].name

    first = cv2.imread(frames[0])
    placement = {
        "detector": node_placement(det_onnx, rt, {det_in: det_preprocess(first, det_cfg["input_hw"])[0]}),
        "segmenter": node_placement(seg_onnx, rt, {seg_in: seg_preprocess(first, seg_cfg["input_hw"])}),
    }

    for name, p in placement.items():
        if not p["nodes_by_provider"].get("CUDAExecutionProvider"):
            sys.exit(f"{name}: no nodes executed on CUDAExecutionProvider "
                     f"({p['nodes_by_provider']}). Refusing to report a CPU run as GPU.")

    # BDD100K sem-seg masks, if extracted (train_id PNGs, 255 = ignore).
    seg_gt_dir = resolve(cfg, "data/bdd100k/labels/sem_seg/val")

    per_frame, det_classes, seg_pixels = [], Counter(), np.zeros(19, np.int64)
    for i, path in enumerate(frames):
        bgr = cv2.imread(path)
        hw = bgr.shape[:2]
        xd, lb = det_preprocess(bgr, det_cfg["input_hw"])
        xs = seg_preprocess(bgr, seg_cfg["input_hw"])

        raw_d_gpu = det_gpu.run(None, {det_in: xd})[0]
        raw_d_cpu = det_cpu.run(None, {det_in: xd})[0]
        raw_s_gpu = seg_gpu.run(None, {seg_in: xs})[0]
        raw_s_cpu = seg_cpu.run(None, {seg_in: xs})[0]

        dets = det_postprocess(raw_d_gpu, lb, det_cfg["score_threshold"], hw)
        dets_cpu = det_postprocess(raw_d_cpu, lb, det_cfg["score_threshold"], hw)
        mask = seg_postprocess(raw_s_gpu, hw)

        det_parity = {"n_gpu": len(dets), "n_cpu": len(dets_cpu)}
        if len(dets) and len(dets_cpu):
            # Pair boxes by IoU, not score rank: near-tied scores can swap order.
            pairs, unmatched = match_by_iou(dets, dets_cpu)
            a, b = dets[[p[0] for p in pairs]], dets_cpu[[p[1] for p in pairs]]
            det_parity.update(matched=len(pairs), unmatched=unmatched,
                              min_iou=float(min(p[2] for p in pairs)) if pairs else None,
                              max_box_abs_diff_px=float(np.abs(a[:, :4] - b[:, :4]).max()),
                              max_score_abs_diff=float(np.abs(a[:, 4] - b[:, 4]).max()),
                              classes_equal=bool((a[:, 5] == b[:, 5]).all()))
        seg_parity = {"max_logit_abs_diff": float(np.abs(raw_s_gpu - raw_s_cpu).max()),
                      "argmax_agreement": float((raw_s_gpu.argmax(1) == raw_s_cpu.argmax(1)).mean())}

        cls_counts = Counter(COCO_NAMES[int(c)] for c in dets[:, 5])
        det_classes.update(cls_counts)
        ids, cnt = np.unique(mask, return_counts=True)
        seg_pixels[ids] += cnt

        rec = {"frame": Path(path).name, "hw": list(hw),
               "detector": {"raw_output_shape": list(raw_d_gpu.shape),
                            "finite": bool(np.isfinite(raw_d_gpu).all()),
                            "n_dets": len(dets), "classes": dict(cls_counts),
                            "gpu_vs_cpu": det_parity},
               "segmenter": {"raw_output_shape": list(raw_s_gpu.shape),
                             "finite": bool(np.isfinite(raw_s_gpu).all()),
                             "top_classes": {CITYSCAPES_NAMES[k]: round(float(v) / mask.size, 4)
                                             for k, v in sorted(zip(ids, cnt), key=lambda t: -t[1])[:5]},
                             "gpu_vs_cpu": seg_parity}}

        gt_path = seg_gt_dir / f"{Path(path).stem}_train_id.png"
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            valid = gt != 255
            rec["segmenter"]["sanity_pixel_acc_vs_bdd_gt"] = float((mask[valid] == gt[valid]).mean())
        per_frame.append(rec)

        if i < s1["n_visualize"]:
            vis = side_by_side(draw_detections(bgr, dets), draw_segmentation(bgr, mask))
            cv2.imwrite(str(out_dir / "vis" / f"{i:02d}_{Path(path).stem}.jpg"), vis)

    accs = [r["segmenter"]["sanity_pixel_acc_vs_bdd_gt"] for r in per_frame
            if "sanity_pixel_acc_vs_bdd_gt" in r["segmenter"]]
    summary = {
        "config": relpath(cfg, cfg["_path"]),
        "frames_glob": relpath(cfg, pattern),
        "n_frames": len(frames),
        "env": {"python": platform.python_version(), "os": platform.platform(),
                "onnxruntime": ort.__version__, "available_providers": ort.get_available_providers(),
                "session_providers": det_gpu.get_providers(), "gpu": gpu_info()},
        "node_placement": placement,
        "aggregate": {
            "detector": {"total_dets": int(sum(det_classes.values())),
                         "classes": dict(det_classes.most_common()),
                         "all_finite": all(r["detector"]["finite"] for r in per_frame),
                         "gpu_cpu_count_match_frames": sum(r["detector"]["gpu_vs_cpu"]["n_gpu"] ==
                                                           r["detector"]["gpu_vs_cpu"]["n_cpu"] for r in per_frame)},
            "segmenter": {"pixel_fraction": {CITYSCAPES_NAMES[k]: round(float(v) / seg_pixels.sum(), 4)
                                             for k, v in sorted(enumerate(seg_pixels), key=lambda t: -t[1]) if v},
                          "all_finite": all(r["segmenter"]["finite"] for r in per_frame),
                          "min_gpu_cpu_argmax_agreement": min(r["segmenter"]["gpu_vs_cpu"]["argmax_agreement"] for r in per_frame),
                          "sanity_pixel_acc_vs_bdd_gt": {"n_frames_with_gt": len(accs),
                                                         "mean": float(np.mean(accs)) if accs else None}},
        },
        "per_frame": per_frame,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_frames", "env", "node_placement", "aggregate")}, indent=2))
    print(f"\nWrote {out_dir / 'summary.json'} and {out_dir / 'vis'}")


if __name__ == "__main__":
    main()
