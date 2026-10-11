"""Detection mAP on BDD100K 100k-val (10 000 frames) against the v1 box labels.

YOLOv10n is COCO-trained, so BDD100K categories are mapped to COCO classes
([stage5].det_class_map, fixed before any evaluation). BDD "rider" counts as
COCO "person"; BDD "traffic sign" has no COCO class and is not evaluated.
Detections of COCO classes outside the map are dropped. Absolute mAP
therefore measures a COCO model on a different domain and label policy; what
Stage 5 compares is the change between precision variants on the same frames.

AP is COCO-style (bsched.metrics.coco_ap: IoU 0.50:0.95, 101 recall points,
top-100 per image and class), with every one of YOLOv10's 300 outputs that
scores >= [stage5].det_score_threshold. Models run exactly as deployed
(ModelRunner: fused graph, raw frame in). Frames that are not 1280x720 are
excluded and listed.

Raw predictions go to results/stage5/det_preds/<key>.npz (not committed)
so the metric can be recomputed or cross-checked without re-running.

Usage:  python scripts/eval_det.py --models detector_fused@trt_fp16 [--n 500]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from bsched.config import load_config, relpath, resolve
from bsched.engine import ModelRunner
from bsched.gpu import Cudart
from bsched.metrics import coco_ap
from bsched.models import COCO_NAMES


def eval_classes(cfg):
    """Evaluated classes, in COCO index order, and the BDD -> class-index map."""
    cmap = cfg["stage5"]["det_class_map"]
    names = sorted(set(cmap.values()), key=COCO_NAMES.index)
    return names, {bdd: names.index(coco) for bdd, coco in cmap.items()}


def load_frames_and_gt(cfg, n: int | None):
    s5 = cfg["stage5"]
    names, bdd_to_idx = eval_classes(cfg)
    lab_dir = resolve(cfg, s5["det_labels_dir"])
    hw = tuple(cfg["data"]["frame_hw"])
    paths, gts, excluded, unmapped = [], [], [], {}
    for p in sorted(glob.glob(str(resolve(cfg, s5["det_frames_glob"])))):
        if n and len(paths) == n:
            break
        img = cv2.imread(p)
        if img.shape[:2] != hw:
            excluded.append(Path(p).name)
            continue
        lab = json.loads((lab_dir / f"{Path(p).stem}.json").read_text())
        boxes = []
        for fr in lab.get("frames", []):
            for o in fr.get("objects", []):
                if "box2d" not in o:
                    continue
                c = bdd_to_idx.get(o["category"])
                if c is None:
                    unmapped[o["category"]] = unmapped.get(o["category"], 0) + 1
                    continue
                b = o["box2d"]
                boxes.append([b["x1"], b["y1"], b["x2"], b["y2"], c])
        paths.append(p)
        gts.append(np.array(boxes, np.float64).reshape(-1, 5))
    return paths, gts, excluded, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=None, help="first N frames only (smoke test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    s5 = cfg["stage5"]
    out_dir = resolve(cfg, s5["out_dir"])
    out_path = resolve(cfg, args.out) if args.out else out_dir / "det_map.json"
    pred_dir = out_dir / "det_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    names, _ = eval_classes(cfg)
    coco_to_idx = {COCO_NAMES.index(c): i for i, c in enumerate(names)}

    t0 = time.time()
    paths, gts, excluded, unmapped = load_frames_and_gt(cfg, args.n)
    print(f"{len(paths)} frames, {sum(len(g) for g in gts)} GT boxes, excluded {len(excluded)}, "
          f"loaded in {time.time() - t0:.0f}s", flush=True)
    cudart = Cudart(cfg["runtime"].get("device_id", 0))
    report = json.loads(out_path.read_text()) if out_path.exists() else {}

    for key in args.models:
        runner = ModelRunner(cfg, key, cudart)
        runner.mcfg = dict(runner.mcfg, score_threshold=s5["det_score_threshold"])
        dets = []
        t0 = time.time()
        for p in paths:
            d, _ = runner.step(cv2.imread(p))
            keep = np.isin(d[:, 5], list(coco_to_idx))
            d = d[keep].astype(np.float64)
            d[:, 5] = [coco_to_idx[int(c)] for c in d[:, 5]]
            dets.append(d)
        t_inf = time.time() - t0
        del runner
        np.savez_compressed(pred_dir / f"{key.replace('@', '__')}.npz",
                            dets=np.concatenate(dets) if dets else np.zeros((0, 6)),
                            image_index=np.repeat(np.arange(len(dets)), [len(d) for d in dets]),
                            frames=np.array([Path(p).name for p in paths]))
        t0 = time.time()
        r = coco_ap(dets, gts, len(names), s5["det_max_dets"])
        r["per_class"] = dict(zip(names, r["per_class"]))
        report[key] = {"model": cfg["models"][key]["name"], "n_frames": len(paths),
                       "frames_from": s5["det_frames_glob"], "excluded_frames": excluded,
                       "unmapped_gt_boxes": unmapped, "classes": names,
                       "score_threshold": s5["det_score_threshold"], "max_dets": s5["det_max_dets"],
                       **r, "inference_wall_s": round(t_inf, 1), "ap_wall_s": round(time.time() - t0, 1)}
        print(f"{key}: mAP={r['mAP']:.4f} mAP50={r['mAP50']:.4f} mAP75={r['mAP75']:.4f} "
              f"(n={len(paths)}, {t_inf:.0f}s infer)", flush=True)
        out_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
