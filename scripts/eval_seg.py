"""Segmentation mIoU on BDD100K 10k-val against the sem-seg masks.

Runs the given segmentation models (config keys) exactly as deployed (via
ModelRunner, CUDA EP) over the first N val frames that have masks, and
accumulates a 19-class confusion matrix (GT 255 = ignore). mIoU is
dataset-level: per-class IoU from the accumulated matrix, averaged over
classes that occur. It is not an average of per-image mIoUs.

Usage:  python scripts/eval_seg.py --models seg_fused_576x1024 seg_fused_288x512 --n 200
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
from bsched.metrics import confusion, miou
from bsched.models import CITYSCAPES_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/stage2b/seg_miou.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    gt_dir = resolve(cfg, cfg["data"]["seg_gt_dir"])

    # Frames whose image size differs from frame_hw are excluded and listed:
    # BDD100K 10k-val has 2 portrait (720x1280) phone frames whose masks are
    # landscape canvases with the unlabelled area filled as class 0 ("road").
    frame_hw = tuple(cfg["data"]["frame_hw"])
    paths, excluded = [], []
    for p in sorted(glob.glob(str(resolve(cfg, cfg["data"]["frames_glob"])))):
        if len(paths) == args.n:
            break
        if not (gt_dir / f"{Path(p).stem}_train_id.png").exists():
            continue
        if cv2.imread(p).shape[:2] != frame_hw:
            excluded.append(Path(p).name)
            continue
        paths.append(p)
    cudart = Cudart(cfg["runtime"].get("device_id", 0))
    out_path = resolve(cfg, args.out)
    report = json.loads(out_path.read_text()) if out_path.exists() else {}

    for key in args.models:
        runner = ModelRunner(cfg, key, cudart)
        conf = np.zeros((19, 19), np.int64)
        t0 = time.time()
        for p in paths:
            mask, _ = runner.step(cv2.imread(p))
            gt = cv2.imread(str(gt_dir / f"{Path(p).stem}_train_id.png"), cv2.IMREAD_UNCHANGED)
            conf += confusion(mask, gt, 19)
        r = miou(conf)
        r["per_class_iou"] = dict(zip(CITYSCAPES_NAMES, r["per_class_iou"]))
        report[key] = {"model": cfg["models"][key]["name"], "input_hw": cfg["models"][key]["input_hw"],
                       "n_frames": len(paths), "frames_from": relpath(cfg, cfg["data"]["frames_glob"]),
                       "excluded_frames": excluded,
                       **r, "eval_wall_s": round(time.time() - t0, 1)}
        print(f"{key}: mIoU={r['miou']:.4f} pixel_acc={r['pixel_acc']:.4f} (n={len(paths)})", flush=True)
        del runner

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
