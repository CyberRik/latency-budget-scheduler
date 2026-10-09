"""Stage 2b: check each fused model matches the CPU-path pipeline it replaces.

Both paths run on the CUDA EP over the same frames:
  * detector_fused   vs  CPU pre/post + models.detector graph
  * seg_fused_<HxW>  vs  CPU pre/post + the same-resolution source graph

Reports detection agreement (IoU matching within class) and segmentation
mask pixel agreement. Residual differences come from float vs uint8-rounded
resize (see build_fused.py).

Usage:  python scripts/verify_fused.py [--config config.toml] [--n 20]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from bsched.config import load_config, resolve
from bsched.engine import ModelRunner
from bsched.gpu import Cudart
from bsched.metrics import det_agreement
from bsched.models import (det_postprocess, det_preprocess, make_session, seg_postprocess,
                           seg_preprocess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rt = cfg["runtime"]
    frames = [cv2.imread(p) for p in sorted(glob.glob(str(resolve(cfg, cfg["data"]["frames_glob"]))))[: args.n]]
    cudart = Cudart(rt.get("device_id", 0))

    report = {}
    for key, m in cfg["models"].items():
        if not m.get("fused"):
            continue
        fused = ModelRunner(cfg, key, cudart)
        ref = make_session(str(resolve(cfg, m["source_onnx"])), rt, "cuda")
        in_name = ref.get_inputs()[0].name
        rows = []
        for bgr in frames:
            got, _ = fused.step(bgr)
            hw = bgr.shape[:2]
            if m["task"] == "detection":
                x, lb = det_preprocess(bgr, m["input_hw"])
                want = det_postprocess(ref.run(None, {in_name: x})[0], lb, m["score_threshold"], hw)
                rows.append(det_agreement(got, want))
            else:
                want = seg_postprocess(ref.run(None, {in_name: seg_preprocess(bgr, m["input_hw"])})[0], hw)
                rows.append({"pixel_agreement": float((got == want).mean())})
        if m["task"] == "detection":
            summ = {"frames": len(rows), "dets_fused": sum(r["n_a"] for r in rows),
                    "dets_cpu_path": sum(r["n_b"] for r in rows),
                    "matched": sum(r["matched"] for r in rows), "unmatched": sum(r["unmatched"] for r in rows),
                    "min_iou": min((r["min_iou"] for r in rows if "min_iou" in r), default=None),
                    "max_box_abs_diff_px": max((r["max_box_abs_diff_px"] for r in rows if "max_box_abs_diff_px" in r), default=None),
                    "max_score_abs_diff": max((r["max_score_abs_diff"] for r in rows if "max_score_abs_diff" in r), default=None)}
        else:
            agr = [r["pixel_agreement"] for r in rows]
            summ = {"frames": len(rows), "mean_pixel_agreement": float(np.mean(agr)),
                    "min_pixel_agreement": float(np.min(agr))}
        report[key] = {"summary": summ, "per_frame": rows}
        print(key, json.dumps(summ))
        del fused, ref

    out = resolve(cfg, "results/stage2b")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fused_equivalence.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
