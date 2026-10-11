"""Stage 5 checks before any timing: for every precision variant,

  - session creation time (first run builds the TensorRT engine and caches it;
    the build time is reported as a one-off cost, separately from latency),
  - node placement: which execution provider ran each node (ORT profiler).
    TensorRT runs a fused subgraph as one node; a node placed on CPU fails
    the check,
  - output agreement with the CUDA FP32 reference on the first N frames
    (detection: IoU matching within class at score >= 0.25; segmentation:
    fraction of pixels with the same class).

Usage:  python scripts/verify_stage5.py [--n 20] [--models detector_fused ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bsched.config import load_config, resolve
from bsched.engine import ModelRunner
from bsched.gpu import Cudart
from bsched.metrics import det_agreement
from bsched.models import node_placement
from bsched.stream import FrameSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s5 = cfg["stage5"]
    src = FrameSource(str(resolve(cfg, cfg["data"]["frames_glob"])), args.n, cfg["harness"]["seed"],
                      tuple(cfg["data"]["frame_hw"]))
    cudart = Cudart(cfg["runtime"].get("device_id", 0))
    out_path = resolve(cfg, s5["out_dir"]) / "verify.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    failures = []

    for base in args.models or s5["models"]:
        ref = None
        for v in s5["variants"]:
            key = f"{base}@{v}"
            m = cfg["models"][key]
            cache_hit = None
            if "trt" in m:
                stem = Path(m["onnx"]).stem
                cdir = resolve(cfg, cfg["tensorrt"]["engine_cache_dir"]) / f"{stem}__{m['trt']['precision']}"
                cache_hit = any(cdir.glob("*.engine"))
            t0 = time.time()
            r = ModelRunner(cfg, key, cudart)
            t_create = time.time() - t0
            outs = [r.step(src[i])[0] for i in range(len(src.frames))]
            frame = src[0][None]
            r_trt = r.trt is not None
            place = node_placement(str(resolve(cfg, r.mcfg["onnx"])), dict(r_rt(cfg, r)), {r.in_name: frame}, r.trt)
            del r
            entry = {"session_create_s": round(t_create, 1), "engine_cache_hit": cache_hit, "placement": place}
            if cache_hit is False:
                entry["engine_build_s"] = entry["session_create_s"]   # includes the one-off engine build
            elif key in report and "engine_build_s" in report[key]:
                entry["engine_build_s"] = report[key]["engine_build_s"]
            if ref is None:
                ref = outs
            elif cfg["models"][key]["task"] == "detection":
                thr = cfg["models"][key]["score_threshold"]
                ag = [det_agreement(a[a[:, 4] >= thr], b[b[:, 4] >= thr]) for a, b in zip(ref, outs)]
                entry["vs_cuda_fp32"] = {"frames": len(ag), "boxes_ref": sum(a["n_a"] for a in ag),
                                         "boxes": sum(a["n_b"] for a in ag), "matched": sum(a["matched"] for a in ag),
                                         "unmatched": sum(a["unmatched"] for a in ag),
                                         "max_score_abs_diff": max((a.get("max_score_abs_diff", 0) for a in ag))}
            else:
                same = [float((a == b).mean()) for a, b in zip(ref, outs)]
                entry["vs_cuda_fp32"] = {"frames": len(same), "pixel_agreement_mean": float(np.mean(same)),
                                         "pixel_agreement_min": float(np.min(same))}
            n_trt = place["nodes_by_provider"].get("TensorrtExecutionProvider", 0)
            if r_trt and n_trt != 1:
                failures.append(f"{key}: {n_trt} TensorRT subgraphs (expected the whole graph as one)")
            if place.get("cpu_ops"):
                failures.append(f"{key}: nodes on CPU {place['cpu_ops']}")
            report[key] = entry
            print(key, json.dumps(entry), flush=True)
            out_path.write_text(json.dumps(report, indent=1))
    if failures:
        raise SystemExit("FAILED: " + "; ".join(failures))


def r_rt(cfg, runner):
    rt = dict(cfg["runtime"])
    if runner.trt is not None:
        rt["tensorrt"] = dict(cfg["tensorrt"], engine_cache_dir=str(resolve(cfg, cfg["tensorrt"]["engine_cache_dir"])))
        rt["graph_optimization"] = cfg["tensorrt"]["graph_optimization"]
    return rt


if __name__ == "__main__":
    main()
