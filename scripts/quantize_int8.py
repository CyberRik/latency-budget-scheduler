"""Stage 5: INT8 post-training quantization (QDQ, explicit) for TensorRT.

For each model in [stage5].models:
  1. Calibration frames: `calib_images` frames drawn (seeded) from the TRAIN
     split of the matching BDD100K archive: 100k/train for the detector and
     10k/train for the segmenters (their eval sets are 100k/val and 10k/val).
     Read straight from the zips; only 1280x720 frames are used. The chosen
     file names are written to the report.
  2. The FP32 core graph is quantized with onnxruntime.quantization
     (quantize_static, QDQ format, symmetric INT8 activations and weights,
     per-channel weights, op types from [stage5].quant_op_types; everything
     else stays float). TensorRT consumes the QDQ graph directly (explicit
     quantization), so no TensorRT calibration table is used.
  3. The fused graph (raw frame in, GPU pre/post) is rebuilt around the QDQ
     core with scripts/build_fused.py's builders, which are unchanged.

Settings are fixed in config.toml before any Stage 5 evaluation run.

Usage:  python scripts/quantize_int8.py [--config config.toml] [--models ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
import numpy as np
import onnx

from bsched.config import load_config, relpath, resolve
from bsched.models import _preload_cuda_dlls, det_preprocess, seg_preprocess

ARCHIVE_FOR_TASK = {"detection": ("bdd100k_images_100k.zip", r"(^|/)100k/train/[^/]+\.jpg$"),
                    "segmentation": ("bdd100k_images_10k.zip", r"(^|/)10k/train/[^/]+\.jpg$")}


def calib_frames(cfg, task: str, n: int, seed: int):
    zname, pat = ARCHIVE_FOR_TASK[task]
    zpath = resolve(cfg, "data/archives") / zname
    z = zipfile.ZipFile(zpath)
    names = sorted(x for x in z.namelist() if re.search(pat, x))
    order = np.random.default_rng(seed).permutation(len(names))
    frames, used, skipped = [], [], 0
    hw = tuple(cfg["data"]["frame_hw"])
    for k in order:
        img = cv2.imdecode(np.frombuffer(z.read(names[k]), np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != hw:
            skipped += 1
            continue
        frames.append(img)
        used.append(Path(names[k]).name)
        if len(frames) == n:
            break
    return frames, {"archive": zname, "split_pattern": pat, "n": len(frames), "skipped_not_1280x720": skipped,
                    "files": used}


class Reader:
    """onnxruntime.quantization CalibrationDataReader over preprocessed frames."""

    def __init__(self, frames, input_name, prep):
        self.frames, self.input_name, self.prep, self.i = frames, input_name, prep, 0

    def get_next(self):
        if self.i == len(self.frames):
            return None
        self.i += 1
        return {self.input_name: self.prep(self.frames[self.i - 1])}   # lazily: keeps RAM low


def main():
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    import build_fused

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s5 = cfg["stage5"]
    _preload_cuda_dlls()
    seed = cfg["harness"]["seed"]
    report_path = resolve(cfg, s5["out_dir"]) / "quantization.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    for base in args.models or s5["models"]:
        m = cfg["models"][f"{base}@trt_int8"]
        core_fp32 = resolve(cfg, m["source_onnx_fp32"])
        core_qdq = resolve(cfg, m["source_onnx"])
        fused_qdq = resolve(cfg, m["onnx"])
        frames, calib = calib_frames(cfg, m["task"], s5["calib_images"], seed)
        input_name = onnx.load(str(core_fp32), load_external_data=False).graph.input[0].name
        if m["task"] == "detection":
            prep = lambda f: det_preprocess(f, m["input_hw"])[0]  # noqa: E731
        else:
            prep = lambda f: seg_preprocess(f, m["input_hw"])  # noqa: E731
        t0 = time.time()
        quantize_static(
            str(core_fp32), str(core_qdq), Reader(frames, input_name, prep),
            quant_format=QuantFormat.QDQ, op_types_to_quantize=s5["quant_op_types"],
            per_channel=s5["per_channel"], activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            calibrate_method=getattr(CalibrationMethod, s5["calib_method"]),
            calibration_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            # Not "CalibMaxIntermediateOutputs": in ORT 1.24.4 it clears each
            # full batch without computing its range, silently dropping data.
            # MinMax keeps only per-tensor min/max scalars anyway.
            # QuantizeBias False: TensorRT rejects int32 bias DequantizeLinear
            # ("only activation datatypes allowed"), which put 82 DQ nodes
            # on CPU and split the detector into 83 engines. TensorRT fuses
            # a float bias into the INT8 conv itself.
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False})
        t_quant = time.time() - t0
        if m["task"] == "detection":
            fused = build_fused.build_detector(onnx.load(str(core_qdq)), cfg["data"]["frame_hw"], m["input_hw"])
        else:   # TensorRT-only graph: ArgMax post-processing (see build_fused.py)
            fused = build_fused.build_segmenter(onnx.load(str(core_qdq)), cfg["data"]["frame_hw"], m["input_hw"],
                                                argmax="onnx")
        onnx.checker.check_model(fused, full_check=True)
        onnx.save(fused, str(fused_qdq))
        q = onnx.load(str(core_qdq), load_external_data=False)
        ops = {}
        for nd in q.graph.node:
            ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
        report[base] = {
            "core_fp32": relpath(cfg, core_fp32), "core_qdq": relpath(cfg, core_qdq), "fused_qdq": relpath(cfg, fused_qdq),
            "settings": {k: s5[k] for k in ("calib_images", "calib_method", "quant_op_types", "per_channel")}
                        | {"activation": "int8 symmetric", "weight": "int8 symmetric per-channel", "bias": "float (not quantized)",
                           "format": "QDQ"},
            "calibration": calib, "quantize_wall_s": round(t_quant, 1),
            "qdq_node_counts": {k: ops.get(k, 0) for k in ("QuantizeLinear", "DequantizeLinear", *s5["quant_op_types"])},
        }
        print(f"{base}: {relpath(cfg, fused_qdq)}  calib n={calib['n']}  {t_quant:.0f}s  "
              f"Q/DQ={report[base]['qdq_node_counts']}", flush=True)
        report_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
