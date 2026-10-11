"""Per-model runner with explicitly timed, device-synchronised stages.

One `step()` processes one frame through five stages, each bracketed by
`time.perf_counter_ns()` (QueryPerformanceCounter on Windows):

  pre   CPU: resize/letterbox/normalise (numpy/OpenCV)
  h2d   copy input into a pre-allocated device buffer, then cudaDeviceSynchronize
  gpu   run_with_iobinding (outputs stay on device), then cudaDeviceSynchronize
  d2h   copy outputs to host
  post  CPU: decode boxes / upsample + argmax

Synchronising between stages attributes time to the stage that caused it
rather than to whichever later call happens to block. The sync calls cost a
few microseconds each and are included in the measured time.

Device buffers are allocated once. Re-creating OrtValues per frame would add
an allocation to every measured iteration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from bsched.config import resolve
from bsched.models import (det_postprocess, det_preprocess, letterbox_params, make_session,
                           seg_postprocess, seg_preprocess)

STAGES = ("pre", "h2d", "gpu", "d2h", "post")


@dataclass
class StepTiming:
    t_start: int
    t_end: int
    stage_ns: dict = field(default_factory=dict)


class ModelRunner:
    def __init__(self, cfg: dict, role: str, cudart):
        self.role = role
        self.mcfg = cfg["models"][role]
        self.task = self.mcfg["task"]                 # "detection" | "segmentation"
        self.fused = bool(self.mcfg.get("fused"))     # raw uint8 frame in, GPU pre/post
        self.cudart = cudart
        rt = cfg["runtime"]
        self.device_id = rt.get("device_id", 0)
        self.trt = self.mcfg.get("trt")
        if self.trt is not None:
            rt = dict(rt, graph_optimization=cfg["tensorrt"]["graph_optimization"],
                      tensorrt=dict(cfg["tensorrt"], engine_cache_dir=str(resolve(cfg, cfg["tensorrt"]["engine_cache_dir"]))))
        self.sess = make_session(str(resolve(cfg, self.mcfg["onnx"])), rt, "cuda", trt=self.trt)
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.in_shape = [int(d) for d in inp.shape]
        self.out_names = [o.name for o in self.sess.get_outputs()]

        in_dtype = np.uint8 if self.fused else np.float32
        self.x_dev = ort.OrtValue.ortvalue_from_shape_and_type(self.in_shape, in_dtype, "cuda", self.device_id)
        self.io = self.sess.io_binding()
        self.io.bind_ortvalue_input(self.in_name, self.x_dev)
        for name in self.out_names:
            self.io.bind_output(name, "cuda", self.device_id)

    # ---- stage functions (kept separate so the scheduler can call them piecemeal)
    def preprocess(self, bgr: np.ndarray):
        hw = bgr.shape[:2]
        if self.fused:
            # The graph does resize/normalise; only a zero-copy batch view here.
            lb = letterbox_params(hw, self.mcfg["input_hw"])[0] if self.task == "detection" else None
            return bgr[None], (lb, hw)
        if self.task == "detection":
            x, lb = det_preprocess(bgr, self.mcfg["input_hw"])
            return x, (lb, hw)
        return seg_preprocess(bgr, self.mcfg["input_hw"]), (None, hw)

    def postprocess(self, out: np.ndarray, meta):
        lb, hw = meta
        if self.task == "detection":
            return det_postprocess(out, lb, self.mcfg["score_threshold"], hw)
        if self.fused:
            return out[0]                             # uint8 mask, already argmaxed on GPU
        return seg_postprocess(out, hw)

    def step(self, bgr: np.ndarray) -> tuple[object, StepTiming]:
        now = time.perf_counter_ns
        sync = self.cudart.synchronize
        t0 = now()
        x, meta = self.preprocess(bgr)
        t1 = now()
        self.x_dev.update_inplace(x)
        sync()
        t2 = now()
        self.sess.run_with_iobinding(self.io)
        sync()
        t3 = now()
        out = self.io.copy_outputs_to_cpu()[0]
        t4 = now()
        result = self.postprocess(out, meta)
        t5 = now()
        return result, StepTiming(t0, t5, dict(zip(STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4))))
