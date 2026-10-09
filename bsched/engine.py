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
from bsched.models import (det_postprocess, det_preprocess, make_session, seg_postprocess,
                           seg_preprocess)

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
        self.cudart = cudart
        rt = cfg["runtime"]
        self.device_id = rt.get("device_id", 0)
        self.sess = make_session(str(resolve(cfg, self.mcfg["onnx"])), rt, "cuda")
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.in_shape = [int(d) for d in inp.shape]
        self.out_names = [o.name for o in self.sess.get_outputs()]

        self.x_dev = ort.OrtValue.ortvalue_from_shape_and_type(self.in_shape, np.float32, "cuda", self.device_id)
        self.io = self.sess.io_binding()
        self.io.bind_ortvalue_input(self.in_name, self.x_dev)
        for name in self.out_names:
            self.io.bind_output(name, "cuda", self.device_id)

    # ---- stage functions (kept separate so the scheduler can call them piecemeal)
    def preprocess(self, bgr: np.ndarray):
        if self.role == "detector":
            x, lb = det_preprocess(bgr, self.mcfg["input_hw"])
            return x, (lb, bgr.shape[:2])
        return seg_preprocess(bgr, self.mcfg["input_hw"]), (None, bgr.shape[:2])

    def postprocess(self, out: np.ndarray, meta):
        lb, hw = meta
        if self.role == "detector":
            return det_postprocess(out, lb, self.mcfg["score_threshold"], hw)
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
