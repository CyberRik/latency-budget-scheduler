"""Frame source and open-loop frame-stream driver.

Arrival model (open loop): frame i *arrives* at t0 + i / fps whether or not
the previous frame has finished, like a camera. Every frame's latency is
measured from its scheduled arrival, so time spent waiting behind a slow
predecessor counts against it. Measuring from when processing *starts*
would hide that queueing ("coordinated omission"). Stage 2 never drops
frames: if service time exceeds the frame period, the backlog and latency
grow, and that is reported as-is. Dropping is a scheduler decision (Stage 4).

fps = 0 means closed loop: each frame arrives when the previous one finishes.
That measures pure service time with no queueing.

Frames are decoded into RAM before the run, so JPEG decode and disk I/O
(which a camera pipeline wouldn't have) are not in the measured path. A fixed
subset is preloaded and cycled to bound RAM use. Repeating frames has no
caching effect on GPU inference.
"""
from __future__ import annotations

import gc
import glob
import time
from typing import Callable

import cv2
import numpy as np


class FrameSource:
    def __init__(self, pattern: str, n_preload: int, seed: int = 0):
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No frames match {pattern}")
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(paths), size=min(n_preload, len(paths)), replace=False))
        self.paths = [paths[i] for i in pick]
        self.frames = [cv2.imread(p) for p in self.paths]
        self.nbytes = sum(f.nbytes for f in self.frames)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.frames[i % len(self.frames)]


def _wait_until(t_ns: int) -> None:
    """Sleep most of the way, then spin. time.sleep alone overshoots by up to ~1 ms."""
    while True:
        rem = t_ns - time.perf_counter_ns()
        if rem <= 0:
            return
        if rem > 2_000_000:
            time.sleep((rem - 1_500_000) / 1e9)


def run_stream(source: FrameSource, n_frames: int, fps: float,
               process: Callable[[int, np.ndarray, int], dict],
               disable_gc: bool = True) -> tuple[list[dict], dict]:
    """Drive `process(i, frame, t_arrival_ns) -> record` over n_frames.

    Returns per-frame records (each gets t_arrival/t_start) and run metadata.
    """
    period_ns = int(round(1e9 / fps)) if fps > 0 else 0
    records = []
    gc.collect()
    gc_was_enabled = gc.isenabled()
    if disable_gc:
        gc.disable()
    try:
        t0 = time.perf_counter_ns() + 50_000_000     # first arrival 50 ms from now
        for i in range(n_frames):
            if period_ns:
                t_arr = t0 + i * period_ns
                _wait_until(t_arr)
            else:
                t_arr = time.perf_counter_ns()
            t_start = time.perf_counter_ns()
            rec = process(i, source[i], t_arr)
            rec["t_arrival"], rec["t_start"] = t_arr, t_start
            records.append(rec)
        t_end = time.perf_counter_ns()
    finally:
        if disable_gc and gc_was_enabled:
            gc.enable()
    first = records[0]["t_arrival"] if records else 0
    meta = {"target_fps": fps, "period_ms": period_ns / 1e6 if period_ns else None,
            "n_frames": n_frames, "wall_s": (t_end - first) / 1e9,
            "achieved_fps": n_frames / ((t_end - first) / 1e9) if records else None,
            "gc_disabled": disable_gc}
    return records, meta
