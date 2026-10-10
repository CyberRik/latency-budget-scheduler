"""Multi-model per-frame pipeline harness (Stages 3 and 4).

A *policy* decides, per frame, which models run. Each frame is processed
synchronously in arrival order on one thread. The harness supplies the
frame's arrival time and the deadline, so a policy can reason about the
remaining budget.

  Stage 3: SequentialPolicy: every model, every frame, in order.
  Stage 4: deadline-aware policy (bsched/scheduler.py).

Measurement is the Stage 2 harness unchanged: open-loop arrivals,
sync-bracketed stages, fresh process per run, telemetry in a separate
process, GC paused during the timed loop.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from bsched.config import relpath, resolve
from bsched.engine import STAGES, ModelRunner
from bsched.gpu import Cudart, Telemetry, nvsmi_used_mib, system_snapshot
from bsched.stats import miss_rate, summarize
from bsched.stream import FrameSource, run_stream

MIB = 1 << 20


class SequentialPolicy:
    """Stage 3 baseline: run every model on every frame, in the given order."""
    name = "sequential"

    def __init__(self, keys: list[str]):
        self.keys = keys

    def run_frame(self, i: int, frame, t_arrival: int, deadline_ns: int, runners: dict) -> list[dict]:
        out = []
        for k in self.keys:
            _, t = runners[k].step(frame)
            out.append({"model": k, "t_start": t.t_start, "t_end": t.t_end, **t.stage_ns})
        return out

    def describe(self) -> dict:
        return {"policy": self.name, "order": self.keys}


def run_pipeline(cfg: dict, keys: list[str], policy, fps: float, out_path: Path, extra: dict | None = None) -> dict:
    import onnxruntime as ort

    h = cfg["harness"]
    deadline_ns = int(h["deadline_ms"] * 1e6)
    sys_before = system_snapshot()
    mem = {"device_used_before_context_mib (nvidia-smi)": nvsmi_used_mib()}

    source = FrameSource(str(resolve(cfg, cfg["data"]["frames_glob"])), h["preload_frames"], h["seed"],
                         tuple(cfg["data"]["frame_hw"]))
    cudart = Cudart(cfg["runtime"].get("device_id", 0))
    ctx_used, total = cudart.mem_info()
    mem["device_total_mib"] = total / MIB
    mem["device_used_after_context_mib"] = ctx_used / MIB
    runners = {k: ModelRunner(cfg, k, cudart) for k in keys}
    mem["device_used_after_sessions_mib"] = cudart.mem_info()[0] / MIB

    # Warmup: the whole pipeline, closed loop, discarded.
    warm_ms = []
    for i in range(h["warmup_iters"]):
        t0 = time.perf_counter_ns()
        for k in keys:
            runners[k].step(source[i])
        warm_ms.append((time.perf_counter_ns() - t0) / 1e6)
    if hasattr(policy, "warmup"):
        policy.warmup(runners, source)   # e.g. seed latency estimates; also discarded
    mem["device_used_after_warmup_mib"] = cudart.mem_info()[0] / MIB
    peak = [cudart.mem_info()[0]]

    def process(i, frame, t_arr):
        execs = policy.run_frame(i, frame, t_arr, deadline_ns, runners)
        u = cudart.mem_info()[0]
        if u > peak[0]:
            peak[0] = u
        return {"execs": execs, "decisions": getattr(policy, "last_decision", None)}

    time.sleep(0.5)
    with Telemetry(str(out_path.with_suffix(".telemetry.csv")), h["telemetry_interval_ms"]) as tel:
        records, meta = run_stream(source, h["measure_frames"], fps, process, h["disable_gc"])
    sys_after = system_snapshot()
    mem["device_used_peak_during_run_mib"] = peak[0] / MIB
    mem["pipeline_delta_peak_minus_context_mib"] = (peak[0] - ctx_used) / MIB

    # ---- per-frame derived quantities (ms); NaN where a model didn't run
    n = len(records)
    arr = np.array([r["t_arrival"] for r in records], np.int64)
    start = np.array([r["t_start"] for r in records], np.int64)
    end = np.array([max(e["t_end"] for e in r["execs"]) if r["execs"] else r["t_start"] for r in records], np.int64)
    per_model = {}
    for k in keys:
        done = np.full(n, np.nan)
        stage = {s: np.full(n, np.nan) for s in STAGES}
        for j, r in enumerate(records):
            for e in r["execs"]:
                if e["model"] == k:
                    done[j] = (e["t_end"] - arr[j]) / 1e6
                    for s in STAGES:
                        stage[s][j] = e[s] / 1e6
        per_model[k] = {"e2e": done, "stages": stage}

    frame_e2e = (end - arr) / 1e6
    queue = (start - arr) / 1e6
    service = (end - start) / 1e6
    sm = lambda a: summarize(a[~np.isnan(a)], h["bootstrap_resamples"], h["seed"])  # noqa: E731

    model_results = {}
    for k, d in per_model.items():
        ran = ~np.isnan(d["e2e"])
        model_results[k] = {
            "task": cfg["models"][k]["task"], "name": cfg["models"][k]["name"],
            "ran_frames": int(ran.sum()), "completion_rate": float(ran.mean()),
            "e2e": sm(d["e2e"]), "stages": {s: sm(v) for s, v in d["stages"].items()},
            # A frame where the model didn't run counts as a miss for it.
            "deadline": {**miss_rate(np.where(ran, d["e2e"], np.inf), h["deadline_ms"]),
                         "on_time_completions": int((d["e2e"] <= h["deadline_ms"]).sum())},
        }

    result = {
        "pipeline": keys, **policy.describe(), "target_fps": fps,
        "mode": "closed_loop" if fps == 0 else "open_loop", **(extra or {}),
        "config": {"harness": h, "runtime": cfg["runtime"]},
        "env": {"python": platform.python_version(), "onnxruntime": ort.__version__,
                "frames": [relpath(cfg, p) for p in source.paths],
                "excluded_frames": [relpath(cfg, p) for p in source.excluded]},
        "system_before": sys_before, "system_after": sys_after, "stream": meta,
        "warmup": {"n_discarded": len(warm_ms), "cold_first_iter_ms": warm_ms[0]},
        "latency_ms": {"frame_e2e": sm(frame_e2e), "queue": sm(queue), "service": sm(service)},
        # Frame deadline: every model that ran finished within the deadline.
        "deadline": {"frame": miss_rate(frame_e2e, h["deadline_ms"])},
        "models": model_results,
        "memory": mem, "telemetry": tel.summary(),
        "per_frame_ms": {"t_arrival_rel": ((arr - arr[0]) / 1e6).round(4).tolist(),
                         "frame_e2e": frame_e2e.round(4).tolist(), "queue": queue.round(4).tolist(),
                         **{f"{k}_e2e": [None if np.isnan(v) else round(float(v), 4) for v in d["e2e"]]
                            for k, d in per_model.items()},
                         **{f"{k}_gpu": [None if np.isnan(v) else round(float(v), 4) for v in d["stages"]["gpu"]]
                            for k, d in per_model.items()}},
    }
    if any(r["decisions"] is not None for r in records):
        result["per_frame_decisions"] = [r["decisions"] for r in records]
    out_path.write_text(json.dumps(result, indent=1))
    return result
