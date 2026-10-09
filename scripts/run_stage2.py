"""Stage 2: single-model latency baselines.

For every (model, fps) in [stage2] the orchestrator launches a fresh child
process (`--worker`), so each run's device memory reflects only that model and
no allocator/cuDNN state carries between runs. Each worker writes
<out_dir>/<model>_fps<fps>.json; the orchestrator then writes summary.json
and summary.md.

Usage:  python scripts/run_stage2.py [--config config.toml]
        python scripts/run_stage2.py --models detector_fused seg_fused_288x512 --fps 0 30             --out-dir results/stage2b                                       (override plan)
        python scripts/run_stage2.py --worker --model detector --fps 30   (one run)
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from bsched.config import load_config, relpath, resolve
from bsched.stats import miss_rate, summarize

MIB = 1 << 20


def worker(cfg: dict, role: str, fps: float, out_path: Path) -> None:
    import onnxruntime as ort

    from bsched.engine import STAGES, ModelRunner
    from bsched.gpu import Cudart, Telemetry, nvsmi_used_mib, system_snapshot
    from bsched.stream import FrameSource, run_stream

    h = cfg["harness"]
    sys_before = system_snapshot()
    mem = {"device_used_before_context_mib (nvidia-smi)": nvsmi_used_mib()}

    source = FrameSource(str(resolve(cfg, cfg["data"]["frames_glob"])), h["preload_frames"], h["seed"],
                         tuple(cfg["data"]["frame_hw"]))

    cudart = Cudart(cfg["runtime"].get("device_id", 0))
    used, total = cudart.mem_info()
    mem["device_total_mib"] = total / MIB
    mem["device_used_after_context_mib"] = used / MIB
    ctx_used = used

    runner = ModelRunner(cfg, role, cudart)
    mem["device_used_after_session_mib"] = cudart.mem_info()[0] / MIB

    # Warmup (closed loop, discarded). First iteration includes cuDNN
    # EXHAUSTIVE algorithm search and lazy allocations: reported as cold start.
    warm_ms = []
    for i in range(h["warmup_iters"]):
        _, t = runner.step(source[i])
        warm_ms.append((t.t_end - t.t_start) / 1e6)
    mem["device_used_after_warmup_mib"] = cudart.mem_info()[0] / MIB

    peak = [cudart.mem_info()[0]]

    def process(i, frame, t_arr):
        _, t = runner.step(frame)
        rec = {"t_end": t.t_end, **{k: v for k, v in t.stage_ns.items()}}
        u = cudart.mem_info()[0]       # outside the timed stages; arena is persistent
        if u > peak[0]:
            peak[0] = u
        return rec

    time.sleep(0.5)
    with Telemetry(str(out_path.with_suffix(".telemetry.csv")), h["telemetry_interval_ms"]) as tel:
        records, meta = run_stream(source, h["measure_frames"], fps, process, h["disable_gc"])
    sys_after = system_snapshot()

    mem["device_used_peak_during_run_mib"] = peak[0] / MIB
    mem["model_delta_peak_minus_context_mib"] = (peak[0] - ctx_used) / MIB
    mem["note"] = ("Device-wide cudaMemGetInfo; includes other processes. Per-process usage is not "
                   "available under WDDM. 'model_delta' = peak during run minus usage right after "
                   "CUDA context creation (sessions not yet created).")

    def col(f):
        return np.array([f(r) for r in records], dtype=np.float64) / 1e6

    e2e = col(lambda r: r["t_end"] - r["t_arrival"])
    queue = col(lambda r: r["t_start"] - r["t_arrival"])
    service = col(lambda r: r["t_end"] - r["t_start"])
    stages = {s: col(lambda r, s=s: r[s]) for s in STAGES}
    t_rel = col(lambda r: r["t_arrival"] - records[0]["t_arrival"])

    sm = lambda a: summarize(a, h["bootstrap_resamples"], h["seed"])  # noqa: E731
    result = {
        "stage": 2, "model": role, "model_name": cfg["models"][role]["name"],
        "onnx": cfg["models"][role]["onnx"], "input_shape": runner.in_shape,
        "target_fps": fps, "mode": "closed_loop" if fps == 0 else "open_loop",
        "config": {"harness": h, "runtime": cfg["runtime"]},
        "provider_options": runner.sess.get_provider_options().get("CUDAExecutionProvider"),
        "env": {"python": platform.python_version(), "onnxruntime": ort.__version__,
                "cudart": Path(cudart.path).name, "frames": [relpath(cfg, p) for p in source.paths],
                "excluded_frames": [relpath(cfg, p) for p in source.excluded]},
        "system_before": sys_before, "system_after": sys_after,
        "stream": meta,
        "warmup": {"n_discarded": len(warm_ms), "cold_first_iter_ms": warm_ms[0],
                   "service_ms_last10": warm_ms[-10:]},
        "latency_ms": {"e2e": sm(e2e), "queue": sm(queue), "service": sm(service),
                       "stages": {s: sm(v) for s, v in stages.items()}},
        "deadline": miss_rate(e2e, h["deadline_ms"]),
        "memory": mem,
        "telemetry": tel.summary(),
        "per_frame_ms": {"t_arrival_rel": t_rel.round(4).tolist(), "e2e": e2e.round(4).tolist(),
                         "queue": queue.round(4).tolist(), "service": service.round(4).tolist(),
                         **{s: v.round(4).tolist() for s, v in stages.items()}},
    }
    out_path.write_text(json.dumps(result, indent=1))


# ------------------------------------------------------------------ reporting

def fmt_ci(s: dict, p: int) -> str:
    v, ci = s.get(f"p{p}"), s.get(f"p{p}_ci95")
    return f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v:.2f}"


def write_summary(out_dir: Path, runs: list[dict]) -> None:
    rows = []
    for r in runs:
        L = r["latency_ms"]
        t = r["telemetry"]
        rows.append({
            "model": r["model_name"], "fps": r["target_fps"] or "closed", "n": L["e2e"]["n"],
            "achieved_fps": round(r["stream"]["achieved_fps"], 2),
            "e2e_p50": fmt_ci(L["e2e"], 50), "e2e_p95": fmt_ci(L["e2e"], 95), "e2e_p99": fmt_ci(L["e2e"], 99),
            "e2e_max": round(L["e2e"]["max"], 2), "e2e_mean": round(L["e2e"]["mean"], 2),
            "gpu_p50": round(L["stages"]["gpu"]["p50"], 2), "gpu_p99": round(L["stages"]["gpu"]["p99"], 2),
            "pre_p50": round(L["stages"]["pre"]["p50"], 2), "post_p50": round(L["stages"]["post"]["p50"], 2),
            "h2d_p50": round(L["stages"]["h2d"]["p50"], 2), "d2h_p50": round(L["stages"]["d2h"]["p50"], 2),
            "miss_rate": f'{r["deadline"]["misses"]}/{r["deadline"]["n"]}',
            "cold_first_ms": round(r["warmup"]["cold_first_iter_ms"], 1),
            "vram_delta_mib": round(r["memory"]["model_delta_peak_minus_context_mib"], 1),
            "sm_mhz_mean": round(t["sm_clock_mhz"]["mean"]) if t.get("sm_clock_mhz") else None,
            "temp_max_c": t["temp_c"]["max"] if t.get("temp_c") else None,
            "on_ac": r["system_before"].get("on_ac_power"),
            "ram_avail_gib": r["system_before"].get("ram", {}).get("avail_gib"),
        })
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1))
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (out_dir / "summary.md").write_text(
        "Latencies in ms; percentiles shown as value [95% bootstrap CI]. e2e is measured from scheduled "
        "frame arrival.\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--models", nargs="+", help="override [stage2].models")
    ap.add_argument("--fps", type=float, nargs="+", help="override [stage2].fps (worker: one value)")
    ap.add_argument("--out-dir", help="override [stage2].out_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = resolve(cfg, args.out_dir or cfg["stage2"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.worker:
        fps = args.fps[0]
        worker(cfg, args.model, fps, out_dir / f"{args.model}_fps{fps:g}.json")
        return

    runs = []
    models = args.models or cfg["stage2"]["models"]
    fps_list = args.fps or cfg["stage2"]["fps"]
    plan = [(m, f) for m in models for f in fps_list]
    for k, (model, fps) in enumerate(plan):
        if k:
            time.sleep(cfg["harness"]["cooldown_s"])
        print(f"[{k + 1}/{len(plan)}] {model} fps={fps or 'closed'}", flush=True)
        cmd = [sys.executable, __file__, "--worker", "--model", model, "--fps", str(fps),
               "--out-dir", str(out_dir)]
        if args.config:
            cmd += ["--config", args.config]
        subprocess.run(cmd, check=True)
        runs.append(json.loads((out_dir / f"{model}_fps{float(fps):g}.json").read_text()))
    write_summary(out_dir, runs)


if __name__ == "__main__":
    main()
