"""Stage 5: latency and memory of each precision variant, and the final table.

Latency: every (model, variant) runs alone through the Stage 3 harness
(bsched.harness.run_pipeline with a one-model SequentialPolicy), closed loop
and at 30 FPS, `repeats` times, each run a fresh process. TensorRT engines
come from the cache that scripts/verify_stage5.py built, so engine build
time is not in any latency number (it is reported separately). Within each
repeat the run order is shuffled (seeded).

Table: joins latency/memory with accuracy from results/stage5/det_map.json
(scripts/eval_det.py) and seg_miou.json (scripts/eval_seg.py), and engine
build time from verify.json. Missing pieces are shown as "not measured".

Usage:  python scripts/run_stage5.py [--config config.toml] [--aggregate-only]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from bsched.config import load_config, resolve


def out_file(out_dir: Path, key: str, rep: int, fps: float) -> Path:
    return out_dir / "latency" / f"{key.replace('@', '__')}__r{rep}__fps{float(fps):g}.json"


def worker(cfg, key, rep, fps, out_dir):
    from bsched.harness import SequentialPolicy, run_pipeline
    out = out_file(out_dir, key, rep, fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_pipeline(cfg, [key], SequentialPolicy([key]), fps, out, {"repeat": rep, "variant": cfg["models"][key]["variant"]})


def valid(r) -> bool:
    """On AC power for the whole run, with GPU telemetry recorded."""
    return bool(r["system_before"].get("on_ac_power") and r["system_after"].get("on_ac_power")
                and r["telemetry"].get("n_samples"))


def aggregate(cfg, out_dir: Path) -> None:
    s5 = cfg["stage5"]
    D = cfg["harness"]["deadline_ms"]
    load = lambda name: json.loads((out_dir / name).read_text()) if (out_dir / name).exists() else {}  # noqa: E731
    det_acc, seg_acc, verify = load("det_map.json"), load("seg_miou.json"), load("verify.json")
    med = lambda xs: (float(np.median(xs)), float(min(xs)), float(max(xs))) if xs else None  # noqa: E731
    rows, excluded = [], []
    for base in s5["models"]:
        ref_acc = None
        for v in s5["variants"]:
            key = f"{base}@{v}"
            task = cfg["models"][key]["task"]
            row = {"model": base, "variant": v, "task": task}
            for fps in s5["fps"]:
                runs = []
                for rep in range(1, s5["repeats"] + 1):
                    f = out_file(out_dir, key, rep, fps)
                    if not f.exists():
                        continue
                    r = json.loads(f.read_text())
                    if not valid(r):
                        excluded.append(f.name)
                        continue
                    runs.append(r)
                m = [r["models"][key] for r in runs]
                tag = "closed" if fps == 0 else f"fps{fps:g}"
                row[tag] = {
                    "repeats": len(runs),
                    "gpu_p50": med([x["stages"]["gpu"]["p50"] for x in m]),
                    "e2e_p50": med([x["e2e"]["p50"] for x in m]), "e2e_p99": med([x["e2e"]["p99"] for x in m]),
                    "miss_rate": (sum(x["deadline"]["misses"] for x in m) / sum(x["deadline"]["n"] for x in m)) if m else None,
                    "mem_peak_minus_context_mib": med([r["memory"]["pipeline_delta_peak_minus_context_mib"] for r in runs]),
                    "sm_clock_mean_mhz": med([r["telemetry"]["sm_clock_mhz"]["mean"] for r in runs]),
                    "power_mean_w": med([r["telemetry"]["power_w"]["mean"] for r in runs]),
                }
            if task == "detection" and key in det_acc:
                a = det_acc[key]
                row["acc"] = {"mAP": a["mAP"], "mAP50": a["mAP50"], "mAP75": a["mAP75"], "n_frames": a["n_frames"]}
            elif task == "segmentation" and key in seg_acc:
                a = seg_acc[key]
                row["acc"] = {"mIoU": a["miou"], "pixel_acc": a["pixel_acc"], "n_frames": a["n_frames"]}
            if v == s5["variants"][0]:
                ref_acc = row.get("acc")
            if ref_acc and row.get("acc"):
                k0 = "mAP" if task == "detection" else "mIoU"
                row["acc"][f"delta_{k0}_vs_{s5['variants'][0]}"] = row["acc"][k0] - ref_acc[k0]
            row["engine_build_s"] = verify.get(key, {}).get("engine_build_s") if v.startswith("trt_") else None
            rows.append(row)
    (out_dir / "summary.json").write_text(json.dumps({"deadline_ms": D, "rows": rows, "excluded_runs": excluded},
                                                     indent=1), encoding="utf-8")

    nm = "not measured"
    pair = lambda a, b, k: f"{a[k][0]:.0f} / {b[k][0]:.0f}" if a.get(k) and b.get(k) else nm  # noqa: E731
    f3 = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]" if t else nm  # noqa: E731
    hdr = ["model", "variant", "GPU p50 (closed)", "e2e p50 (closed)", "e2e p99 (closed)", "e2e p99 @30 FPS",
           "miss @30 FPS", "GPU p50 @30 FPS", "SM MHz closed / @30", "board W closed / @30", "device mem (MiB)", "accuracy", "Δ vs cuda_fp32", "engine build (s)"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        c, f = r.get("closed", {}), r.get("fps30", {})
        acc = r.get("acc")
        if acc is None:
            a_s, d_s = nm, nm
        elif r["task"] == "detection":
            a_s = f"mAP {acc['mAP']:.4f} / AP50 {acc['mAP50']:.4f}"
            d_s = f"{acc.get('delta_mAP_vs_cuda_fp32', 0):+.4f}"
        else:
            a_s = f"mIoU {acc['mIoU']:.4f}"
            d_s = f"{acc.get('delta_mIoU_vs_cuda_fp32', 0):+.4f}"
        md.append("| " + " | ".join([
            r["model"], r["variant"], f3(c.get("gpu_p50")), f3(c.get("e2e_p50")), f3(c.get("e2e_p99")),
            f3(f.get("e2e_p99")), f"{100 * f['miss_rate']:.1f}%" if f.get("miss_rate") is not None else nm,
            f3(f.get("gpu_p50")), pair(c, f, "sm_clock_mean_mhz"), pair(c, f, "power_mean_w"),
            f"{c['mem_peak_minus_context_mib'][0]:.0f}" if c.get("mem_peak_minus_context_mib") else nm,
            a_s, d_s, f"{r['engine_build_s']:.0f}" if r["engine_build_s"] is not None else "–"]) + " |")
    text = ("Latency: ms, median over repeats [min, max]; each model alone, fresh process per run. "
            "Device memory: peak during the closed-loop run minus the CUDA context (median over repeats). "
            "SM clock and board power: nvidia-smi mean over the run (median over repeats). "
            "Accuracy: detection mAP@[.5:.95] on BDD100K 100k-val (COCO classes mapped, see README); "
            "segmentation mIoU on BDD100K 10k-val. Engine build: one-off TensorRT build at first session creation.\n\n"
            + "\n".join(md) + f"\n\nExcluded runs (not on AC power, or no GPU telemetry): {excluded or 'none'}\n")
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    sys.stdout.buffer.write(text.encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--key")
    ap.add_argument("--rep", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s5 = cfg["stage5"]
    out_dir = resolve(cfg, s5["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(cfg, args.key, args.rep, args.fps, out_dir)
        return
    if not args.aggregate_only:
        keys = [f"{b}@{v}" for b in s5["models"] for v in s5["variants"]]
        plan = []
        for rep in range(1, s5["repeats"] + 1):
            jobs = [(k, float(f)) for k in keys for f in s5["fps"]]
            order = np.random.default_rng(cfg["harness"]["seed"] + rep).permutation(len(jobs))
            plan += [(jobs[i][0], rep, jobs[i][1]) for i in order]
        ran_any = False
        for n, (key, rep, fps) in enumerate(plan):
            if out_file(out_dir, key, rep, fps).exists():
                print(f"[{n + 1}/{len(plan)}] skip (exists) {key} r{rep} fps={fps:g}", flush=True)
                continue
            if ran_any:
                time.sleep(cfg["harness"]["cooldown_s"])
            ran_any = True
            print(f"[{n + 1}/{len(plan)}] {key} r{rep} fps={fps:g}", flush=True)
            cmd = [sys.executable, __file__, "--worker", "--key", key, "--rep", str(rep), "--fps", str(fps)]
            if args.config:
                cmd += ["--config", args.config]
            subprocess.run(cmd, check=True)
    aggregate(cfg, out_dir)


if __name__ == "__main__":
    main()
