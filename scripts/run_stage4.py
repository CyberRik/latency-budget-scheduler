"""Stage 4: deadline-aware scheduler, same FPS sweep as Stage 3, compared with it.

Each (repeat, fps) run is a fresh process; within a repeat the FPS order is
shuffled with the same seeds as Stage 3. Resumable: runs whose JSON exists
are skipped.

Comparison with Stage 3 at the same FPS uses the same per-frame definitions
for both:
  det miss        detector finished after arrival + deadline
  frame miss      some model that ran on the frame finished after the deadline
  seg on time     a segmentation mask (any rung) finished by the deadline
  seg late        segmentation ran but finished after the deadline
                  (for the scheduler: it predicted a fit and was wrong)
Rates are pooled over all frames of all repeats; percentiles are the median
over repeats [min, max]. Runs not on AC power for the whole run are excluded
and listed.

Usage:  python scripts/run_stage4.py [--config config.toml] [--aggregate-only]
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


def out_file(out_dir: Path, rep, fps) -> Path:
    return out_dir / f"deadline_aware__r{rep}__fps{float(fps):g}.json"


def stage3_file(cfg, pipe, rep, fps) -> Path:
    return resolve(cfg, cfg["stage3"]["out_dir"]) / f"{'+'.join(pipe)}__r{rep}__fps{float(fps):g}.json"


def worker(cfg, rep, fps, out_dir):
    from bsched.harness import run_pipeline
    from bsched.scheduler import DeadlineAwarePolicy
    s4 = cfg["stage4"]
    keys = [s4["detector"], *s4["ladder"]]
    policy = DeadlineAwarePolicy(s4["detector"], s4["ladder"], fps, cfg["scheduler"])
    run_pipeline(cfg, keys, policy, fps, out_file(out_dir, rep, fps), {"repeat": rep})


def on_ac(r) -> bool:
    return bool(r["system_before"].get("on_ac_power") and r["system_after"].get("on_ac_power"))


def frame_metrics(r: dict, det: str, segs: list[str], D: float) -> dict:
    """Per-run counts from the per-frame arrays (works for Stage 3 and 4 runs)."""
    pf = r["per_frame_ms"]
    nan = lambda xs: np.array([np.nan if v is None else v for v in xs], float)  # noqa: E731
    det_e2e = nan(pf[f"{det}_e2e"])
    seg_e2e = {k: nan(pf[f"{k}_e2e"]) for k in segs if f"{k}_e2e" in pf}
    n = det_e2e.size
    seg_any = np.full(n, np.nan)
    ran_by = {}
    for k, v in seg_e2e.items():
        m = ~np.isnan(v)
        seg_any[m] = v[m]
        ran_by[k] = int(m.sum())
    ran = ~np.isnan(seg_any)
    return {
        "n": n,
        "det_miss": int((det_e2e > D).sum()),
        "frame_miss": int((np.array(pf["frame_e2e"]) > D).sum()),
        "seg_ran_by_rung": ran_by, "seg_skipped": int((~ran).sum()),
        "seg_on_time": int((seg_any[ran] <= D).sum()),
        "seg_on_time_by_rung": {k: int((v[~np.isnan(v)] <= D).sum()) for k, v in seg_e2e.items()},
        "seg_late": int((seg_any[ran] > D).sum()),
    }


def summarize_runs(runs, det, segs, D) -> dict:
    ms = [frame_metrics(r, det, segs, D) for r in runs]
    N = sum(m["n"] for m in ms)
    tot = lambda k: sum(m[k] for m in ms)  # noqa: E731
    med = lambda xs: (float(np.median(xs)), float(min(xs)), float(max(xs)))  # noqa: E731
    rung = {k: sum(m["seg_ran_by_rung"].get(k, 0) for m in ms) for k in segs}
    rung_ok = {k: sum(m["seg_on_time_by_rung"].get(k, 0) for m in ms) for k in segs}
    return {
        "repeats": len(runs), "frames": N,
        "achieved_fps": med([r["stream"]["achieved_fps"] for r in runs]),
        "frame_e2e_p50": med([r["latency_ms"]["frame_e2e"]["p50"] for r in runs]),
        "frame_e2e_p99": med([r["latency_ms"]["frame_e2e"]["p99"] for r in runs]),
        "det_e2e_p99": med([r["models"][det]["e2e"]["p99"] for r in runs]),
        "det_miss_rate": tot("det_miss") / N, "det_misses": tot("det_miss"),
        "frame_miss_rate": tot("frame_miss") / N, "frame_misses": tot("frame_miss"),
        "seg_on_time_rate": tot("seg_on_time") / N, "seg_on_time": tot("seg_on_time"),
        "seg_late": tot("seg_late"), "seg_skipped": tot("seg_skipped"),
        "seg_ran_by_rung": rung, "seg_on_time_by_rung": rung_ok,
        "sm_clock_mean_mhz": med([r["telemetry"]["sm_clock_mhz"]["mean"] for r in runs]),
    }


def aggregate(cfg, out_dir: Path) -> None:
    s4, s3 = cfg["stage4"], cfg["stage3"]
    D = cfg["harness"]["deadline_ms"]
    det, ladder = s4["detector"], s4["ladder"]
    excluded, rows = [], []

    def load(paths):
        out = []
        for f in paths:
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            if not on_ac(r):
                excluded.append({"run": f.name, "reason": "not on AC power for the whole run"})
                continue
            out.append(r)
        return out

    for fps in s4["fps"]:
        row = {"fps": fps or "closed"}
        runs = load([out_file(out_dir, rep, fps) for rep in range(1, s4["repeats"] + 1)])
        if runs:
            row["scheduler"] = summarize_runs(runs, det, ladder, D)
            dec = [d for r in runs for d in r["per_frame_decisions"]]
            row["scheduler"]["budget_binding_counts"] = {
                b: sum(d["binding"] == b for d in dec) for b in ("own_deadline", "next_frame_detection")}
            row["scheduler"]["det_started_late"] = sum(d["det_late_at_start"] for d in dec)
        for pipe in s3["pipelines"]:
            if fps not in s3["fps"]:
                continue
            r3 = load([stage3_file(cfg, pipe, rep, fps) for rep in range(1, s3["repeats"] + 1)])
            if r3:
                row["sequential " + pipe[-1]] = summarize_runs(r3, det, [pipe[-1]], D)
        rows.append(row)
    (out_dir / "summary.json").write_text(json.dumps({"deadline_ms": D, "rows": rows, "excluded_runs": excluded},
                                                     indent=1))

    def pct(x):
        return f"{100 * x:.1f}%"

    def f3(t):
        return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    short = {"scheduler": "scheduler", **{"sequential " + p[-1]: "seq " + p[-1].split("_")[-1] for p in s3["pipelines"]}}
    hdr = ["fps", "policy", "achieved fps", "frame e2e p50", "frame e2e p99", "frame miss", "det e2e p99",
           "det miss", "seg on time", "seg late", "seg 384 / 288 / skip", "SM MHz"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for row in rows:
        for key, label in short.items():
            if key not in row:
                continue
            m = row[key]
            rung = m["seg_ran_by_rung"]
            md.append("| " + " | ".join([
                str(row["fps"]), label, f'{m["achieved_fps"][0]:.1f}', f3(m["frame_e2e_p50"]), f3(m["frame_e2e_p99"]),
                f'{pct(m["frame_miss_rate"])} ({m["frame_misses"]})', f3(m["det_e2e_p99"]),
                f'{pct(m["det_miss_rate"])} ({m["det_misses"]})',
                f'{pct(m["seg_on_time_rate"])} ({m["seg_on_time"]})', str(m["seg_late"]),
                f'{rung.get(ladder[0], 0)} / {rung.get(ladder[1], 0)} / {m["seg_skipped"]}',
                f'{m["sm_clock_mean_mhz"][0]:.0f}']) + " |")
    text = ("Deadline 33 ms. Percentiles: median over repeats [min, max]; ms. Counts and rates pooled over all "
            "frames of all repeats (3000 per cell). 'seq' rows are the Stage 3 runs at the same FPS.\n\n"
            + "\n".join(md) + f"\n\nExcluded runs (automatic): {json.dumps(excluded) if excluded else 'none'}. "
            "Runs removed by hand are listed in invalid/README.md of each stage.\n")
    (out_dir / "summary.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--rep", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s4 = cfg["stage4"]
    out_dir = resolve(cfg, s4["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.worker:
        worker(cfg, args.rep, args.fps, out_dir)
        return
    if not args.aggregate_only:
        plan = []
        for rep in range(1, s4["repeats"] + 1):
            order = list(np.random.default_rng(cfg["harness"]["seed"] + rep).permutation(s4["fps"]))
            plan += [(rep, float(f)) for f in order]
        ran_any = False
        for k, (rep, fps) in enumerate(plan):
            if out_file(out_dir, rep, fps).exists():
                print(f"[{k + 1}/{len(plan)}] skip (exists) r{rep} fps={fps:g}", flush=True)
                continue
            if ran_any:
                time.sleep(cfg["harness"]["cooldown_s"])
            ran_any = True
            print(f"[{k + 1}/{len(plan)}] deadline_aware r{rep} fps={fps:g}", flush=True)
            cmd = [sys.executable, __file__, "--worker", "--rep", str(rep), "--fps", str(fps)]
            if args.config:
                cmd += ["--config", args.config]
            subprocess.run(cmd, check=True)
    aggregate(cfg, out_dir)


if __name__ == "__main__":
    main()
