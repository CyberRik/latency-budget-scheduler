"""Stage 3: sequential contention. Both models on every frame, FPS sweep.

For each repeat, for each pipeline, the FPS list is run in a shuffled order
(seeded per repeat), so GPU heat-up and time-of-run drift aren't confounded
with FPS. Each (repeat, pipeline, fps) run is a fresh process.

Aggregation across repeats: medians of per-run percentiles with the
[min, max] across repeats, and miss rates pooled over all frames of all
repeats. The FPS at which p99 crosses the deadline is reported as a bracket
of *tested* FPS values (highest passing, lowest failing). It is not
interpolated.

Usage:  python scripts/run_stage3.py [--config config.toml] [--aggregate-only]
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


def run_name(pipe: list[str]) -> str:
    return "+".join(pipe)


def out_file(out_dir: Path, pipe, rep, fps) -> Path:
    return out_dir / f"{run_name(pipe)}__r{rep}__fps{float(fps):g}.json"


def worker(cfg, pipe, rep, fps, out_dir):
    from bsched.harness import SequentialPolicy, run_pipeline
    run_pipeline(cfg, pipe, SequentialPolicy(pipe), fps, out_file(out_dir, pipe, rep, fps), {"repeat": rep})


def aggregate(cfg, out_dir: Path) -> None:
    s3 = cfg["stage3"]
    D = cfg["harness"]["deadline_ms"]
    rows, crossings, excluded = [], {}, []
    for pipe in s3["pipelines"]:
        det, seg = pipe[0], pipe[-1]
        per_fps = {}
        for fps in s3["fps"]:
            runs = []
            for rep in range(1, s3["repeats"] + 1):
                if not (f := out_file(out_dir, pipe, rep, fps)).exists():
                    continue
                r = json.loads(f.read_text())
                # Valid only if on AC power for the whole run: on battery this
                # GPU drops to P3 under a software power cap.
                if not (r["system_before"].get("on_ac_power") and r["system_after"].get("on_ac_power")):
                    excluded.append({"run": f.name, "reason": "not on AC power for the whole run"})
                    continue
                runs.append(r)
            if not runs:
                continue
            fe = [r["latency_ms"]["frame_e2e"] for r in runs]
            de = [r["models"][det]["e2e"] for r in runs]
            se = [r["models"][seg]["e2e"] for r in runs]
            pooled = lambda dl: (sum(d["misses"] for d in dl), sum(d["n"] for d in dl))  # noqa: E731
            fm, fn = pooled([r["deadline"]["frame"] for r in runs])
            dm, dn = pooled([r["models"][det]["deadline"] for r in runs])
            med = lambda xs: (float(np.median(xs)), float(min(xs)), float(max(xs)))  # noqa: E731
            row = {
                "pipeline": run_name(pipe), "fps": fps or "closed", "repeats": len(runs),
                "achieved_fps": med([r["stream"]["achieved_fps"] for r in runs]),
                "frame_e2e_p50": med([x["p50"] for x in fe]), "frame_e2e_p99": med([x["p99"] for x in fe]),
                "frame_e2e_max": med([x["max"] for x in fe]),
                "frame_miss_rate": fm / fn, "frame_misses": f"{fm}/{fn}",
                "frame_miss_rate_per_repeat": [r["deadline"]["frame"]["rate"] for r in runs],
                "det_e2e_p99": med([x["p99"] for x in de]), "det_miss_rate": dm / dn, "det_misses": f"{dm}/{dn}",
                "seg_e2e_p99": med([x["p99"] for x in se]),
                "det_gpu_p50": med([r["models"][det]["stages"]["gpu"]["p50"] for r in runs]),
                "seg_gpu_p50": med([r["models"][seg]["stages"]["gpu"]["p50"] for r in runs]),
                "sm_clock_mean_mhz": med([r["telemetry"]["sm_clock_mhz"]["mean"] for r in runs]),
                "ram_avail_gib_min": min(r["system_before"]["ram"]["avail_gib"] for r in runs),
                "all_on_ac": all(r["system_before"].get("on_ac_power") for r in runs),
            }
            rows.append(row)
            if fps:
                per_fps[fps] = ([x["p99"] for x in fe], row["frame_e2e_p99"][0])
        tested = sorted(per_fps)
        passing = [f for f in tested if per_fps[f][1] <= D]
        failing = [f for f in tested if per_fps[f][1] > D]
        crossings[run_name(pipe)] = {
            "criterion": f"median-over-repeats frame e2e p99 <= {D} ms",
            "highest_tested_fps_passing": max(passing) if passing else None,
            "lowest_tested_fps_failing": min(failing) if failing else None,
            "monotonic": (max(passing) < min(failing)) if passing and failing else None,
            "repeats_agree_at_each_fps": {f: (all(p <= D for p in per_fps[f][0]) or all(p > D for p in per_fps[f][0]))
                                          for f in tested},
        }
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "p99_crossing": crossings, "excluded_runs": excluded,
                                                       "manually_excluded": "see invalid/README.md"}, indent=1))

    def f3(t):
        return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    hdr = ["pipeline", "fps", "achieved fps", "frame e2e p50", "frame e2e p99", "frame miss rate",
           "det e2e p99", "det miss rate", "seg e2e p99", "det/seg GPU p50", "SM MHz"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        md.append("| " + " | ".join([
            r["pipeline"].replace("detector_fused+", "det+"), str(r["fps"]), f'{r["achieved_fps"][0]:.1f}',
            f3(r["frame_e2e_p50"]), f3(r["frame_e2e_p99"]), f'{r["frame_miss_rate"]:.3f} ({r["frame_misses"]})',
            f3(r["det_e2e_p99"]), f'{r["det_miss_rate"]:.3f} ({r["det_misses"]})', f3(r["seg_e2e_p99"]),
            f'{r["det_gpu_p50"][0]:.2f} / {r["seg_gpu_p50"][0]:.2f}', f'{r["sm_clock_mean_mhz"][0]:.0f}']) + " |")
    text = ("Values: median over repeats [min, max over repeats]; ms. Miss rates pooled over all frames "
            "of all repeats. e2e measured from scheduled frame arrival.\n\n" + "\n".join(md) +
            "\n\np99 crossing (tested FPS only, not interpolated):\n\n```\n" + json.dumps(crossings, indent=1) + "\n```\n" +
            f"\nExcluded runs (automatic): {json.dumps(excluded) if excluded else 'none'}. "
            "Runs removed by hand are listed in invalid/README.md.\n")
    (out_dir / "summary.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--pipe", nargs="+")
    ap.add_argument("--rep", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s3 = cfg["stage3"]
    out_dir = resolve(cfg, s3["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.worker:
        worker(cfg, args.pipe, args.rep, args.fps, out_dir)
        return
    if not args.aggregate_only:
        plan = []
        for rep in range(1, s3["repeats"] + 1):
            order = list(np.random.default_rng(cfg["harness"]["seed"] + rep).permutation(s3["fps"]))
            for pipe in s3["pipelines"]:
                plan += [(pipe, rep, float(f)) for f in order]
        for k, (pipe, rep, fps) in enumerate(plan):
            if out_file(out_dir, pipe, rep, fps).exists():
                print(f"[{k + 1}/{len(plan)}] skip (exists) {run_name(pipe)} r{rep} fps={fps:g}", flush=True)
                continue
            if k:
                time.sleep(cfg["harness"]["cooldown_s"])
            print(f"[{k + 1}/{len(plan)}] {run_name(pipe)} r{rep} fps={fps:g}", flush=True)
            cmd = [sys.executable, __file__, "--worker", "--pipe", *pipe, "--rep", str(rep), "--fps", str(fps)]
            if args.config:
                cmd += ["--config", args.config]
            subprocess.run(cmd, check=True)
    aggregate(cfg, out_dir)


if __name__ == "__main__":
    main()
