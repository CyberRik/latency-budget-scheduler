"""Stage 4: deadline-aware scheduler policy (plugs into bsched.harness).

Per frame:
  1. Detection always runs, first, even if the frame is already late
     (requirement: detection must always complete).
  2. Segmentation is chosen after detection finishes, from the remaining
     budget: the first rung of the ladder (best quality first) whose latency
     estimate fits; if none fits, segmentation is skipped for this frame.

The budget for segmentation ends at the earlier of:
  - this frame's deadline:            t_arrival + D
  - the latest time the next frame's detection can start and still meet its
    own deadline:                     t_arrival + P + D - est(detector)
    (P = frame period. This only binds when est(detector) > P, i.e. near
    detector-only capacity, or when the GPU is slowed down. It is what keeps
    segmentation from starving the next frame's detection.)
In closed-loop mode (fps = 0) the next frame arrives when this one is done,
so only this frame's deadline applies.

Latency estimate per model: EWMA of the step wall time (pre + h2d + gpu + d2h
+ post, as timed by ModelRunner.step) and of its absolute deviation, with
est = mean + K * dev. This is the Jacobson/Karels estimator used for TCP
retransmission timers (RFC 6298). alpha, beta and K are fixed in config.toml
before any Stage 4 run, at the RFC values, and not tuned on the evaluation
sweep. Estimates update only when a model runs, so a rung that is skipped
keeps its estimate. Without a correction this starves segmentation: in a
200-frame smoke test (before any evaluation run), one slow 288x512 frame
pushed its estimate above the budget and segmentation never ran again.
Correction (probe): a rung that has not run for `probe_after_frames` frames
is run once if its EWMA *mean* (no deviation margin) fits the budget, which
gives the estimator a fresh sample. Probes are logged.
"""
from __future__ import annotations

import time


class Ewma:
    def __init__(self, alpha: float, beta: float, k: float):
        self.alpha, self.beta, self.k = alpha, beta, k
        self.mean = self.dev = None
        self.n = 0

    def update(self, x_ns: int) -> None:
        if self.mean is None:
            self.mean, self.dev = float(x_ns), x_ns / 2.0
        else:
            self.dev = (1 - self.beta) * self.dev + self.beta * abs(x_ns - self.mean)
            self.mean = (1 - self.alpha) * self.mean + self.alpha * x_ns
        self.n += 1

    def estimate(self) -> float:
        return self.mean + self.k * self.dev


class DeadlineAwarePolicy:
    name = "deadline_aware"

    def __init__(self, detector: str, ladder: list[str], fps: float, sc: dict):
        self.detector, self.ladder = detector, list(ladder)
        self.period_ns = int(round(1e9 / fps)) if fps > 0 else 0
        self.sc = sc
        self.est = {k: Ewma(sc["ewma_alpha"], sc["ewma_beta"], sc["k_dev"]) for k in [detector, *ladder]}
        self.since_run = {k: 0 for k in ladder}   # frames since each rung last ran
        self.last_decision = None

    def warmup(self, runners: dict, source) -> None:
        """Seed the estimators by running each model closed loop (discarded)."""
        for k, e in self.est.items():
            for i in range(self.sc["seed_iters"]):
                _, t = runners[k].step(source[i])
                e.update(t.t_end - t.t_start)

    def run_frame(self, i: int, frame, t_arrival: int, deadline_ns: int, runners: dict) -> list[dict]:
        execs = []
        det_est_before = self.est[self.detector].estimate()
        _, t = runners[self.detector].step(frame)
        self.est[self.detector].update(t.t_end - t.t_start)
        execs.append({"model": self.detector, "t_start": t.t_start, "t_end": t.t_end, **t.stage_ns})

        now = time.perf_counter_ns()
        end_own = t_arrival + deadline_ns
        if self.period_ns:
            end_next = t_arrival + self.period_ns + deadline_ns - self.est[self.detector].estimate()
            budget_end = min(end_own, end_next)
            binding = "own_deadline" if end_own <= end_next else "next_frame_detection"
        else:
            budget_end, binding = end_own, "own_deadline"
        budget = budget_end - now

        ests = {k: self.est[k].estimate() for k in self.ladder}
        # Best-quality rung that fits; a rung idle for probe_after_frames may
        # instead qualify on its mean alone (otherwise a stale, inflated
        # estimate would lock it out for good, also when a lower rung fits).
        choice, probe = None, False
        for k in self.ladder:
            if ests[k] <= budget:
                choice = k
                break
            if self.since_run[k] >= self.sc["probe_after_frames"] and self.est[k].mean <= budget:
                choice, probe = k, True
                break
        for k in self.ladder:
            self.since_run[k] = 0 if k == choice else self.since_run[k] + 1
        if choice is not None:
            _, t = runners[choice].step(frame)
            self.est[choice].update(t.t_end - t.t_start)
            execs.append({"model": choice, "t_start": t.t_start, "t_end": t.t_end, **t.stage_ns})

        ms = lambda v: round(v / 1e6, 3)  # noqa: E731
        self.last_decision = {
            "det_late_at_start": bool(execs[0]["t_start"] > end_own - det_est_before),
            "seg_budget_ms": ms(budget), "binding": binding,
            "est_ms": {k: ms(v) for k, v in ests.items()}, "seg": choice, "probe": probe,
        }
        return execs

    def describe(self) -> dict:
        return {"policy": self.name, "detector": self.detector, "ladder": self.ladder,
                "period_ms": self.period_ns / 1e6 if self.period_ns else None, "scheduler_config": self.sc}
