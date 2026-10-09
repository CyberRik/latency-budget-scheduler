"""Latency statistics. Percentiles use numpy's default (linear) method on the
raw samples; 95% CIs are percentile-bootstrap over frames.

Tail CIs are only as good as the sample count: with n = 1000, p99 rests on ~10
samples. The CI and n are reported next to every percentile so this is visible.
"""
from __future__ import annotations

import numpy as np

PCTS = (50, 95, 99)


def summarize(ms: np.ndarray, resamples: int = 1000, seed: int = 0) -> dict:
    ms = np.asarray(ms, dtype=np.float64)
    out = {"n": int(ms.size)}
    if ms.size == 0:
        return out
    out.update(mean=float(ms.mean()), std=float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
               min=float(ms.min()), max=float(ms.max()))
    rng = np.random.default_rng(seed)
    boot = None
    if resamples and ms.size >= 20:
        idx = rng.integers(0, ms.size, size=(resamples, ms.size))
        boot = np.percentile(ms[idx], PCTS, axis=1)       # [len(PCTS), resamples]
    for k, p in enumerate(PCTS):
        out[f"p{p}"] = float(np.percentile(ms, p))
        if boot is not None:
            lo, hi = np.percentile(boot[k], [2.5, 97.5])
            out[f"p{p}_ci95"] = [float(lo), float(hi)]
    return out


def miss_rate(e2e_ms: np.ndarray, deadline_ms: float) -> dict:
    e2e_ms = np.asarray(e2e_ms)
    misses = int((e2e_ms > deadline_ms).sum())
    return {"deadline_ms": deadline_ms, "misses": misses, "n": int(e2e_ms.size),
            "rate": misses / e2e_ms.size if e2e_ms.size else None}
