# Excluded Stage 3 runs

Kept unmodified for the record. They are not used in `summary.json` / `summary.md`.

| run | reason |
|---|---|
| `detector_fused+seg_fused_288x512__r2__fps45` | AC power was removed during the run (`system_before.on_ac_power = true`, `system_after.on_ac_power = false`, battery 80% → 78%). From 02:34:58 the telemetry shows P3 with `sw_power_cap` + `sync_boost` on 318 of 405 samples, the SM clock down to 210 MHz, and one `[Unknown Error]` sample at 02:35:49. Achieved 12.6 FPS against 37.6 FPS for the same setting in repeat 1. Re-run on AC power. |
