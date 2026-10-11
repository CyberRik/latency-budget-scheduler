# Invalid Stage 5 latency sweep (not used anywhere)

`latency_cpu_contended/` is the first Stage 5 latency sweep (2026-10-12 00:19–01:11 IST).
During that whole window an unrelated CPU-bound job (the pycocotools mAP
cross-check, `scripts/crosscheck_coco.py`, single process at ~100% of one core,
64 min CPU time) was running on the same machine. The harness host thread
competes for CPU, so these latencies are not clean measurements. The sweep was
re-run with nothing else running; only that re-run is reported. These files are
kept for transparency, not for comparison.

`no_telemetry/detector_fused__cuda_fp32__r3__fps0.*` is a run from the clean
re-sweep whose nvidia-smi logger recorded zero samples (empty CSV,
`telemetry.n_samples = 0`), so its clocks and power state can't be checked like
every other run. It finished around 02:00 IST, while the working tree was being
stashed and reset in GitHub Desktop. It was re-run; the re-run is the one
reported.
