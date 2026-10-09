# Latency-budget scheduler for a two-model perception stack

Object detection (YOLOv10n) and semantic segmentation (SegFormer-B0, Cityscapes)
run over a BDD100K frame stream on one GPU under a hard per-frame deadline
(33 ms). Detection must always complete; segmentation is degraded or dropped
when the budget is exhausted.

**Every number in `results/` was measured on the hardware recorded alongside it.
Nothing is interpolated or estimated. Anything that could not be measured is
listed as such.**

Status: Stages 1 (setup/correctness), 2 (measurement harness + single-model
baselines) and 2b (GPU-fused pipeline, segmentation resolution sweep) complete.
Stage 3 (sequential contention) next.

## Hardware and software this was measured on

Recorded automatically in each run's JSON (`env` block). Development machine:

| | |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop, 4 GB, 48 W cap, Ampere sm_86, WDDM driver 566.07 |
| CPU / RAM | Ryzen 7 5800HS / 16 GB |
| OS | Windows 11 |
| Runtime | onnxruntime-gpu 1.24.4, CUDA EP, CUDA 12 / cuDNN 9 from pip wheels |

Caveats specific to this platform:

- **WDDM** (the default Windows driver model) adds submission jitter that mostly
  shows up in tail latency. Treat absolute p99 values as specific to this setup.
  Comparisons between configurations on the same machine are what's meaningful.
- **Laptop GPU**: power- and thermal-limited, and clocks cannot be locked.
  Runs log clocks, temperature and power.
- **Per-process VRAM is not available under WDDM** (NVML reports N/A). Memory is
  reported as device-wide used memory minus a pre-load baseline, and labelled
  that way.

## Layout

```
config.toml              single config for all stages
bsched/
  config.py              TOML config, path helpers
  models.py              ORT sessions, DLL setup, pre/post-processing
  engine.py              per-model runner: IOBinding, per-stage sync-bracketed timing
  stream.py              frame source + open-loop frame-stream driver
  gpu.py                 cudart sync/mem-info, nvidia-smi telemetry, system snapshot
  stats.py               percentiles + bootstrap CIs, deadline miss rate
  vis.py                 overlays
scripts/export_models.py export both models to static-shape FP32 ONNX
scripts/prepare_data.py  sort BDD100K archives by content, extract val splits
scripts/verify_stage1.py stage-1 correctness check
scripts/run_stage2.py    stage-2 single-model baselines
models/                  ONNX files + export_report.json
data/                    archives/ (raw zips), bdd100k/ (extracted), manifest.json
results/<stage>/         JSON results + visualisations
```

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# one-off export (needs torch/ultralytics/transformers):
.venv\Scripts\python -m pip install -r requirements-export.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python scripts\export_models.py
```

CUDA and cuDNN are loaded from the `nvidia-*` pip wheels (see
`bsched/models.py::_preload_cuda_dlls`), so no system CUDA toolkit is required.

Resolved versions on the dev machine: onnxruntime-gpu 1.24.4, cuDNN 9.27.0.42,
cuBLAS 12.9.2.10, CUDA runtime 12.9.79, numpy 2.5.3, opencv-python-headless 5.0.0.93.

## Data

BDD100K must be downloaded manually (registration required). Put the zips in
`data/` and run `python scripts/prepare_data.py`. It identifies each archive by
its contents, not its filename, and records sha256 and counts in
`data/manifest.json`.

| Set | Use |
|---|---|
| 10k val images (1000) + sem-seg `*_train_id.png` | segmentation mIoU; default frame stream |
| 100k val images (10000) + v1 per-image JSON labels | detection mAP |

Note: BDD100K images are independent stills from different videos, not
consecutive frames. Latency measurement doesn't need temporal coherence, but the
scheduler cannot exploit frame-to-frame reuse.

## Models

| Role | Model | Input | Output | Classes |
|---|---|---|---|---|
| Detection | YOLOv10n (ultralytics 8.3.202) | 1x3x640x640, letterboxed | 1x300x6 (x1,y1,x2,y2,score,cls), NMS-free | COCO-80 |
| Segmentation | `nvidia/segformer-b0-finetuned-cityscapes-512-1024` | 1x3x576x1024 (keeps 16:9) | 1x19x144x256 logits | Cityscapes-19 = BDD100K sem-seg |

Why SegFormer rather than DeepLabV3-MobileNetV3 or BiSeNet: torchvision's
DeepLabV3-MobileNetV3 is trained on the 21 Pascal VOC classes, which have no
road, sidewalk or sign classes, so mIoU against BDD100K would be meaningless.
BiSeNet Cityscapes weights only come from unofficial third-party links.
SegFormer-B0 uses the same 19-class label space as BDD100K's sem-seg labels.

Export checks (in `models/export_report.json`): SegFormer ONNX vs PyTorch, max
abs logit diff 1.3e-5 and 100% argmax agreement on a fixed random input.

Licensing: YOLOv10 / ultralytics are AGPL-3.0.

## Stage 1: correctness

`scripts/verify_stage1.py` runs both models on the first N frames and records:

- **Node placement**: which execution provider actually ran each node, from
  ORT's profiler. `get_providers()` only lists registered providers, which
  doesn't prove the work ran on the GPU.
- **GPU vs CPU agreement**: each frame is also run on the CPU EP. The script
  reports detection count/box/score/class agreement and segmentation logit diff
  and argmax agreement.
- **Output sanity**: shapes, finiteness, detection class histogram,
  segmentation class pixel fractions.
- **Segmentation label check**: where a BDD100K mask exists, pixel accuracy
  against it. This only checks that the class-id mapping is right. It is not
  the Stage 5 evaluation.
- Overlays in `results/stage1/vis/`.

No latency is reported in Stage 1.

### Stage 1 results (20 BDD100K 10k-val frames, `results/stage1/summary.json`)

| Check | Detector (YOLOv10n) | Segmenter (SegFormer-B0) |
|---|---|---|
| Nodes executed on CUDA EP / CPU EP | 236 / 0 | 376 / 0 |
| Outputs finite | yes | yes |
| GPU vs CPU EP | 127/127 dets matched (same class, IoU ≥ 0.9986); max box diff 0.13 px; max score diff 0.0076 | max logit diff 0.033; argmax agreement ≥ 99.95% per frame |
| Pixel accuracy vs BDD100K masks (sanity only) | n/a | mean 0.825 (min 0.515, max 0.961) |

The small GPU/CPU differences are consistent with Ampere's TF32 convolutions in
cuDNN versus FP32 on the CPU. The low end of the segmentation pixel accuracy
comes from domain shift (Cityscapes-trained model on BDD100K scenes such as
snow; see `vis/00_*`), not from a label-mapping error. A mapping error would
put accuracy near chance.

**Two issues found and fixed during Stage 1** (kept here because they would
have silently corrupted every later number):

1. *Silent CPU fallback.* With the pip CUDA wheels, cuDNN 9.27 lazily loads
   `cudnn_engines_tensor_ir64_9.dll`. `onnxruntime.preload_dlls()` doesn't
   preload that DLL and its directory isn't on the DLL search path, so the
   first Conv failed and **ORT's Python wrapper silently rebuilt the session on
   CPU only**. The run still "succeeded" with plausible outputs. Fixes: every
   `nvidia/*/bin` directory is added to the DLL search path; sessions call
   `disable_fallback()`; and Stage 1 refuses to report if the profiler shows no
   nodes on the CUDA EP.
2. *Detection parity matching.* Pairing GPU and CPU boxes by score rank gave a
   spurious 710 px "difference" when two near-tied scores swapped order.
   YOLOv10 can also emit one box under two classes. Boxes are now paired by IoU
   within the same class.

## Latency methodology (Stage 2+)

### What "latency" means here

**End-to-end latency (e2e)** runs from a frame's **scheduled arrival** to the
moment its post-processed result exists on the host. This is the quantity the
33 ms deadline applies to.

```
 arrival ──queue──► start ──pre──► ──h2d──► ──gpu──► ──d2h──► ──post──► end
 (camera clock)     (dequeued)     CPU      copy+sync run+sync  copy     CPU
 |<────────────────────────────── e2e ──────────────────────────────────>|
                    |<───────────────────── service ────────────────────>|
```

- **Open-loop arrivals.** Frame *i* arrives at `t0 + i/fps` regardless of
  whether frame *i-1* has finished, like a camera. When the pipeline falls
  behind, the waiting time (`queue`) counts against the frame. Timing from
  dequeue instead would hide exactly the overload behaviour this project is
  about (the "coordinated omission" problem in load testing). Stage 2 never
  drops frames; overload shows up as growing queue time.
- **Closed loop (`fps = 0`)** gives pure service time with no queueing: each
  frame "arrives" when the previous one finishes.
- **Frames are pre-decoded into RAM** (100 frames, cycled), so JPEG decode and
  disk I/O, which a camera pipeline doesn't have, are not measured.

### How each stage is timed

- Clock: `time.perf_counter_ns()` (QueryPerformanceCounter on Windows).
- The input lives in a device buffer allocated once. Inference runs through
  ORT IOBinding with outputs left on the device, so the H2D copy, the GPU
  compute and the D2H copy are separate, observable steps.
- **`cudaDeviceSynchronize()` is called after the H2D copy and after
  inference**, through the same CUDA runtime DLL ORT uses (same primary
  context). Without it, asynchronous GPU work would be charged to whichever
  later call happens to block. Each sync costs a few µs, and that cost is
  included in the stage it follows.
- The `gpu` stage is host-observed: kernel launches plus ORT's host-side
  overhead plus device execution. It is not a pure device-side kernel time.
  For a deadline that is the relevant number. CUDA-event device timing isn't
  used because ORT runs on its own stream.

### Warmup, sample size, statistics

- 50 closed-loop warmup iterations are discarded before every run. The very
  first iteration (cuDNN's exhaustive algorithm search plus lazy allocations)
  is reported separately as `cold_first_iter_ms`, not mixed into the
  steady-state stats.
- 1000 measured frames per run. Reported: p50, p95, p99, max, mean, std.
  Each percentile has a **95% percentile-bootstrap CI** (1000 resamples). With
  n = 1000, p99 rests on about 10 samples, and the CI shows how loose that is.
- Python's garbage collector is paused during the timed loop. A GC pause
  would otherwise land in the tail as an artifact of the harness, not the
  model.
- Each (model, fps) run is a **fresh process**, so device memory and cuDNN
  state don't carry over. There is a 15 s idle cooldown between runs.

### Memory

Per-process GPU memory isn't available under WDDM. `cudaMemGetInfo` (device-wide)
is sampled before the CUDA context exists (via nvidia-smi), after context
creation, after session creation, after warmup, and after every frame (outside
the timed stages). `vram_delta` = peak during the run minus usage right after
context creation. ORT's arena uses `kSameAsRequested`, so it grows only by
what is requested, not in power-of-two steps. Other processes' allocations
during a run would inflate the figure. The baseline-to-peak change is reported
rather than a claimed per-process value.

### Conditions logged with every run

`nvidia-smi` samples SM/memory clocks, temperature, power, utilisation,
P-state and **clock-event (throttle) reasons** every 100 ms. It runs as a
separate process so it doesn't compete with the timed loop for the Python
GIL. Before and after each run the harness also records available RAM,
commit charge, AC/battery state and the Windows power plan. A run on battery
or under memory pressure is identifiable from its JSON.

### Stage 2 results: single-model baselines (`results/stage2/summary.md`)

Measured 2026-10-09 on the machine above, on AC power. Available RAM at run
start was only 1.7–2.4 GiB (other apps open), which likely inflates the
CPU-stage tails. 1000 frames per run; ms; p-values with 95% bootstrap CI.

| Model | Arrivals | e2e p50 | e2e p99 | e2e max | GPU p50 / p99 | pre p50 | post p50 | Misses > 33 ms | VRAM Δ | SM clock mean |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv10n | closed loop | 18.36 [18.06, 18.70] | 39.67 [36.90, 40.54] | 48.06 | 8.62 / 16.13 | 8.40 | 0.13 | 39/1000 | 116 MiB | 1552 MHz |
| YOLOv10n | 30 FPS | 20.66 [20.53, 20.81] | 46.42 [29.37, 57.25] | 62.47 | 13.77 / 14.80 | 6.50 | 0.10 | 15/1000 | 116 MiB | 797 MHz |
| SegFormer-B0 576×1024 | closed loop | 83.39 [83.23, 83.69] | 123.97 [115.27, 134.80] | 148.73 | 34.14 / 37.81 | 15.41 | 31.53 | 1000/1000 | 850 MiB | 1710 MHz |
| SegFormer-B0 576×1024 | 30 FPS | (unbounded queue; see below) | | | 34.17 / 34.45 | 15.20 | 32.31 | 1000/1000 | 850 MiB | 1724 MHz |

Observations (all from the run JSONs):

1. **CPU pre/post-processing is as large as GPU inference.** For YOLOv10n,
   CPU preprocessing (letterbox + float conversion) is 8.4 ms p50 against
   8.6 ms on the GPU, and it dominates the tail (pre p99 21.6 ms). For
   SegFormer, post-processing (bilinear upsampling of 19-channel logits to
   1280×720 on the CPU, then argmax) costs 31.5 ms, almost as much as
   inference.
2. **SegFormer at 576×1024 can't meet 33 ms on its own.** GPU time alone
   is 34.1 ms p50, and service time is 83 ms. At 30 FPS arrivals the queue
   grows without bound: achieved throughput is 11.7 FPS, and e2e reaches
   51 s by the 1000th frame. That number reflects backlog length, not a
   per-frame latency, and is reported only to show the overload.
3. **Paced arrivals make the GPU slower per frame.** At 30 FPS the detector's
   GPU stage is 13.8 ms p50 versus 8.6 ms back-to-back. Telemetry shows the
   mean SM clock falling from 1552 to 797 MHz, with `gpu_idle` clock-event
   reasons: the GPU down-clocks in the idle gaps between frames. Clocks can't
   be locked on this GeForce laptop GPU, so closed-loop numbers understate
   real-time latency.
4. **SegFormer runs hit the 48 W power cap** (`sw_power_cap` in 146 of 217
   samples in the closed-loop run).
5. **Run-to-run variance is material.** A smoke-test run of the same
   detector/closed-loop config 20 minutes earlier gave p50 16.4 ms and p99
   24.2 ms, versus 18.4 / 39.7 here, with less free RAM this time. Stage 3
   will repeat each configuration rather than trust a single run.
6. **SegFormer's 850 MiB VRAM Δ** is mostly cuDNN workspace
   (`cudnn_conv_use_max_workspace=1`, the ORT default), allocated during
   warmup.
7. *Data caveat, found in Stage 2b:* the 100-frame preload set included one
   of the two portrait frames described below (`80a9e37d-e4548ac1`, preload
   index 10), so 10 of each run's 1000 measured frames were 720×1280. The CPU
   path handles any size. From Stage 2b on, the frame source excludes
   non-1280×720 frames and lists them in each result JSON.

## Stage 2b: making segmentation feasible

Stage 2 showed SegFormer at 576×1024 could not meet 33 ms alone. Before
Stage 3, two changes were made and measured against the Stage 2 numbers.

### Pre/post-processing moved into the graph

`scripts/build_fused.py` wraps each exported model with ONNX ops so the
fused graph takes the **raw camera frame** (`uint8 [1,720,1280,3]` BGR) and
does on the GPU what `bsched/models.py` did on the CPU:

- detector: Cast → NCHW → BGR→RGB → Resize (bilinear, half-pixel) → Pad 114
  (letterbox) → ×1/255 → YOLOv10n → `[1,300,6]` (box decode stays on CPU; it
  is ~0.1 ms)
- segmenter: Cast → NCHW → RGB → Resize → ImageNet normalise → SegFormer →
  bilinear upsample of logits to 720×1280 → ArgMax → `uint8` mask `[1,720,1280]`

That also shrinks the copies: the H2D transfer goes from 4.9–7.1 MB of float32
to 2.8 MB of uint8, and the segmentation D2H goes from 19 logit channels to a
0.9 MB mask.

**Equivalence** (`scripts/verify_fused.py`, 20 frames, both paths on the
CUDA EP; `results/stage2b/fused_equivalence.json`):

| | Fused vs Stage 1/2 CPU path |
|---|---|
| Detector | 126/127 detections matched (same class, IoU ≥ 0.991); max box diff 2.5 px; max score diff 0.032 |
| SegFormer (each resolution) | mask pixel agreement mean 99.4–99.6%, min 97.6% |

The SegFormer residual was traced, not assumed. The CPU path resizes in uint8
(cv2 rounds to integers) and the graph resizes in float. A CPU path with a
float resize agrees with the fused graph on **99.98–100%** of pixels, so the
whole gap is that ±0.5/255 input rounding. This is a model sensitivity, not a
fusion error, and it costs no accuracy (mIoU below). It's also a warning for
INT8 in Stage 5: SegFormer flips up to ~2% of pixels under a half-level
input perturbation.

### Segmentation resolution sweep

SegFormer-B0 exported at 576×1024, 448×800, 384×672 and 288×512 (all
divisible by 32), each torch-vs-ONNX checked (max |Δ| ≤ 1.9e-5, 100% argmax
agreement).

**mIoU** (`scripts/eval_seg.py`; first 200 labelled 10k-val frames;
dataset-level IoU from an accumulated 19-class confusion matrix, GT 255
ignored; `results/stage2b/seg_miou.json`):

| Variant | mIoU | Pixel acc |
|---|---|---|
| CPU path 576×1024 (Stage 2 baseline) | 0.3973 | 0.853 |
| Fused 576×1024 | 0.3976 | 0.854 |
| Fused 448×800 | 0.3972 | 0.865 |
| Fused 384×672 | 0.3924 | 0.866 |
| Fused 288×512 | 0.3767 | 0.865 |

The absolute mIoU (~0.40) is a Cityscapes-trained model evaluated on BDD100K
without fine-tuning. The domain gap, not the pipeline, sets it.

**Excluded data:** two 10k-val images (`80a9e37d-e4548ac1`,
`9342e334-33d167eb`) are portrait phone frames (720×1280) whose masks are
1280×720 landscape canvases. The labelled scene occupies the top-left
720×720 and the rest is filled with class 0 ("road"). Those labels are
defective. Both frames are excluded from evaluation and from the stream and
listed in the outputs; they are not corrected.

**Selection rule (fixed before measuring latency):** the deployed
segmentation resolution is the largest whose fused SegFormer p50 + fused
detector p50, both at 30 FPS open-loop, is ≤ 0.8 × 33 ms = 26.4 ms. The 20%
margin is for tails and for the down-clocking seen in Stage 2.

### Stage 2b latency results (`results/stage2b/*.json`)

Same harness and settings as Stage 2, with the final fused graphs
(elementwise argmax; see the VRAM section below). 1000 frames per run; ms;
95% bootstrap CI in brackets. All runs on AC power; available RAM at start
1.6–2.6 GiB.

| Variant | Arrivals | e2e p50 | e2e p99 | e2e max | GPU p50 / p99 | CPU pre+post p50 | Misses > 33 ms | VRAM Δ | SM clock mean |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv10n fused | closed | 9.77 [9.76, 9.79] | 11.92 [11.56, 13.62] | 17.8 | 9.04 / 10.94 | 0.11 | 0/1000 | 94 MiB | 1740 MHz |
| YOLOv10n fused | 30 FPS | 15.38 [15.37, 15.40] | 16.83 [16.52, 17.39] | 21.8 | 14.52 / 15.83 | 0.12 | 0/1000 | 94 MiB | 896 MHz |
| SegFormer fused 576×1024 | closed | 40.32 [40.31, 40.33] | 41.23 [41.11, 41.34] | 41.8 | 39.44 / 40.28 | 0.01 | 1000/1000 | 846 MiB | 1652 MHz |
| SegFormer fused 576×1024 | 30 FPS | overload: 24.8 FPS achieved, queue grows | | | 39.42 / 40.48 | 0.01 | 1000/1000 | 846 MiB | 1651 MHz |
| SegFormer fused 448×800 | closed | 25.44 [25.43, 25.45] | 25.95 [25.88, 26.02] | 27.0 | 24.55 / 24.92 | 0.01 | 0/1000 | 546 MiB | 1680 MHz |
| SegFormer fused 448×800 | 30 FPS | 25.59 [25.57, 25.62] | 28.14 [27.88, 28.72] | 30.6 | 24.52 / 26.00 | 0.02 | 0/1000 | 546 MiB | 1723 MHz |
| SegFormer fused 384×672 | closed | 19.50 [19.48, 19.50] | 20.96 [20.24, 22.05] | 23.9 | 18.58 / 19.37 | 0.01 | 0/1000 | 486 MiB | 1734 MHz |
| SegFormer fused 384×672 | 30 FPS | 19.79 [19.76, 19.81] | 25.32 [23.42, 27.01] | 42.9 | 18.69 / 22.90 | 0.02 | 1/1000 | 486 MiB | 1736 MHz |
| SegFormer fused 288×512 | closed | 13.96 [13.94, 13.98] | 18.56 [17.02, 19.04] | 29.7 | 13.04 / 17.48 | 0.01 | 0/1000 | 358 MiB | 1736 MHz |
| SegFormer fused 288×512 | 30 FPS | 14.31 [14.30, 14.32] | 18.26 [16.93, 19.60] | 21.3 | 13.33 / 17.27 | 0.01 | 0/1000 | 358 MiB | 1641 MHz |

Against Stage 2 (CPU pre/post):

- **YOLOv10n at 30 FPS:** e2e p50 20.66 → 15.38 ms, p99 46.42 → 16.83 ms,
  misses 15 → 0 per 1000. CPU pre-processing went from 6.5 ms to ~0.02 ms
  (just a batch view).
- **SegFormer 576×1024, closed loop:** e2e p50 83.4 → 40.3 ms. The GPU stage
  grew 34.1 → 39.4 ms because it now includes the resize, the 720×1280
  logit upsample and the argmax, but ~47 ms of CPU work disappeared. It still
  cannot sustain 30 FPS alone (24.8 FPS achieved).
- **Tails tightened.** Most of Stage 2's tail came from the CPU stages under
  memory pressure.
- **VRAM:** SegFormer 576×1024 is 850 → 846 MiB, unchanged overall even
  though the upsample/argmax now run on the GPU.

**Run-to-run variance at paced FPS is several ms.** The same fused detector
graph at 30 FPS measured e2e p50 12.96 ms in an earlier run of this stage
(with the ArgMax-era graphs; the detector graph was identical) and 15.38 ms
here. Mean SM clock was 1155 vs 896 MHz: the down-clocking between frames
varies by run. The detector is the light workload that idles the GPU most.
SegFormer ≥ 384×672 keeps clocks near boost. Stage 3 repeats each
configuration for this reason.

**Selection-rule outcome: no resolution qualifies.** With these numbers the
detector's p50 at 30 FPS is 15.38 ms, so the rule needs SegFormer p50
≤ 11.0 ms:

| SegFormer | p50 @ 30 FPS | det + seg p50 | ≤ 26.4 ms? | ≤ 33 ms? | mIoU |
|---|---|---|---|---|---|
| 576×1024 | overload | n/a | no | no | 0.3976 |
| 448×800 | 25.59 | 40.97 | no | no | 0.3972 |
| 384×672 | 19.79 | 35.17 | no | no | 0.3924 |
| 288×512 | 14.31 | 29.69 | no | yes | 0.3767 |

(With the earlier run's detector p50 of 12.96 ms, 384×672 summed to 31.45 ms
and fit under 33 ms. That it now doesn't is the run-to-run variance above.)
Sums of single-model p50s are only a guide: real sequential latency is what
Stage 3 measures, and when both models run every frame the GPU idles less,
so clocks may stay higher than in either single-model run.

**Decision (project owner): degrade ladder.** No single resolution meets the
rule, so segmentation degrades per frame in Stage 4: **384×672 → 288×512 →
skip**, chosen by remaining budget. This matches the brief ("segmentation
degrades or is dropped"). Stage 3 measures sequential det+seg at both ladder
resolutions.

### VRAM: ORT's CUDA ArgMax was the problem

With the first fused graphs, detector + both ladder SegFormers in one
process left only **375 MiB of the 4 GB free**. Each fused SegFormer added
~1.4 GB even though the plain SegFormer graph needs 254 MiB (288×512) or
414 MiB (384×672). Hypotheses were tested one at a time, each in a fresh
process:

| Hypothesis | Test | Result |
|---|---|---|
| cuDNN conv workspace | `cudnn_conv_use_max_workspace=0` | no change (1412 MiB) |
| EXHAUSTIVE algo search trial buffers | `cudnn_conv_algo_search=HEURISTIC` | no change (1412 / 1382 MiB) |
| The fused post-processing | Resize+ArgMax graph in isolation | Resize alone 90 MiB; **Resize+ArgMax 1190 MiB** |

ORT's CUDA `ArgMax` over a 19×720×1280 tensor held ~1.1 GB, and
`ReduceMax`/`ReduceMin` had the same problem. The fused graph now computes
the channel argmax without a reduction op: `Split` into 19 channels →
variadic elementwise `Max` → a descending chain of `Equal`/`Where` (lowest
index wins ties, exactly as `ArgMax` does).

| | ArgMax post | Elementwise post |
|---|---|---|
| Isolated probe VRAM (fresh process, ×2) | 1190 MiB | 300 MiB |
| Isolated probe time incl. D2H | 3.3–3.9 ms | 4.2–4.3 ms |
| Masks vs ArgMax graph, 50 real frames × 4 resolutions | | **bit-identical** |
| det + seg384 + seg288 in one process: VRAM Δ / free | 2928 MiB / 375 MiB | **940 MiB / 2363 MiB** |

The masks are bit-identical, so the mIoU and equivalence results above
carry over unchanged. Latency was re-measured with the new graphs (table
above).

Methodology note: an earlier attempt measured several variants one after
another in a single process and got inconsistent figures (ArgMax 650 vs
1190 MiB), because ORT arenas don't release memory to the device
predictably. Memory comparisons are only reported from fresh processes.

### Outputs

`results/stage2/<model>_fps<N>.json` holds the full config snapshot, ORT
provider options, environment, memory, telemetry summary, summary stats and
**per-frame arrays** (arrival offset, e2e, queue, service and each stage, in
ms), so plots can be regenerated without re-running. The raw telemetry CSV
sits next to it. `summary.json` / `summary.md` hold the cross-run table.
