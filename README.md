# Latency-budget scheduler for a two-model perception stack

Object detection (YOLOv10n) and semantic segmentation (SegFormer-B0, Cityscapes)
run over a BDD100K frame stream on one GPU under a hard per-frame deadline
(33 ms). Detection must always complete; segmentation is degraded or dropped
when the budget is exhausted.

**Every number in `results/` was measured on the hardware recorded alongside it.
Nothing is interpolated or estimated. Anything that could not be measured is
listed as such.**

Status: Stage 1 (setup and correctness) complete. Stage 2 (measurement harness) next.

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
bsched/                  library: config, ORT sessions + pre/post, visualisation
scripts/export_models.py export both models to static-shape FP32 ONNX
scripts/prepare_data.py  sort BDD100K archives by content, extract val splits
scripts/verify_stage1.py stage-1 correctness check
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

To be written with Stage 2.
