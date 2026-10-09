"""Export YOLOv10n and SegFormer-B0 (Cityscapes) to static-shape FP32 ONNX.

Needs the export toolchain (torch, ultralytics, transformers, onnx); see
requirements-export.txt. The runtime venv does not need any of these.

Usage:  python scripts/export_models.py [--config config.toml]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx
import torch

from bsched.config import load_config, relpath, resolve


def export_yolo(cfg: dict) -> dict:
    from ultralytics import YOLO

    m = cfg["models"]["detector"]
    out = resolve(cfg, m["onnx"])
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = m["input_hw"]
    assert h == w, "ultralytics export takes a square imgsz"

    # Download weights into models/ rather than the cwd.
    weights = resolve(cfg, cfg["paths"]["models_dir"]) / m["weights"]
    model = YOLO(str(weights) if weights.exists() else m["weights"])
    if not weights.exists() and Path(m["weights"]).exists():
        shutil.move(m["weights"], weights)
    produced = model.export(format="onnx", imgsz=h, opset=cfg["export"]["opset"],
                            dynamic=False, simplify=False, half=False, batch=1)
    shutil.move(produced, out)
    return {"model": m["name"], "onnx": relpath(cfg, out), "source": m["weights"]}


class _LogitsOnly(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pixel_values):
        return self.net(pixel_values=pixel_values).logits


def export_segformer(cfg: dict, hw=None, out=None) -> dict:
    from transformers import SegformerForSemanticSegmentation

    m = cfg["models"]["segmenter"]
    out = resolve(cfg, out or m["onnx"])
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = hw or m["input_hw"]

    net = SegformerForSemanticSegmentation.from_pretrained(m["hf_id"]).eval()
    wrapped = _LogitsOnly(net).eval()
    dummy = torch.randn(1, 3, h, w)
    torch.onnx.export(wrapped, (dummy,), str(out), opset_version=cfg["export"]["opset"],
                      input_names=["pixel_values"], output_names=["logits"],
                      dynamic_axes=None, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(out)))

    # Parity: ONNX (CPU EP) vs the PyTorch model on a fixed random input.
    import onnxruntime as ort
    x = torch.randn(1, 3, h, w, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = wrapped(x).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"pixel_values": x.numpy()})[0]
    parity = {
        "max_abs_diff": float(np.abs(ref - got).max()),
        "argmax_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
        "output_shape": list(got.shape),
    }
    return {"model": m["name"], "onnx": relpath(cfg, out), "source": m["hf_id"],
            "input_hw": [h, w], "num_labels": net.config.num_labels, "id2label": net.config.id2label,
            "torch_vs_onnx_cpu": parity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    import transformers
    import ultralytics
    report = {
        "toolchain": {"torch": torch.__version__, "onnx": onnx.__version__,
                      "ultralytics": ultralytics.__version__,
                      "transformers": transformers.__version__},
        "detector": export_yolo(cfg),
        "segmenter": export_segformer(cfg),
    }
    # Resolution sweep (Stage 2b): one static-shape export per input size.
    sweep = cfg["models"]["segmenter"].get("sweep_hw", [])
    pattern = cfg["models"]["segmenter"].get("sweep_onnx", "models/segformer_b0_{h}x{w}_fp32.onnx")
    for h, w in sweep:
        r = export_segformer(cfg, (h, w), pattern.format(h=h, w=w))
        report[f"segmenter_{h}x{w}"] = {k: v for k, v in r.items() if k != "id2label"}
    for key in [k for k in report if k != "toolchain"]:
        mp = onnx.load(str(resolve(cfg, report[key]["onnx"])))
        report[key]["inputs"] = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
                                 for i in mp.graph.input]
        report[key]["outputs"] = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                                  for o in mp.graph.output]
        report[key]["opset"] = mp.opset_import[0].version

    out = resolve(cfg, cfg["paths"]["models_dir"]) / "export_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()}, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
