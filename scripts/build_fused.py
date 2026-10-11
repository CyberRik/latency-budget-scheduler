"""Build "fused" ONNX models that take the raw camera frame and do the
pre/post-processing on the GPU, inside the graph.

  detector:   frame uint8 [1,H,W,3] BGR
              -> Cast f32 -> NHWC->NCHW -> BGR->RGB -> Resize(linear, half_pixel)
              -> Pad(114) to square (letterbox) -> *1/255 -> YOLOv10n -> [1,300,6]
  segmenter:  frame uint8 [1,H,W,3] BGR
              -> Cast -> NCHW -> RGB -> Resize(linear, half_pixel) -> ImageNet normalise
              -> SegFormer -> logits [1,19,h/4,w/4]
              -> Resize logits (linear, half_pixel) to H x W -> ArgMax -> uint8 mask [1,H,W]

These ops mirror bsched/models.py's CPU path. One difference: cv2.resize on
uint8 rounds the resized image to integers, while the in-graph Resize runs in
float. scripts/verify_fused.py measures the effect on outputs.

The frame size is fixed (BDD100K is 1280x720), so the graphs are static-shape.

Needs only `onnx` (export toolchain).  Usage: python scripts/build_fused.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper

from bsched.config import load_config, resolve

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


def _model(nodes, inputs, outputs, inits, opset, ir_version, name):
    g = helper.make_graph(nodes, name, inputs, outputs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = ir_version
    return m


def _frame_to_nchw_rgb(H, W):
    """Nodes: frame(uint8 NHWC BGR) -> rgb (float NCHW). Returns (nodes, inits)."""
    nodes = [
        helper.make_node("Cast", ["frame"], ["f32"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["f32"], ["nchw"], perm=[0, 3, 1, 2]),
        helper.make_node("Gather", ["nchw", "bgr2rgb"], ["rgb"], axis=1),
    ]
    return nodes, [_const("bgr2rgb", np.array([2, 1, 0], np.int64))]


def _resize(src, dst, sizes_name):
    return helper.make_node("Resize", [src, "", "", sizes_name], [dst], mode="linear",
                            coordinate_transformation_mode="half_pixel")


def build_detector(core: onnx.ModelProto, frame_hw, input_hw) -> onnx.ModelProto:
    H, W = frame_hw
    IH, IW = input_hw
    opset, irv = core.opset_import[0].version, core.ir_version
    in_name = core.graph.input[0].name

    # Same letterbox geometry as bsched.models.det_preprocess.
    s = min(IH / H, IW / W)
    nh, nw = round(H * s), round(W * s)
    top, left = round((IH - nh) / 2 - 0.1), round((IW - nw) / 2 - 0.1)
    bottom, right = IH - nh - top, IW - nw - left

    nodes, inits = _frame_to_nchw_rgb(H, W)
    nodes += [
        _resize("rgb", "resized", "rs_sizes"),
        helper.make_node("Pad", ["resized", "pads", "padval"], ["padded"], mode="constant"),
        helper.make_node("Mul", ["padded", "inv255"], [in_name]),
    ]
    inits += [_const("rs_sizes", np.array([1, 3, nh, nw], np.int64)),
              _const("pads", np.array([0, 0, top, left, 0, 0, bottom, right], np.int64)),
              _const("padval", np.array(114.0, np.float32)),
              _const("inv255", np.array(1 / 255.0, np.float32))]
    pre = _model(nodes, [helper.make_tensor_value_info("frame", TensorProto.UINT8, [1, H, W, 3])],
                 [helper.make_tensor_value_info(in_name, TensorProto.FLOAT, [1, 3, IH, IW])],
                 inits, opset, irv, "det_pre")
    fused = compose.merge_models(pre, core, io_map=[(in_name, in_name)])
    fused.graph.name = "yolov10n_fused"
    return fused


def build_segmenter(core: onnx.ModelProto, frame_hw, input_hw, argmax: str = "elementwise") -> onnx.ModelProto:
    """argmax="elementwise" (CUDA EP, Stages 2-4) or "onnx" (TensorRT variants,
    Stage 5): TensorRT's parser rejects the uint8 Equal/Where chain, and its
    own ArgMax is not ORT's CUDA kernel, so TensorRT graphs use ArgMax + Cast."""
    H, W = frame_hw
    IH, IW = input_hw
    opset, irv = core.opset_import[0].version, core.ir_version
    in_name, out_name = core.graph.input[0].name, core.graph.output[0].name
    C = core.graph.output[0].type.tensor_type.shape.dim[1].dim_value

    nodes, inits = _frame_to_nchw_rgb(H, W)
    nodes += [
        _resize("rgb", "resized", "rs_sizes"),
        helper.make_node("Sub", ["resized", "mean255"], ["centered"]),
        helper.make_node("Mul", ["centered", "inv_std255"], [in_name]),
    ]
    inits += [_const("rs_sizes", np.array([1, 3, IH, IW], np.int64)),
              _const("mean255", (IMNET_MEAN * 255).reshape(1, 3, 1, 1)),
              _const("inv_std255", (1 / (IMNET_STD * 255)).reshape(1, 3, 1, 1).astype(np.float32))]
    pre = _model(nodes, [helper.make_tensor_value_info("frame", TensorProto.UINT8, [1, H, W, 3])],
                 [helper.make_tensor_value_info(in_name, TensorProto.FLOAT, [1, 3, IH, IW])],
                 inits, opset, irv, "seg_pre")

    # Channel argmax built from elementwise ops instead of ArgMax. ORT's CUDA
    # ArgMax (a cuDNN reduction) held ~1190 MiB for a 19x720x1280 input in an
    # isolated fresh-process probe; this form held ~300 MiB, ran ~0.5-1 ms
    # slower, and gave bit-identical output (ties resolve to the lowest index,
    # as ArgMax does). See README, Stage 2b.
    if argmax == "onnx":
        post = _model([_resize(out_name, "up", "up_sizes"),
                       helper.make_node("ArgMax", ["up"], ["am"], axis=1, keepdims=0),
                       helper.make_node("Cast", ["am"], ["mask"], to=TensorProto.UINT8)],
                      [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)],
                      [helper.make_tensor_value_info("mask", TensorProto.UINT8, [1, H, W])],
                      [_const("up_sizes", np.array([1, C, H, W], np.int64))], opset, irv, "seg_post")
        fused = compose.merge_models(pre, core, io_map=[(in_name, in_name)])
        fused = compose.merge_models(fused, post, io_map=[(out_name, out_name)])
        fused.graph.name = "segformer_fused"
        return fused
    post_nodes = [
        _resize(out_name, "up", "up_sizes"),
        helper.make_node("Split", ["up", "split_ones"], [f"ch{k}" for k in range(C)], axis=1),
        helper.make_node("Max", [f"ch{k}" for k in range(C)], ["chmax"]),
    ]
    post_inits = [_const("up_sizes", np.array([1, C, H, W], np.int64)),
                  _const("split_ones", np.ones(C, np.int64)),
                  _const("idx_last", np.full((1, 1, 1, 1), C - 1, np.uint8)),
                  _const("squeeze_ax", np.array([1], np.int64))]
    prev = "idx_last"
    for k in range(C - 2, -1, -1):          # descending: the lowest matching index is applied last
        post_nodes += [helper.make_node("Equal", [f"ch{k}", "chmax"], [f"is{k}"]),
                       helper.make_node("Where", [f"is{k}", f"idx{k}", prev], [f"sel{k}"])]
        post_inits.append(_const(f"idx{k}", np.array(k, np.uint8)))
        prev = f"sel{k}"
    post_nodes.append(helper.make_node("Squeeze", [prev, "squeeze_ax"], ["mask"]))
    post = _model(post_nodes,
                  [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)],
                  [helper.make_tensor_value_info("mask", TensorProto.UINT8, [1, H, W])],
                  post_inits, opset, irv, "seg_post")

    fused = compose.merge_models(pre, core, io_map=[(in_name, in_name)])
    fused = compose.merge_models(fused, post, io_map=[(out_name, out_name)])
    fused.graph.name = "segformer_fused"
    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    frame_hw = cfg["data"]["frame_hw"]

    jobs = []
    for key, m in cfg["models"].items():
        if not m.get("fused"):
            continue
        if "@" in key:
            # Stage 5: TensorRT FP32/FP16 segmenters get the ArgMax post graph.
            # (INT8 graphs are built by quantize_int8.py.)
            if m["task"] == "segmentation" and m["variant"] == "trt_fp32":   # fp16 shares the file
                jobs.append((key, resolve(cfg, m["source_onnx"]), resolve(cfg, m["onnx"]),
                             lambda c, f, i: build_segmenter(c, f, i, argmax="onnx"), m["input_hw"]))
            continue
        src = resolve(cfg, m["source_onnx"])
        build = build_detector if m["task"] == "detection" else build_segmenter
        jobs.append((key, src, resolve(cfg, m["onnx"]), build, m["input_hw"]))
    for key, src, dst, build, input_hw in jobs:
        fused = build(onnx.load(str(src)), frame_hw, input_hw)
        onnx.checker.check_model(fused, full_check=True)
        onnx.save(fused, str(dst))
        print(f"{key}: {src.name} -> {dst.name}  in={[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in fused.graph.input]}"
              f"  out={[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in fused.graph.output]}")


if __name__ == "__main__":
    main()
