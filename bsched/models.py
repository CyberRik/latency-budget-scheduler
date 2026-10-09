"""ONNX Runtime sessions plus pre/post-processing for the two models.

Pre/post-processing is pure numpy/OpenCV on the CPU and is kept separate from
`infer()` so later stages can time GPU inference and CPU work independently.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

_PRELOADED = False


def _preload_cuda_dlls() -> None:
    """Load CUDA/cuDNN from the nvidia-* pip wheels (no system toolkit needed).

    preload_dlls() loads a fixed list of DLLs, but cuDNN 9 lazily loads further
    sub-libraries (e.g. cudnn_engines_tensor_ir64_9.dll) by name at first use.
    Those must be findable on the DLL search path, so every nvidia/*/bin
    directory is added to it as well.
    """
    global _PRELOADED
    if _PRELOADED:
        return
    import glob
    import sys
    site = os.path.dirname(os.path.dirname(ort.__file__))
    for d in glob.glob(os.path.join(site, "nvidia", "*", "bin")):
        if sys.platform == "win32":
            os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    _PRELOADED = True


_OPT_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def make_session(onnx_path: str, rt: dict, device: str = "cuda",
                 profile_dir: str | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = _OPT_LEVELS[rt.get("graph_optimization", "all")]
    so.log_severity_level = 3
    if profile_dir:
        so.enable_profiling = True
        so.profile_file_prefix = os.path.join(profile_dir, "ort_profile")
    if device == "cuda":
        _preload_cuda_dlls()
        providers = [("CUDAExecutionProvider", {
            "device_id": rt.get("device_id", 0),
            "cudnn_conv_algo_search": rt.get("cudnn_conv_algo_search", "EXHAUSTIVE"),
            "arena_extend_strategy": rt.get("arena_extend_strategy", "kNextPowerOfTwo"),
            "use_tf32": int(rt.get("use_tf32", True)),
            "cudnn_conv_use_max_workspace": int(rt.get("cudnn_conv_use_max_workspace", True)),
        }), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    if device == "cuda" and rt.get("require_cuda", True):
        # The Python wrapper otherwise silently rebuilds the session CPU-only
        # if a CUDA kernel fails at run time.
        sess.disable_fallback()
        if sess.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA EP not active for {onnx_path}: {sess.get_providers()}. "
                               "Check onnxruntime-gpu install and CUDA/cuDNN DLLs.")
    return sess


def node_placement(onnx_path: str, rt: dict, feed: dict) -> dict:
    """Run once with profiling on and count which EP executed each node.

    get_providers() only lists registered EPs; this shows where nodes ran.
    """
    with tempfile.TemporaryDirectory() as td:
        sess = make_session(onnx_path, rt, "cuda", profile_dir=td)
        sess.run(None, feed)
        prof = sess.end_profiling()
        with open(prof) as f:
            events = json.load(f)
    per_node = {}
    for e in events:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            args = e.get("args", {})
            per_node[e["name"][: -len("_kernel_time")]] = (args.get("provider"), args.get("op_name"))
    by_ep = Counter(p for p, _ in per_node.values())
    cpu_ops = Counter(op for p, op in per_node.values() if p == "CPUExecutionProvider")
    return {"nodes_by_provider": dict(by_ep), "cpu_ops": dict(cpu_ops)}


# ---------------------------------------------------------------- detector

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float


def letterbox_params(frame_hw, input_hw) -> tuple[Letterbox, int, int]:
    """Letterbox geometry shared by the CPU path and the fused graph."""
    H, W = input_hw
    h, w = frame_hw
    s = min(H / h, W / w)
    nh, nw = round(h * s), round(w * s)
    top, left = round((H - nh) / 2 - 0.1), round((W - nw) / 2 - 0.1)
    return Letterbox(s, left, top), nh, nw


def det_preprocess(bgr: np.ndarray, input_hw) -> tuple[np.ndarray, Letterbox]:
    """Ultralytics-style letterbox (pad value 114), RGB, [0,1], NCHW float32."""
    H, W = input_hw
    lb, nh, nw = letterbox_params(bgr.shape[:2], input_hw)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((H, W, 3), 114, np.uint8)
    canvas[lb.pad_y:lb.pad_y + nh, lb.pad_x:lb.pad_x + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), lb


def det_postprocess(out: np.ndarray, lb: Letterbox, score_thr: float, orig_hw) -> np.ndarray:
    """YOLOv10 is NMS-free: output is [1, 300, 6] = x1,y1,x2,y2,score,cls.

    Returns [N, 6] in original image pixels, filtered by score.
    """
    d = out[0]
    d = d[d[:, 4] >= score_thr].copy()
    d[:, [0, 2]] = (d[:, [0, 2]] - lb.pad_x) / lb.scale
    d[:, [1, 3]] = (d[:, [1, 3]] - lb.pad_y) / lb.scale
    h, w = orig_hw
    d[:, [0, 2]] = d[:, [0, 2]].clip(0, w)
    d[:, [1, 3]] = d[:, [1, 3]].clip(0, h)
    return d


# --------------------------------------------------------------- segmenter

CITYSCAPES_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                    "truck", "bus", "train", "motorcycle", "bicycle"]
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]], np.uint8)  # RGB

_IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def seg_preprocess(bgr: np.ndarray, input_hw) -> np.ndarray:
    """Matches SegformerImageProcessor: resize, RGB, /255, ImageNet normalise."""
    H, W = input_hw
    rgb = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    x = (rgb.astype(np.float32) / 255.0 - _IMNET_MEAN) / _IMNET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def seg_postprocess(logits: np.ndarray, orig_hw) -> np.ndarray:
    """Bilinear-upsample logits to original size, then argmax -> [H, W] uint8."""
    h, w = orig_hw
    lg = logits[0].transpose(1, 2, 0)                     # [h/4, w/4, C]
    up = cv2.resize(lg, (w, h), interpolation=cv2.INTER_LINEAR)
    return up.argmax(-1).astype(np.uint8)
