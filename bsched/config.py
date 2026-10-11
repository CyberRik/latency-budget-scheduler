"""Config loading. Everything is driven from one TOML file (stdlib tomllib)."""
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.toml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_root"] = path.resolve().parent
    cfg["_path"] = str(path.resolve())
    expand_variants(cfg)
    return cfg


def expand_variants(cfg: dict) -> None:
    """Add a model entry "<model>@<variant>" for each Stage 5 precision variant.

    trt_int8 uses the QDQ graph written by scripts/quantize_int8.py
    ("..._fp32.onnx" -> "..._int8qdq.onnx"); the others use the FP32 graph.
    """
    s5 = cfg.get("stage5")
    if not s5:
        return
    for base in s5["models"]:
        m = cfg["models"][base]
        for v in s5["variants"]:
            e = dict(m, variant=v, base=base, name=f"{m['name']}@{v}")
            if v.startswith("trt_"):
                e["trt"] = {"precision": v.removeprefix("trt_")}
            if v.startswith("trt_") and m["task"] == "segmentation":
                # ArgMax post-processing graph (see scripts/build_fused.py)
                e["onnx"] = m["onnx"].replace("_fp32.onnx", "_trtpost_fp32.onnx")
            if v == "trt_int8":
                e["onnx"] = e["onnx"].replace("_fp32.onnx", "_int8qdq.onnx")
                e["source_onnx_fp32"] = m["source_onnx"]
                e["source_onnx"] = m["source_onnx"].replace("_fp32.onnx", "_int8qdq.onnx")
            cfg["models"][f"{base}@{v}"] = e


def relpath(cfg: dict, p: str | Path) -> str:
    """Path relative to the project root, for anything written into results."""
    try:
        return Path(p).resolve().relative_to(cfg["_root"]).as_posix()
    except ValueError:
        return str(p)


def resolve(cfg: dict, rel: str) -> Path:
    """Resolve a config-relative path."""
    p = Path(rel)
    return p if p.is_absolute() else cfg["_root"] / p
