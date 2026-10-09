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
    return cfg


def resolve(cfg: dict, rel: str) -> Path:
    """Resolve a config-relative path."""
    p = Path(rel)
    return p if p.is_absolute() else cfg["_root"] / p
