"""Seeding, device selection, YAML config with command-line overrides, pickle helpers."""

from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """cfg['a']['b'] = value for dotted='a.b' (creates missing levels)."""
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply 'a.b=value' strings; the value is parsed as a YAML scalar/list."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like a.b=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_by_path(cfg, key.strip(), _parse_value(raw))
    return cfg


def _parse_value(raw: str) -> Any:
    """YAML-parse a value; YAML 1.1 leaves '1e-05' a string, so try int then float on strings."""
    val = yaml.safe_load(raw)
    if isinstance(val, str):
        for conv in (int, float):
            try:
                num = conv(val)
            except ValueError:
                continue
            if isinstance(num, float) and not math.isfinite(num):
                break  # keep 'nan' / 'inf' as text
            return num
    return val


SPLIT_SEED_POOL = [42, 7, 123, 2024, 99, 11, 3, 77, 500, 2025]


def split_seeds(cfg: Dict[str, Any]) -> List[int]:
    """cfg['data']['split_seed'] as a list of ints (it may be a single int or a list)."""
    raw = (cfg.get("data") or {}).get("split_seed", 42)
    seeds = [int(s) for s in raw] if isinstance(raw, (list, tuple)) else [int(raw)]
    if not seeds:
        raise ValueError("data.split_seed must not be an empty list")
    return seeds


def repeat_splits(cfg: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Copy of cfg with data.split_seed set to the first n entries of SPLIT_SEED_POOL."""
    if not 1 <= n <= len(SPLIT_SEED_POOL):
        raise ValueError(f"--repeat-splits must be between 1 and {len(SPLIT_SEED_POOL)}, got {n}")
    return apply_overrides(cfg, [f"data.split_seed={SPLIT_SEED_POOL[:n]}"])


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def save_pickle(obj: Any, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
