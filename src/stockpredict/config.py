"""Load config.yaml once and expose a cached, attribute-accessible view."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import PROJECT_ROOT


class Config(dict):
    """Dict that also exposes keys as attributes for ergonomic access."""

    def __getattr__(self, key: str) -> Any:
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(v) if isinstance(v, dict) else v


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def cache_dir() -> Path:
    cfg = load_config()
    d = PROJECT_ROOT / cfg.data["cache_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir() -> Path:
    d = PROJECT_ROOT / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = PROJECT_ROOT / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
