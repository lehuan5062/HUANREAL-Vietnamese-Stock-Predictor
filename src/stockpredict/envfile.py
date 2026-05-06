"""Tiny .env loader. Looks for a `.env` file at the project root and pushes
KEY=VALUE pairs into os.environ. Lets double-click .bat files pick up API keys
without the user editing system environment variables."""
from __future__ import annotations

import os

from . import PROJECT_ROOT


def load(path=None) -> dict[str, str]:
    p = path or (PROJECT_ROOT / ".env")
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)
            loaded[key] = val
    return loaded
