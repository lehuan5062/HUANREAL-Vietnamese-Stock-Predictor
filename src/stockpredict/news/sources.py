"""URL builders for Vietnamese + global news sources."""
from __future__ import annotations

from ..config import load_config


def vn_urls(symbol: str) -> dict[str, str]:
    cfg = load_config().news
    out = {}
    for name, tmpl in cfg["vn_sources"].items():
        try:
            out[name] = tmpl.format(symbol=symbol.upper())
        except KeyError:
            out[name] = tmpl  # template doesn't use {symbol}
    return out


def global_urls() -> dict[str, str]:
    return dict(load_config().news["global_sources"])
