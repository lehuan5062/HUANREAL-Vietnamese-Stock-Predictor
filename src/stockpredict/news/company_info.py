"""Look up company name + (when available) sector for a ticker.

The Listing API gives us `organ_name` (e.g., "Cong ty co phan Tap doan FPT").
That's enough for an LLM to infer the business — combined with the ticker
mention in the prompt, Claude/Gemini will identify the industry, the typical
2-day-horizon news drivers (commodity prices for producers, central bank
policy for banks, real estate inventory for developers, etc.), and search
accordingly.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..data.universe import load_universe


@lru_cache(maxsize=1)
def _name_index() -> dict[str, str]:
    try:
        u = load_universe(refresh=False)
    except Exception:
        return {}
    if "symbol" not in u.columns or "organ_name" not in u.columns:
        return {}
    return {
        str(row["symbol"]).upper(): str(row["organ_name"])
        for _, row in u.iterrows()
        if pd.notna(row.get("organ_name"))
    }


def organ_name(symbol: str) -> str:
    return _name_index().get(symbol.upper(), "")


def enrich(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an `organ_name` column attached."""
    out = candidates.copy()
    out["organ_name"] = out["symbol"].map(lambda s: organ_name(s))
    return out
