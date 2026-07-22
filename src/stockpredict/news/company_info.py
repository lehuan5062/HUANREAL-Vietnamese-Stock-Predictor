"""Look up company name + (when available) sector for a ticker.

The Listing API gives us `organ_name` (e.g., "Cong ty co phan Tap doan FPT").
That's enough for an LLM to infer the business — combined with the ticker
mention in the prompt, Claude will identify the industry, the typical
2-day-horizon news drivers (commodity prices for producers, central bank
policy for banks, real estate inventory for developers, etc.), and search
accordingly.

For ETF rows (instrument_type=='ETF'), `organ_name` is typically the fund
manager (DCVFM, SSIAM, KIM, Mirae Asset, …); the LLM needs the underlying
index name instead, so the news prompts switch to an ETF-specific research
rubric when the enriched frame's instrument_type column flags them.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..data.universe import instrument_type as _instrument_type
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
    """Return a copy with `organ_name` + `instrument_type` columns attached.

    ``instrument_type`` is one of STOCK / ETF / CW / OTHER. Downstream news
    prompt builders branch on this to switch from company-business research
    (for stocks) to basket/flow/NAV reasoning (for ETFs).

    ETFs without a fund-manager name fall back to the ticker symbol itself
    as ``organ_name`` so the news plan shows e.g. ``FUEVFVND`` rather than
    a blank or ``(unknown)``. The ticker is more useful to the LLM than an
    empty string — the ETF rubric already tells it to derive the underlying
    index from the ticker shape. Stocks without names retain their existing
    fallback behavior (handled in the prompt builders).
    """
    out = candidates.copy()
    out["organ_name"] = out["symbol"].map(lambda s: organ_name(s))
    out["instrument_type"] = out["symbol"].map(lambda s: _instrument_type(s))
    is_etf_mask = out["instrument_type"].astype(str).str.upper() == "ETF"
    blank_name = out["organ_name"].isna() | (out["organ_name"].astype(str).str.strip() == "")
    fallback = is_etf_mask & blank_name
    out.loc[fallback, "organ_name"] = out.loc[fallback, "symbol"]
    return out
