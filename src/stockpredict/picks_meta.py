"""Helpers for the picks DataFrame.

Selection is exactly-N by ``pred_mean`` (see ``model.predict.rank_today``),
so there is no per-category "best" badge to compute anymore — every returned
row is a pick. The only helper left is the filename suffix that lists the
returned tickers so a directory listing of ``reports/`` surfaces them at a
glance.
"""
from __future__ import annotations

import pandas as pd


def picks_suffix(picks: pd.DataFrame) -> str:
    """Filename suffix listing the returned tickers in DataFrame order:
    ``"_HII-MSB-HNG"``. Empty string when the frame is empty or lacks a
    ``symbol`` column. Used so a directory listing of ``reports/`` surfaces
    which tickers each report covers at a glance."""
    if picks is None or len(picks) == 0:
        return ""
    if "symbol" not in picks.columns:
        return ""
    tickers = picks["symbol"].astype(str).tolist()
    if not tickers:
        return ""
    return "_" + "-".join(tickers)
