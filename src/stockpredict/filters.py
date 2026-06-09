"""Liquidity / quality gating applied before ranking."""
from __future__ import annotations

import pandas as pd

from .config import load_config


def liquidity_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series aligned to df.index: True where the row passes the filter.
    Expects df to have columns [close, adv_active_days_20] and a date index per
    symbol.

    The traded-value gate counts how many of the last 20 days actually cleared
    the per-day bar (``min_adv_vnd``) and requires at least
    ``min_adv_active_days`` of them. This rejects mostly-dead stocks that only
    pass a mean-ADV test because of a single block-trade spike (e.g. VMS, whose
    median day prints a single frozen lot)."""
    cfg = load_config().universe["liquidity_filter"]
    cond = (
        (df["close"] >= cfg["min_close_vnd"])
        & (df["adv_active_days_20"] >= cfg.get("min_adv_active_days", 15))
    )
    return cond.fillna(False)


def has_enough_history(df: pd.DataFrame) -> bool:
    cfg = load_config().universe["liquidity_filter"]
    return len(df) >= cfg["min_history_days"]
