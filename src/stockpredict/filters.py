"""Liquidity / quality gating applied before ranking."""
from __future__ import annotations

import pandas as pd

from .config import load_config


def liquidity_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series aligned to df.index: True where the row passes the filter.
    Expects df to have columns [close, adv_vnd_20] and a date index per symbol."""
    cfg = load_config().universe["liquidity_filter"]
    cond = (
        (df["close"] >= cfg["min_close_vnd"])
        & (df["adv_vnd_20"] >= cfg["min_adv_vnd"])
    )
    return cond.fillna(False)


def has_enough_history(df: pd.DataFrame) -> bool:
    cfg = load_config().universe["liquidity_filter"]
    return len(df) >= cfg["min_history_days"]
