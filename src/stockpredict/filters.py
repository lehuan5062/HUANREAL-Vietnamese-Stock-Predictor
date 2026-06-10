"""Liquidity / quality gating applied before ranking."""
from __future__ import annotations

from functools import lru_cache

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


@lru_cache(maxsize=1)
def _ceiling_params() -> tuple[dict, float]:
    cfg = load_config().pricing
    limits = dict(cfg.get("ceiling_limits",
                          {"HSX": 0.07, "HNX": 0.10, "UPCOM": 0.15}))
    tol = float(cfg.get("ceiling_tol", 0.015))
    return limits, tol


def ceiling_lock_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series aligned to df.index: True where the row is NOT locked
    limit-up (i.e. still buyable).

    A bar is a ceiling-lock when it closed at its high (``close_at_high``) AND
    its 1-day return (``ret_1d``) is within ``ceiling_tol`` of that exchange's
    daily price band. Such a name opens the next session with a buy queue and
    no sellers, so a limit-buy cannot fill — exclude it from the pickable
    universe. Symbols whose exchange is unknown, or frames lacking the support
    columns, are left untouched (treated as buyable)."""
    if "ret_1d" not in df.columns or "close_at_high" not in df.columns:
        return pd.Series(True, index=df.index)
    limits, tol = _ceiling_params()
    from .data.universe import load_universe
    uni = load_universe()
    ex_map = (dict(zip(uni["symbol"].astype(str), uni["exchange"]))
              if uni is not None and len(uni) else {})
    limit = df["symbol"].astype(str).map(ex_map).map(limits).astype(float)
    locked = (
        df["close_at_high"].fillna(False).astype(bool)
        & (df["ret_1d"] >= (limit - tol))
        & limit.notna()
    )
    return ~locked.fillna(False)
