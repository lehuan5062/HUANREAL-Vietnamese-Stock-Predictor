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


# Margin above the legal price band before a 1-day move is called a corporate
# action. A legitimate limit move sits exactly AT the band (e.g. HOSE +7%), so a
# small margin keeps real limit days from being flagged while still catching the
# physically-impossible moves (VVS's -38% ex-rights gap is 5x the HOSE band).
_CORP_ACTION_MARGIN = 0.02


def _row_band_threshold(df: pd.DataFrame) -> pd.Series:
    """Per-row corporate-action threshold = that symbol's exchange price band +
    margin. Symbols whose exchange is unknown fall back to the WIDEST band
    (UPCOM 15%), so only moves impossible on EVERY exchange ever trip the
    filter. Vietnamese exchanges cap a single session's close-to-close move at
    the band (HOSE 7% / HNX 10% / UPCOM 15%); anything beyond it is a corporate
    action (split / rights / special dividend) showing through as an unadjusted
    price jump."""
    limits, _tol = _ceiling_params()
    widest = max(limits.values()) if limits else 0.15
    from .data.universe import load_universe
    uni = load_universe()
    ex_map = (dict(zip(uni["symbol"].astype(str), uni["exchange"]))
              if uni is not None and len(uni) else {})
    band = (df["symbol"].astype(str).map(ex_map).map(limits)
            .astype(float).fillna(widest))
    return band + _CORP_ACTION_MARGIN


def corporate_action_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series aligned to df.index: True where the row is NOT polluted by
    a recent corporate action.

    ``max_abs_ret_20`` carries the worst 1-day move over the feature window; a
    row is dropped while that worst move exceeds its exchange band + margin (see
    :func:`_row_band_threshold`) — i.e. a band-breaking corporate action sits
    inside the look-back window and has poisoned mom_*/atr_14/rsi_14. Frames
    lacking the support column are left untouched."""
    if "max_abs_ret_20" not in df.columns or "symbol" not in df.columns:
        return pd.Series(True, index=df.index)
    artifact = df["max_abs_ret_20"].astype(float) > _row_band_threshold(df)
    return ~artifact.fillna(False)


def band_break_flags(df: pd.DataFrame) -> pd.Series:
    """Per-row boolean: True where THAT bar's own 1-day move broke the price band
    — i.e. a corporate action happened on that bar. Used to detect contamination
    in a forward (target) window, where a single break makes the realized
    forward return a fake move. Frames lacking ``ret_1d`` are treated as clean."""
    if "ret_1d" not in df.columns or "symbol" not in df.columns:
        return pd.Series(False, index=df.index)
    return (df["ret_1d"].abs() > _row_band_threshold(df)).fillna(False)
