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


def overbought_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series aligned to df.index: True where the row is NOT overbought.

    Excludes candidates whose ``rsi_14`` exceeds ``pricing.overbought_rsi_max``.
    An overbought blow-off (price run too far) tends to reverse, so buying the
    top is a poor T+2 entry — historically RSI>80 names win far less often. The
    knob ``0`` (or a missing ``rsi_14`` column) disables the gate (all True).
    This is the exhaustion guard; the liquidity ``min_adv_active_days`` filter is
    a separate volume-spike/tradability guard."""
    level = float(load_config().pricing.get("overbought_rsi_max", 0) or 0)
    if level <= 0 or "rsi_14" not in df.columns:
        return pd.Series(True, index=df.index)
    overbought = df["rsi_14"].astype(float) > level
    return ~overbought.fillna(False)


def downtrend_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean Series aligned to df.index: True where the row is in a downtrend
    (i.e. a rebound candidate).

    The rebound strategy only bets on names that have pulled back, so this gate
    keeps rows that are BOTH trending down over the medium term and sitting a
    meaningful distance below their recent high, within an RSI band that
    excludes free-falling knives (too low) and already-recovered names (too
    high). All thresholds come from ``strategy.downtrend`` in config:

        mom_20        < mom20_max        (log-return; 0 => a 20-day decline)
        high_prox_20 <= high_prox_max    (fraction below the 20-day high, e.g. -0.05)
        rsi_14       >= rsi_floor         (0 disables this leg)
        rsi_14       <= rsi_ceil

    A missing column, or a disabled leg, is treated as passing (True) so the
    gate degrades gracefully. When the ``strategy.downtrend`` block is absent
    entirely the gate is a no-op (all True), preserving legacy behaviour."""
    cfg = load_config()
    strat = dict(getattr(cfg, "strategy", {}) or {})
    dt = dict(strat.get("downtrend", {}) or {})
    if not dt:
        return pd.Series(True, index=df.index)

    cond = pd.Series(True, index=df.index)
    if "mom_20" in df.columns and dt.get("mom20_max") is not None:
        cond &= df["mom_20"].astype(float) < float(dt["mom20_max"])
    if "high_prox_20" in df.columns and dt.get("high_prox_max") is not None:
        cond &= df["high_prox_20"].astype(float) <= float(dt["high_prox_max"])
    if "rsi_14" in df.columns:
        floor = float(dt.get("rsi_floor", 0) or 0)
        if floor > 0:
            cond &= df["rsi_14"].astype(float) >= floor
        ceil = dt.get("rsi_ceil")
        if ceil is not None:
            cond &= df["rsi_14"].astype(float) <= float(ceil)
    return cond.fillna(False)


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


