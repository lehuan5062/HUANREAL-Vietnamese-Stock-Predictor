"""Mechanical gates applied before handing the universe to the LLM agent.

Per the LLM-agent-only architecture, only TRUE mechanical gates live here —
things that make a name literally unbuyable or its data literally unusable.
Judgment thresholds (liquidity size, overbought RSI, downtrend shape) are no
longer gates: their underlying columns (``adv_vnd_20``, ``adv_active_days_20``,
``close``, ``rsi_14``, ``history_days``) are simply included as plain data in
the universe frame handed to the LLM agent, which reasons over them itself.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .config import load_config


def staleness_mask(df: pd.DataFrame, on: pd.Timestamp) -> pd.Series:
    """Boolean Series aligned to df.index: True where the row's bar date (the
    frame's index) is no more than ``max_staleness_days`` business days before
    ``on``. Expects a latest-cross-section frame (one row per symbol, date
    index).

    Data-integrity gate, not a strategy knob: a dormant name whose cache ended
    months ago would otherwise be scored on a stale close, and that close
    silently recorded as the entry price. ``0`` (or a missing key) disables the
    gate (all True)."""
    import numpy as np
    max_stale = int(load_config().universe["liquidity_filter"]
                    .get("max_staleness_days", 0) or 0)
    if max_stale <= 0:
        return pd.Series(True, index=df.index)
    dates = pd.to_datetime(df.index).values.astype("datetime64[D]")
    ref = np.datetime64(pd.Timestamp(on).normalize(), "D")
    age = np.busday_count(dates, ref)
    return pd.Series(age <= max_stale, index=df.index)


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
