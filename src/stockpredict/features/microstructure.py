"""Liquidity / microstructure features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config


def adv_vnd(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """20-day average daily traded value (close * volume), in VND."""
    return (df["close"] * df["volume"]).rolling(window).mean()


def active_days_above(df: pd.DataFrame, threshold: float,
                      window: int = 20) -> pd.Series:
    """Rolling count of days whose traded value (close * volume) clears
    ``threshold``.

    Robust to single volume spikes that fool a mean: a stock must trade
    actively on MANY days, not just look liquid because one block-trade day
    lifts the average. Used by the liquidity gate.
    """
    val = df["close"] * df["volume"]
    return (val >= threshold).rolling(window).sum()


def active_days_calendar(panel: pd.DataFrame, threshold: float,
                         window: int = 20) -> np.ndarray:
    """Calendar-aware rolling count of active days per symbol.

    Like :func:`active_days_above`, but counts over the trailing ``window``
    *market* trading days rather than the trailing ``window`` *rows* the symbol
    happens to have. Days the symbol did not trade (but the market was open) are
    filled as zero volume → inactive, so a stock that trades only sporadically
    can't pass by stringing together old active days.

    ``panel`` is a long frame indexed by date with columns
    ``[symbol, close, volume]`` (dates repeat across symbols). The market
    calendar is the set of dates on which a *quorum* of symbols traded — a
    plain union of all dates would be polluted by weekend/glitch rows where only
    a handful of symbols print, which would wrongly mark blue chips inactive on
    those days. A date counts as a real session only if at least
    ``0.25 * busiest_day`` symbols traded it (this cleanly separates
    ~900-symbol sessions from the 2–10-symbol weekend/glitch rows). Returns a
    float ``ndarray`` aligned to ``panel`` row order (NaN until the window fills).
    """
    per_date = panel.index.value_counts()
    quorum = max(1, int(0.25 * int(per_date.max())))
    cal = per_date[per_date >= quorum].index.sort_values()
    dates = panel.index
    syms = panel["symbol"].to_numpy()
    val = (panel["close"] * panel["volume"]).to_numpy(dtype=float)
    result = np.full(len(panel), np.nan)
    for sym in pd.unique(syms):
        mask = syms == sym
        s = pd.Series(val[mask], index=dates[mask])
        s = s[~s.index.duplicated(keep="last")]
        full = s.reindex(cal, fill_value=0.0)
        active = (full >= threshold).rolling(window).sum()
        result[mask] = active.reindex(dates[mask]).to_numpy()
    return result


def intraday_range(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Average normalized intraday range as a slippage proxy."""
    rng = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    return rng.rolling(window).mean()


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["adv_vnd_20"] = adv_vnd(df, 20)
    out["range_20"] = intraday_range(df, 20)
    # Gate-support column (not a model feature): how many of the last 20 days
    # actually traded >= the per-day liquidity bar. Row-based fallback for
    # single-symbol callers; build_panel overwrites it with the calendar-aware
    # count (active_days_calendar) once the full set of market dates is known.
    min_adv = load_config().universe["liquidity_filter"]["min_adv_vnd"]
    out["adv_active_days_20"] = active_days_above(df, min_adv, 20)
    # Gate-support columns (not model features) for the ceiling-lock filter:
    # the 1-day close-to-close return and whether the bar closed at its high.
    # A bar that closed at the high with a near-limit gain is locked limit-up
    # (no sellers) — the next session opens unbuyable.
    out["ret_1d"] = df["close"].pct_change()
    out["close_at_high"] = df["close"] >= df["high"]
    # Gate-support column (not a model feature) for the corporate-action filter:
    # the worst (largest-magnitude) close-to-close move over the last
    # ``corp_action_lookback`` bars. Vietnamese exchanges cap a single day's
    # legal move at the price band (HOSE 7% / HNX 10% / UPCOM 15%), so any
    # |ret_1d| beyond the band is physically impossible without a corporate
    # action (split, rights issue, big special dividend). When such a bar sits
    # inside the feature window it poisons mom_*/atr_14/rsi_14, so the filter
    # drops the ticker until the artifact ages out. We carry the MAX here and
    # let the mask apply the per-exchange band (the lookback matches the longest
    # feature window, mom_20, so contamination is fully covered).
    lookback = int(load_config().pricing.get("corp_action_lookback", 20))
    out["max_abs_ret_20"] = (
        out["ret_1d"].abs().rolling(lookback, min_periods=1).max()
    )
    return out
