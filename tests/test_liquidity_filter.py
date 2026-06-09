"""Tests for the active-days liquidity gate.

The gate counts how many of the last 20 days actually traded >= min_adv_vnd and
requires >= min_adv_active_days of them, so a single block-trade spike can't
sneak a mostly-dead stock (e.g. VMS) through a mean-ADV test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredict.features.microstructure import (
    active_days_above,
    active_days_calendar,
    add_all,
)
from stockpredict.filters import liquidity_mask


def _frame(values_mvnd, close=20.0):
    """Build a 20-row OHLCV frame whose daily traded value (close*volume, in
    thousand-VND-shares) matches `values_mvnd` (a list of per-day values in the
    same kVND-shares unit min_adv_vnd uses)."""
    idx = pd.date_range("2026-01-01", periods=len(values_mvnd), freq="B")
    vol = [v / close for v in values_mvnd]
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": vol},
        index=idx,
    )


def test_active_days_counts_days_over_threshold():
    """Only days at/above the threshold are counted in the trailing window."""
    vals = [2_000_000] * 12 + [10] * 8   # 12 active, 8 dead
    s = active_days_above(_frame(vals), threshold=1_000_000, window=20)
    assert s.iloc[-1] == 12


def test_spike_fooled_stock_is_rejected():
    """One huge block day + 19 dead days: mean would pass, active-days fails.
    This is the VMS failure mode."""
    vals = [50_000_000] + [50_000] * 19   # mean ~2.5M (>1M) but only 1 active day
    df = add_all(_frame(vals))
    mask = liquidity_mask(df)
    assert bool(mask.iloc[-1]) is False
    # Sanity: the spike really does inflate the mean over the bar.
    assert df["adv_vnd_20"].iloc[-1] >= 1_000_000


def test_consistently_traded_stock_passes():
    """Clears the bar on >= 15 of 20 days → passes."""
    vals = [2_000_000] * 16 + [10] * 4
    df = add_all(_frame(vals))
    assert bool(liquidity_mask(df).iloc[-1]) is True


def test_borderline_below_active_days_is_rejected():
    """Clears the bar on only 14 of 20 days → just misses the 15-day rule."""
    vals = [2_000_000] * 14 + [10] * 6
    df = add_all(_frame(vals))
    assert bool(liquidity_mask(df).iloc[-1]) is False


def test_penny_stock_rejected_even_if_active():
    """min_close still applies regardless of activity."""
    vals = [2_000_000] * 20
    df = add_all(_frame(vals, close=3.0))   # 3,000 VND < min_close 5
    assert bool(liquidity_mask(df).iloc[-1]) is False


# ---------------------------------------------------------------------------
# Calendar-aware counting
# ---------------------------------------------------------------------------

def test_calendar_aware_penalizes_gappy_stock():
    """A stock that trades only every other market day should count the skipped
    days as inactive — even though, counted over its own rows alone, it looks
    fully active. The market calendar is set by a symbol that trades daily."""
    cal = pd.date_range("2026-01-01", periods=40, freq="B")
    vol = 2_000_000 / 20.0
    # M trades every market day -> defines the 40-day calendar.
    M = pd.DataFrame({"symbol": "M", "close": 20.0, "volume": vol}, index=cal)
    # G trades every other day: 20 active rows spanning 40 calendar days.
    G = pd.DataFrame({"symbol": "G", "close": 20.0, "volume": vol}, index=cal[::2])
    panel = pd.concat([M, G]).sort_index()

    # Row-based view of G alone would say all 20 of its rows are active.
    assert active_days_above(G, 1_000_000, 20).iloc[-1] == 20

    res = active_days_calendar(panel, threshold=1_000_000, window=20)
    panel = panel.assign(adv=res)
    # M: active on every one of the trailing 20 market days.
    assert panel[panel["symbol"] == "M"]["adv"].iloc[-1] == 20
    # G: only 10 of the trailing 20 market days were trading days for G.
    assert panel[panel["symbol"] == "G"]["adv"].iloc[-1] == 10


def test_calendar_ignores_junk_dates_without_quorum():
    """A weekend/glitch date where only a couple of symbols print must NOT be
    treated as a market session — otherwise a stock that trades every real day
    gets wrongly marked inactive on the junk date."""
    cal = pd.date_range("2026-01-01", periods=30, freq="B")
    vol = 2_000_000 / 20.0
    # 60 symbols trade every real session (sets the quorum bar high).
    frames = [pd.DataFrame({"symbol": f"S{i}", "close": 20.0, "volume": vol},
                           index=cal) for i in range(60)]
    # Two junk rows on a Sunday — only these 2 symbols print that day.
    junk = pd.Timestamp("2026-01-11")  # a Sunday, not in the business-day cal
    frames.append(pd.DataFrame({"symbol": ["S0", "S1"], "close": [20.0, 20.0],
                                "volume": [vol, vol]},
                               index=pd.DatetimeIndex([junk, junk])))
    panel = pd.concat(frames).sort_index()
    res = active_days_calendar(panel, threshold=1_000_000, window=20)
    panel = panel.assign(adv=res)
    # S2 (never prints on the junk day) is still 20/20 — the junk date was
    # excluded from the calendar by the quorum, so it didn't count against it.
    assert panel[panel["symbol"] == "S2"]["adv"].iloc[-1] == 20
