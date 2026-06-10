"""Tests for the ceiling-lock (limit-up) filter.

A stock that closed at the daily ceiling opens the next session locked with a
buy queue and no sellers, so a limit-buy can't fill. ceiling_lock_mask drops it
from the pickable universe (latest-day only).
"""
from __future__ import annotations

import pandas as pd

from stockpredict import filters
from stockpredict.features.microstructure import add_all


def _bar(symbol, prev_close, close, high):
    """A 2-row OHLCV frame so ret_1d/close_at_high are defined on the last bar."""
    idx = pd.date_range("2026-06-01", periods=2, freq="B")
    return pd.DataFrame(
        {"open": [prev_close, close], "high": [prev_close, high],
         "low": [prev_close, close], "close": [prev_close, close],
         "volume": [100000, 100000]},
        index=idx,
    ).assign(symbol=symbol)


def _patch_universe(monkeypatch, mapping):
    uni = pd.DataFrame({"symbol": list(mapping), "exchange": list(mapping.values())})
    monkeypatch.setattr(filters, "load_universe", lambda: uni, raising=False)
    import stockpredict.data.universe as u
    monkeypatch.setattr(u, "load_universe", lambda: uni)
    filters._ceiling_params.cache_clear()


def _mask_last(df):
    return bool(filters.ceiling_lock_mask(df).iloc[-1])


def test_hnx_ceiling_lock_excluded(monkeypatch):
    """HNX band 10%: +9.3% close-at-high (tick-rounded ceiling) is locked."""
    _patch_universe(monkeypatch, {"DST": "HNX"})
    df = add_all(_bar("DST", 10.8, 11.8, 11.8))   # +9.26%, close==high
    assert _mask_last(df) is False                 # locked -> excluded


def test_strong_up_day_not_at_ceiling_kept(monkeypatch):
    """HSX band 7%: a +4% day that closed at high is NOT a ceiling-lock."""
    _patch_universe(monkeypatch, {"AAA": "HSX"})
    df = add_all(_bar("AAA", 20.0, 20.8, 20.8))   # +4%, close==high but far from 7%
    assert _mask_last(df) is True                  # buyable


def test_close_below_high_not_locked(monkeypatch):
    """Even a near-limit gain isn't a lock if it didn't close at the high
    (there were sellers below the top)."""
    _patch_universe(monkeypatch, {"BBB": "HNX"})
    df = add_all(_bar("BBB", 10.0, 10.9, 11.2))   # +9% but close 10.9 < high 11.2
    assert _mask_last(df) is True


def test_unknown_exchange_kept(monkeypatch):
    """No exchange info -> can't determine the band -> leave it buyable."""
    _patch_universe(monkeypatch, {"OTHER": "HSX"})   # ZZZ not in map
    df = add_all(_bar("ZZZ", 10.0, 11.0, 11.0))
    assert _mask_last(df) is True


def test_upcom_wider_band(monkeypatch):
    """UPCOM band 15%: a +9% close-at-high is well short of the ceiling -> kept;
    a +14% close-at-high is locked."""
    _patch_universe(monkeypatch, {"U1": "UPCOM", "U2": "UPCOM"})
    kept = add_all(_bar("U1", 10.0, 10.9, 10.9))    # +9% < 15%-tol
    locked = add_all(_bar("U2", 10.0, 11.4, 11.4))  # +14% within tol of 15%
    assert _mask_last(kept) is True
    assert _mask_last(locked) is False
