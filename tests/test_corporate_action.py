"""Tests for the corporate-action filter.

Vietnamese exchanges cap a single session's close-to-close move at the price
band (HOSE 7% / HNX 10% / UPCOM 15%). Any |ret_1d| beyond the band is
physically impossible without a corporate action (split / rights / special
dividend) — the unadjusted feed shows a phantom crash/spike that poisons the
momentum / ATR / RSI features. corporate_action_mask drops such a ticker while
the band-breaking bar sits inside the feature window.
"""
from __future__ import annotations

import pandas as pd

from stockpredict import filters
from stockpredict.features.microstructure import add_all


def _bar(symbol, prev_close, close):
    """A 2-row OHLCV frame so ret_1d / max_abs_ret_20 are defined on the last bar."""
    idx = pd.date_range("2026-06-01", periods=2, freq="B")
    return pd.DataFrame(
        {"open": [prev_close, close], "high": [prev_close, max(prev_close, close)],
         "low": [prev_close, min(prev_close, close)], "close": [prev_close, close],
         "volume": [100000, 100000]},
        index=idx,
    ).assign(symbol=symbol)


def _series(symbol, closes):
    """An N-row frame from a close series (open=high=low=close, flat range)."""
    idx = pd.date_range("2026-05-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [100000] * len(closes)},
        index=idx,
    ).assign(symbol=symbol)


def _patch_universe(monkeypatch, mapping):
    uni = pd.DataFrame({"symbol": list(mapping), "exchange": list(mapping.values())})
    monkeypatch.setattr(filters, "load_universe", lambda: uni, raising=False)
    import stockpredict.data.universe as u
    monkeypatch.setattr(u, "load_universe", lambda: uni)
    filters._ceiling_params.cache_clear()


def _mask_last(df):
    return bool(filters.corporate_action_mask(df).iloc[-1])


def test_ex_rights_gap_excluded(monkeypatch):
    """HOSE: a -38% ex-rights gap (VVS-style) is 5x the 7% band -> dropped."""
    _patch_universe(monkeypatch, {"VVS": "HSX"})
    df = add_all(_bar("VVS", 116.8, 72.3))   # -38%, impossible on any exchange
    assert _mask_last(df) is False


def test_normal_move_kept(monkeypatch):
    """HOSE: a +5% day is well within the band -> kept."""
    _patch_universe(monkeypatch, {"AAA": "HSX"})
    df = add_all(_bar("AAA", 20.0, 21.0))    # +5%
    assert _mask_last(df) is True


def test_legit_limit_day_not_flagged(monkeypatch):
    """HOSE: an exact +7% limit-up day sits at the band, within the margin -> kept."""
    _patch_universe(monkeypatch, {"LIM": "HSX"})
    df = add_all(_bar("LIM", 10.0, 10.7))    # +7%, a legal ceiling day
    assert _mask_last(df) is True


def test_upcom_wider_band(monkeypatch):
    """UPCOM band 15%: a +14% move is legal -> kept; a -40% gap -> dropped."""
    _patch_universe(monkeypatch, {"U1": "UPCOM", "U2": "UPCOM"})
    kept = add_all(_bar("U1", 10.0, 11.4))     # +14% < 15%
    gap = add_all(_bar("U2", 10.0, 6.0))       # -40% corporate action
    assert _mask_last(kept) is True
    assert _mask_last(gap) is False


def test_unknown_exchange_uses_widest_band(monkeypatch):
    """No exchange info -> widest band (UPCOM 15%): only moves impossible on
    EVERY exchange flag it. A -38% gap still drops."""
    _patch_universe(monkeypatch, {"OTHER": "HSX"})   # ZZZ absent from the map
    df = add_all(_bar("ZZZ", 100.0, 62.0))           # -38%
    assert _mask_last(df) is False


def test_artifact_ages_out_of_window(monkeypatch):
    """Once the band-breaking bar falls outside the lookback window, the
    features are clean again and the ticker returns to the universe."""
    _patch_universe(monkeypatch, {"OLD": "HSX"})
    # One -38% gap, then 25 calm bars (> the 20-bar lookback) at the new level.
    closes = [116.8, 72.3] + [72.0 + 0.1 * i for i in range(25)]
    df = add_all(_series("OLD", closes))
    assert _mask_last(df) is True
