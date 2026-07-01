"""Verify the eligible-universe filter drops symbols missing from
tradable_symbols() (defense-in-depth that catches delisted tickers like HTK),
and degrades gracefully when the universe parquet is missing."""
from __future__ import annotations

import pandas as pd

from stockpredict.model import predict as predict_mod


def _downtrend_panel(symbols, last_date="2026-05-27"):
    """Minimal in-memory panel of downtrend rows that clear the eligible-universe
    filter cascade (liquidity + downtrend). One row per symbol, same date."""
    rows = []
    end = pd.Timestamp(last_date)
    for sym in symbols:
        rows.append({
            "symbol": sym,
            "close": 50.0,
            "adv_vnd_20": 5_000_000.0,
            "adv_active_days_20": 20.0,   # clears the liquidity gate
            "rsi_14": 40.0,               # in the downtrend RSI band
            "mom_5": -0.02, "mom_20": -0.08,   # 20-day decline
            "high_prox_20": -0.08,        # >5% below the 20-day high
            "vol_z_20": 0.0, "atr_14": 1.0,
        })
    df = pd.DataFrame(rows, index=[end] * len(rows))
    df.index.name = "date"
    return df


def test_eligible_drops_non_tradable_symbols(monkeypatch):
    """A cached symbol not in tradable_symbols() must be dropped (HTK leak)."""
    panel = _downtrend_panel(["VCB", "FPT", "HTK", "AGX"])
    monkeypatch.setattr(predict_mod, "tradable_symbols",
                        lambda: {"VCB", "FPT", "AGX"})
    snap = predict_mod.eligible_universe(panel=panel)
    syms = set(snap["symbol"].astype(str))
    assert "HTK" not in syms
    assert {"VCB", "FPT", "AGX"}.issubset(syms)


def test_eligible_skips_filter_when_universe_missing(monkeypatch):
    """Cold start: tradable_symbols() is None → don't wipe the cross-section."""
    panel = _downtrend_panel(["VCB", "FPT", "HTK"])
    monkeypatch.setattr(predict_mod, "tradable_symbols", lambda: None)
    snap = predict_mod.eligible_universe(panel=panel)
    assert set(snap["symbol"].astype(str)) == {"VCB", "FPT", "HTK"}
