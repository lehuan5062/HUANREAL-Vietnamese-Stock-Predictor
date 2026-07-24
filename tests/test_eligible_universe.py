"""Verify selector.eligible_universe drops symbols missing from
tradable_symbols() (defense-in-depth that catches delisted tickers like HTK),
degrades gracefully when the universe parquet is missing, and includes the
plain liquidity/technical reference columns the LLM agent now reasons over
directly (no more coded liquidity/overbought/downtrend gates)."""
from __future__ import annotations

import pandas as pd

from stockpredict import selector


def _panel(symbols, last_date="2026-05-27"):
    """Minimal in-memory panel that clears the mechanical gates (staleness /
    ceiling-lock / corporate-action)."""
    rows = []
    end = pd.Timestamp(last_date)
    for sym in symbols:
        rows.append({
            "symbol": sym,
            "close": 50.0,
            "adv_vnd_20": 5_000_000.0,
            "adv_active_days_20": 20.0,
            "rsi_14": 40.0,
            "mom_5": -0.02, "mom_20": -0.08,
            "high_prox_20": -0.08,
            "vol_z_20": 0.0, "atr_14": 1.0,
        })
    df = pd.DataFrame(rows, index=[end] * len(rows))
    df.index.name = "date"
    return df


def test_eligible_drops_non_tradable_symbols(monkeypatch):
    """A cached symbol not in tradable_symbols() must be dropped (HTK leak)."""
    panel = _panel(["VCB", "FPT", "HTK", "AGX"])
    monkeypatch.setattr(selector, "tradable_symbols",
                        lambda: {"VCB", "FPT", "AGX"})
    snap = selector.eligible_universe(panel=panel)
    syms = set(snap["symbol"].astype(str))
    assert "HTK" not in syms
    assert {"VCB", "FPT", "AGX"}.issubset(syms)


def test_eligible_skips_filter_when_universe_missing(monkeypatch):
    """Cold start: tradable_symbols() is None → don't wipe the cross-section."""
    panel = _panel(["VCB", "FPT", "HTK"])
    monkeypatch.setattr(selector, "tradable_symbols", lambda: None)
    snap = selector.eligible_universe(panel=panel)
    assert set(snap["symbol"].astype(str)) == {"VCB", "FPT", "HTK"}


def test_eligible_universe_carries_plain_reference_columns(monkeypatch):
    """No coded liquidity/overbought/downtrend gate any more — the underlying
    columns just ride through as plain data for the agent."""
    panel = _panel(["VCB"])
    monkeypatch.setattr(selector, "tradable_symbols", lambda: None)
    snap = selector.eligible_universe(panel=panel)
    assert len(snap) == 1
    for col in ("adv_vnd_20", "adv_active_days_20", "rsi_14",
               "mom_5", "mom_20", "high_prox_20", "history_days"):
        assert col in snap.columns
