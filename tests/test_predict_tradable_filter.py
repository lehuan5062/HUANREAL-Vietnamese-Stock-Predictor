"""Verify rank_today drops symbols missing from tradable_symbols() — even
when the caller passes symbols=None (the default predict / base.run path).
This is the defense-in-depth that catches delisted tickers like HTK after
the cli forgot to wire pred_syms through to base.run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredict.model import predict as predict_mod


def _fake_panel(symbols, last_date="2026-05-27"):
    """Build a minimal in-memory panel that satisfies rank_today's column
    expectations. One row per symbol, all on the same date."""
    rows = []
    end = pd.Timestamp(last_date)
    for sym in symbols:
        rows.append({
            "symbol": sym,
            # liquidity / features the downstream pipeline reads
            "close": 50.0,
            "adv_vnd_20": 5_000_000.0,  # well above min_adv_vnd
            "rsi_14": 50.0, "mom_5": 0.01, "mom_20": 0.02,
            "vol_z_20": 0.0, "atr_14": 1.0,
            "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
            "high_prox_20": 0.0, "gap": 0.0, "realvol_20": 0.02,
            "range_20": 0.02,
            "target": 0.01,
        })
    df = pd.DataFrame(rows, index=[end] * len(rows))
    df.index.name = "date"
    return df


class _FakeMeanModel:
    """Stub TrainedModel — returns deterministic predictions so the test
    doesn't need a trained pickle on disk."""

    def predict(self, snap):
        # Highest pred_mean = first symbol (deterministic ranking).
        n = len(snap)
        return {"pred_mean": np.linspace(0.05, 0.01, n),
                "pred_std":  np.full(n, 0.001)}


def test_rank_today_drops_non_tradable_symbols(monkeypatch, tmp_path):
    """rank_today must skip a cached symbol that's not in tradable_symbols(),
    even with symbols=None. Mirrors the HTK leak from picks_2026-05-28."""
    panel = _fake_panel(["VCB", "FPT", "HTK", "AGX"])

    # Tradable set excludes HTK (matches the post-DELISTED-filter universe).
    monkeypatch.setattr(predict_mod, "tradable_symbols",
                        lambda: {"VCB", "FPT", "AGX"})
    # Stub build_panel so we don't hit disk.
    monkeypatch.setattr(predict_mod, "build_panel",
                        lambda **kw: panel)
    # Stub liquidity_mask + add_price_suggestions to keep the test focused
    # on the tradable filter (the upstream tests cover those separately).
    monkeypatch.setattr(predict_mod, "liquidity_mask",
                        lambda df: pd.Series(True, index=df.index))
    monkeypatch.setattr(predict_mod, "add_price_suggestions",
                        lambda df: df.assign(
                            entry_vnd=df["close"] * 1000,
                            close_vnd=df["close"] * 1000,
                            entry_limit_pct=-0.05,
                        ))

    out = predict_mod.rank_today(model=_FakeMeanModel(), n_picks=10,
                                  low_model=None)
    syms = set(out["symbol"].astype(str))
    assert "HTK" not in syms, f"HTK leaked through despite not being tradable: {out}"
    assert {"VCB", "FPT", "AGX"}.issubset(syms)


def test_rank_today_skips_filter_when_universe_missing(monkeypatch):
    """If the universe parquet is missing (cold start), tradable_symbols()
    returns None and rank_today must NOT silently drop every symbol — degrade
    to the legacy behaviour and pass the cross-section through unchanged."""
    panel = _fake_panel(["VCB", "FPT", "HTK"])

    monkeypatch.setattr(predict_mod, "tradable_symbols", lambda: None)
    monkeypatch.setattr(predict_mod, "build_panel",
                        lambda **kw: panel)
    monkeypatch.setattr(predict_mod, "liquidity_mask",
                        lambda df: pd.Series(True, index=df.index))
    monkeypatch.setattr(predict_mod, "add_price_suggestions",
                        lambda df: df.assign(
                            entry_vnd=df["close"] * 1000,
                            close_vnd=df["close"] * 1000, entry_limit_pct=-0.05,
                        ))

    out = predict_mod.rank_today(model=_FakeMeanModel(), n_picks=10,
                                  low_model=None)
    syms = set(out["symbol"].astype(str))
    # All three survive — no tradable set to filter against.
    assert syms == {"VCB", "FPT", "HTK"}
