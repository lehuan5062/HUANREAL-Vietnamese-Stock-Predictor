"""Verify the exact-N selection in rank_today (and base.run's payload).

rank_today(n_picks=N) ranks the WHOLE scored universe by pred_mean desc and
returns exactly the top N (fewer only when the eligible universe is smaller).
The glitch filter still drops implausible forecasts before ranking. base.run
writes the top_n metadata fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredict.model import predict as predict_mod


def _fake_panel(symbols, last_date="2026-05-27"):
    """One row per symbol, all on the same date — enough for rank_today once
    liquidity_mask / add_price_suggestions are stubbed."""
    end = pd.Timestamp(last_date)
    rows = [{"symbol": s, "close": 50.0, "atr_14": 1.0} for s in symbols]
    df = pd.DataFrame(rows, index=[end] * len(rows))
    df.index.name = "date"
    return df


class _FakeMeanModel:
    """Deterministic descending pred_mean (by row order) so ranking is
    predictable; all within the glitch cap."""

    def predict(self, snap):
        n = len(snap)
        return {"pred_mean": np.linspace(0.05, 0.01, n),
                "pred_std": np.full(n, 0.001)}


def _price_stub():
    """add_price_suggestions stub that flags every row's quality as fine
    (below_breakeven=False) and adds the minimal VND columns."""
    def _stub(df):
        out = df.copy()
        out["below_breakeven"] = False
        out["entry_vnd"] = (out["close"] * 1000).round(0).astype("Int64")
        out["close_vnd"] = (out["close"] * 1000).round(0).astype("Int64")
        return out
    return _stub


def _wire(monkeypatch, panel):
    monkeypatch.setattr(predict_mod, "tradable_symbols", lambda: None)
    monkeypatch.setattr(predict_mod, "build_panel", lambda **kw: panel)
    monkeypatch.setattr(predict_mod, "liquidity_mask",
                        lambda df: pd.Series(True, index=df.index))
    monkeypatch.setattr(predict_mod, "ceiling_lock_mask",
                        lambda df: pd.Series(True, index=df.index))
    # Don't load a real low-quantile model from disk (its predict() needs the
    # full feature matrix); keep entry == close for these focused tests.
    monkeypatch.setattr(predict_mod, "_try_load_low_model", lambda: None)
    monkeypatch.setattr(predict_mod, "add_price_suggestions", _price_stub())


def test_returns_exactly_n(monkeypatch):
    """n_picks=5 over an 8-symbol universe returns exactly 5, pred_mean-desc."""
    panel = _fake_panel(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"])
    _wire(monkeypatch, panel)

    out = predict_mod.rank_today(model=_FakeMeanModel(), n_picks=5, low_model=None)
    assert len(out) == 5
    assert out["pred_mean"].is_monotonic_decreasing


def test_shortfall_returns_all_when_universe_small(monkeypatch):
    """n_picks=5 but only 3 eligible symbols -> all 3 returned, no crash."""
    panel = _fake_panel(["AAA", "BBB", "CCC"])
    _wire(monkeypatch, panel)

    out = predict_mod.rank_today(model=_FakeMeanModel(), n_picks=5, low_model=None)
    assert len(out) == 3
    assert set(out["symbol"].astype(str)) == {"AAA", "BBB", "CCC"}


class _GlitchMeanModel:
    """First symbol gets an implausible forecast (split/corp-action artifact);
    the rest are normal. Used to verify the max_abs_pred_mean sanity filter."""

    def predict(self, snap):
        # +50% in 2 days for GLITCH (a data artifact, must be dropped); a
        # normal +2% for everyone else. Keyed by symbol so it's robust to the
        # cross-section row order.
        sym = snap["symbol"].astype(str)
        preds = np.where(sym.values == "GLITCH", 0.50, 0.02)
        return {"pred_mean": preds, "pred_std": np.full(len(snap), 0.001)}


def test_glitch_pred_mean_is_filtered_before_ranking(monkeypatch):
    """A row whose |pred_mean| exceeds pricing.max_abs_pred_mean (default 0.05)
    is dropped before pricing/ranking, so a data artifact can't become a pick."""
    panel = _fake_panel(["GLITCH", "AAA", "BBB"])
    _wire(monkeypatch, panel)

    out = predict_mod.rank_today(model=_GlitchMeanModel(), n_picks=10, low_model=None)
    assert "GLITCH" not in set(out["symbol"].astype(str))
    assert set(out["symbol"].astype(str)) == {"AAA", "BBB"}


def test_base_run_payload_top_n(monkeypatch, tmp_path):
    """base.run writes the top_n metadata: selection, requested_picks, n_picks,
    n_below_breakeven — computed from the below_breakeven column."""
    from stockpredict.modes import base

    stub = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC"],
        "below_breakeven": [False, False, True],
    })
    monkeypatch.setattr(base, "rank_today", lambda **kw: stub)
    monkeypatch.setattr(base, "reports_dir", lambda: tmp_path)

    picks, out = base.run(on="2026-05-27", n_picks=3)

    assert len(picks) == 3
    payload = json.loads(Path(out).read_text(encoding="utf-8"))
    assert payload["selection"] == "top_n"
    assert payload["requested_picks"] == 3
    assert payload["n_picks"] == 3
    assert payload["n_below_breakeven"] == 1
    assert "n_actionable" not in payload
