"""Verify the actionable-only scan in rank_today (and base.run's empty report).

rank_today(actionable_only=True) must price the WHOLE scored universe and
return every row that clears the `actionable` gate, sorted by pred_mean desc,
with no cap. The legacy `top_k` path must be untouched. base.run must write a
valid (empty) report when nothing is actionable.
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
    """Deterministic descending pred_mean so ranking is predictable."""

    def predict(self, snap):
        n = len(snap)
        return {"pred_mean": np.linspace(0.05, 0.01, n),
                "pred_std": np.full(n, 0.001)}


def _price_stub(actionable_syms):
    """add_price_suggestions stub that marks a row `actionable` iff its symbol
    is in `actionable_syms` (None => every row actionable)."""
    def _stub(df):
        out = df.copy()
        sym = out["symbol"].astype(str)
        mask = pd.Series(True, index=out.index) if actionable_syms is None \
            else sym.isin(set(actionable_syms))
        out["actionable"] = mask.values
        out["entry_vnd"] = (out["close"] * 1000).round(0).astype("Int64")
        out["close_vnd"] = (out["close"] * 1000).round(0).astype("Int64")
        return out
    return _stub


def _wire(monkeypatch, panel, price_stub):
    monkeypatch.setattr(predict_mod, "tradable_symbols", lambda: None)
    monkeypatch.setattr(predict_mod, "build_panel", lambda **kw: panel)
    monkeypatch.setattr(predict_mod, "liquidity_mask",
                        lambda df: pd.Series(True, index=df.index))
    # Don't load a real low-quantile model from disk (its predict() needs the
    # full feature matrix); keep entry == close for these focused tests.
    monkeypatch.setattr(predict_mod, "_try_load_low_model", lambda: None)
    monkeypatch.setattr(predict_mod, "add_price_suggestions", price_stub)


def test_actionable_only_empty_when_none_actionable(monkeypatch):
    """Zero actionable rows -> empty frame, no crash (the typical T+2 case)."""
    panel = _fake_panel(["AAA", "BBB", "CCC"])
    _wire(monkeypatch, panel, _price_stub(set()))  # nothing actionable

    out = predict_mod.rank_today(model=_FakeMeanModel(), actionable_only=True,
                                 low_model=None)
    assert len(out) == 0
    assert "symbol" in out.columns  # schema preserved


def test_actionable_only_lists_all_actionable(monkeypatch):
    """actionable_only lists EVERY actionable row (no cap), sorted by pred_mean."""
    panel = _fake_panel(["AAA", "BBB", "CCC", "DDD", "EEE"])
    _wire(monkeypatch, panel, _price_stub({"AAA", "CCC", "EEE"}))  # 3 of 5 actionable

    out = predict_mod.rank_today(model=_FakeMeanModel(), actionable_only=True,
                                 low_model=None)
    assert len(out) == 3  # every actionable row, nothing capped
    assert set(out["symbol"].astype(str)) == {"AAA", "CCC", "EEE"}
    assert out["actionable"].all()
    assert out["pred_mean"].is_monotonic_decreasing


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
    is dropped before pricing/ranking, so a data artifact can't become the top
    actionable pick."""
    panel = _fake_panel(["GLITCH", "AAA", "BBB"])
    _wire(monkeypatch, panel, _price_stub(None))  # everything else actionable

    out = predict_mod.rank_today(model=_GlitchMeanModel(), actionable_only=True,
                                 low_model=None)
    assert "GLITCH" not in set(out["symbol"].astype(str))
    assert set(out["symbol"].astype(str)) == {"AAA", "BBB"}


def test_legacy_top_k_path_ignores_actionable(monkeypatch):
    """actionable_only=False keeps the legacy cut-to-top_k behavior, returning
    rows regardless of the actionable flag."""
    panel = _fake_panel(["AAA", "BBB", "CCC", "DDD"])
    _wire(monkeypatch, panel, _price_stub(set()))  # none actionable

    out = predict_mod.rank_today(model=_FakeMeanModel(), top_k=2, low_model=None)
    assert len(out) == 2  # not filtered by actionable
    assert not out["actionable"].any()


def test_base_run_writes_empty_report(monkeypatch, tmp_path):
    """When nothing is actionable, base.run still writes a valid report with
    an empty picks list and the actionable-only metadata fields."""
    from stockpredict.modes import base

    monkeypatch.setattr(base, "rank_today", lambda **kw: pd.DataFrame())
    monkeypatch.setattr(base, "reports_dir", lambda: tmp_path)

    picks, out = base.run(on="2026-05-27")

    assert len(picks) == 0
    payload = json.loads(Path(out).read_text(encoding="utf-8"))
    assert payload["selection"] == "actionable_only"
    assert payload["n_actionable"] == 0
    assert payload["picks"] == []
