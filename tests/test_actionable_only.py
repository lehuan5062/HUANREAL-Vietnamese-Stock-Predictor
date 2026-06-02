"""Verify the actionable-only scan in rank_today (and base.run's empty report).

rank_today(actionable_only=True) must price the WHOLE scored universe, keep
only rows that clear the `actionable` gate, sort by pred_mean desc, and cap at
`max_picks` (falling back to config.report.max_picks). The legacy `top_k` path
must be untouched. base.run must write a valid (empty) report when nothing is
actionable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stockpredict.config import load_config
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
    def _stub(df, units=None, budget_vnd=None):
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


def test_actionable_only_filters_and_caps(monkeypatch):
    """Only actionable rows survive, capped at max_picks, sorted by pred_mean."""
    panel = _fake_panel(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"])
    _wire(monkeypatch, panel, _price_stub({"AAA", "CCC", "EEE"}))

    out = predict_mod.rank_today(model=_FakeMeanModel(), actionable_only=True,
                                 max_picks=2, low_model=None)

    assert len(out) == 2
    assert out["actionable"].all()
    assert set(out["symbol"].astype(str)).issubset({"AAA", "CCC", "EEE"})
    # Highest-conviction actionable picks first.
    assert out["pred_mean"].is_monotonic_decreasing


def test_actionable_only_empty_when_none_actionable(monkeypatch):
    """Zero actionable rows -> empty frame, no crash (the typical T+2 case)."""
    panel = _fake_panel(["AAA", "BBB", "CCC"])
    _wire(monkeypatch, panel, _price_stub(set()))  # nothing actionable

    out = predict_mod.rank_today(model=_FakeMeanModel(), actionable_only=True,
                                 low_model=None)
    assert len(out) == 0
    assert "symbol" in out.columns  # schema preserved


def test_actionable_only_falls_back_to_config_ceiling(monkeypatch):
    """max_picks=None -> cap comes from config.report.max_picks."""
    panel = _fake_panel(["AAA", "BBB", "CCC", "DDD", "EEE"])
    _wire(monkeypatch, panel, _price_stub(None))  # all five actionable
    monkeypatch.setattr(predict_mod, "load_config",
                        lambda: {"report": {"max_picks": 3}})

    out = predict_mod.rank_today(model=_FakeMeanModel(), actionable_only=True,
                                 low_model=None)
    assert len(out) == 3
    assert out["pred_mean"].is_monotonic_decreasing


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
    expected_cap = int(load_config().get("report", {}).get("max_picks", 20))
    assert payload["max_picks"] == expected_cap
