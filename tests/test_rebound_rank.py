"""Rebound ranking + pricing integration."""
import numpy as np
import pandas as pd
import pytest

from stockpredict.model import predict as predict_mod
from stockpredict.model.train import RecoveryKMModel


def _bucket(prob, days, profit, n=100):
    return {"recovery_prob": prob, "days": days, "profit": profit, "n": n}


def _panel(symbols, realvol_20=0.5):
    # Two dates so latest_cross_section has history; all rows in a downtrend.
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    rows = []
    for s in symbols:
        for d in dates:
            rows.append({
                "date": d, "symbol": s, "close": 20.0,
                "rsi_14": 35.0, "mom_20": -0.08, "mom_5": -0.02,
                "high_prox_20": -0.08, "atr_14": 0.5, "vol_z_20": 0.0,
                "adv_vnd_20": 5_000_000.0, "adv_active_days_20": 20.0,
                "ret_1d": -0.01, "close_at_high": False, "max_abs_ret_20": 0.03,
                "realvol_20": realvol_20,
            })
    return pd.DataFrame(rows).set_index("date")


def _model(panel):
    return RecoveryKMModel(
        buckets={}, pooled=_bucket(0.9, 3.0, 0.06),
        rsi_edges=[30, 40, 50], high_prox_edges=[-0.20, -0.10, -0.05],
        p_quantile=0.5, min_bucket_obs=50,
        train_end=panel.index.max(), train_rows=0,
    )


def _stub_predict(model, monkeypatch, mapping):
    def _fake_predict(X, history=None):
        prob = []; days = []; profit = []
        for s in X["symbol"].astype(str):
            p, d, pr = mapping[s]
            prob.append(p); days.append(d); profit.append(pr)
        return pd.DataFrame({"pred_recovery_prob": prob, "pred_days": days,
                             "pred_profit": profit}, index=X.index)
    monkeypatch.setattr(model, "predict", _fake_predict)


def test_rebound_rank_scores_by_profit_per_day(monkeypatch):
    panel = _panel(["FAST", "MID", "SLOW"])
    model = _model(panel)
    # All clear the recovery gate (fixed min_recovery_prob=0.5, injected below
    # -- NOT read from the live config.yaml, so tuning that knob elsewhere
    # can't silently break this test); ordering is pure P/N:
    # FAST 0.06/3=0.020, MID 0.03/3=0.010, SLOW 0.06/12=0.005.
    _stub_predict(model, monkeypatch, {
        "FAST": (0.96, 3.0, 0.06), "MID": (0.96, 3.0, 0.03),
        "SLOW": (0.96, 12.0, 0.06)})
    monkeypatch.setattr(predict_mod, "tradable_symbols", lambda: None)
    monkeypatch.setattr(predict_mod, "load_config",
                        lambda: type("Cfg", (), {"strategy": {"recovery": {"min_recovery_prob": 0.5}}})())

    out = predict_mod.rank_today(recovery_model=model, n_picks=3, panel=panel)
    order = list(out["symbol"])
    assert order == ["FAST", "MID", "SLOW"], out[["symbol", "score"]]
    # Recovery pricing columns present; no legacy entry_vnd / stop / rr columns.
    assert {"target_vnd", "pred_days", "score", "net_reward_vnd",
            "close_vnd"}.issubset(out.columns)
    assert "rr_ratio" not in out.columns   # legacy ATR risk-reward gone
    assert "entry_vnd" not in out.columns  # no entry-price prediction; buy at close
    # hold_days was dropped (cd8a32c) -- pred_days is the single source of
    # truth for expected hold length now, rounded only at display time.
    assert "hold_days" not in out.columns
    # buy price = close; target = close * (1 + P).
    fast = out[out["symbol"] == "FAST"].iloc[0]
    assert fast["close_vnd"] == 20000
    assert abs(int(fast["target_vnd"]) - round(20000 * 1.06)) <= 1


def test_rebound_rank_healthy_gate_drops_low_prob(monkeypatch):
    """The min_recovery_prob gate (fixed at 0.5 here, injected below -- not
    read from the live config.yaml) filters out a chronic falling-knife name
    before ranking, even if its P/N would rank it high."""
    panel = _panel(["HEALTHY", "KNIFE"])
    model = _model(panel)
    # KNIFE has a huge P/N but only 5% recovery probability -> gated out.
    _stub_predict(model, monkeypatch, {
        "HEALTHY": (0.95, 3.0, 0.04), "KNIFE": (0.05, 2.0, 0.20)})
    monkeypatch.setattr(predict_mod, "tradable_symbols", lambda: None)
    monkeypatch.setattr(predict_mod, "load_config",
                        lambda: type("Cfg", (), {"strategy": {"recovery": {"min_recovery_prob": 0.5}}})())

    out = predict_mod.rank_today(recovery_model=model, n_picks=5, panel=panel)
    syms = set(out["symbol"])
    assert "HEALTHY" in syms
    assert "KNIFE" not in syms


def test_rebound_rank_vol_penalty_demotes_choppy_name(monkeypatch):
    """With vol_penalty.k > 0, two names with IDENTICAL pred_profit/pred_days/
    pred_recovery_prob must be ordered by realvol_20 -- the choppier one
    ranks lower, and the vol_penalty column matches 1/(1 + k*realvol_20)."""
    panel = _panel(["LOWVOL"], realvol_20=0.3)
    panel = pd.concat([panel, _panel(["HIGHVOL"], realvol_20=1.5)])
    model = _model(panel)
    _stub_predict(model, monkeypatch, {
        "LOWVOL": (0.96, 3.0, 0.06), "HIGHVOL": (0.96, 3.0, 0.06)})
    monkeypatch.setattr(predict_mod, "tradable_symbols", lambda: None)
    monkeypatch.setattr(predict_mod, "load_config",
                        lambda: type("Cfg", (), {"strategy": {"recovery": {
                            "min_recovery_prob": 0.5,
                            "vol_penalty": {"k": 1.0, "measure": "realvol_20"}}}})())

    out = predict_mod.rank_today(recovery_model=model, n_picks=2, panel=panel)
    assert list(out["symbol"]) == ["LOWVOL", "HIGHVOL"]
    lowvol = out[out["symbol"] == "LOWVOL"].iloc[0]
    highvol = out[out["symbol"] == "HIGHVOL"].iloc[0]
    assert abs(lowvol["vol_penalty"] - 1.0 / (1.0 + 1.0 * 0.3)) < 1e-9
    assert abs(highvol["vol_penalty"] - 1.0 / (1.0 + 1.0 * 1.5)) < 1e-9
    base = (0.06 / 3.0) * 0.96
    # add_recovery_price_suggestions rounds score to 6 decimals for display.
    assert abs(lowvol["score"] - base * lowvol["vol_penalty"]) < 1e-6


def test_rebound_rank_vol_penalty_off_by_default_ordering_unchanged(monkeypatch):
    """k=0 (the _model default config injected elsewhere) must reproduce plain
    P/N ordering -- the guard that keeps existing tests unaffected."""
    panel = _panel(["LOWVOL"], realvol_20=0.3)
    panel = pd.concat([panel, _panel(["HIGHVOL"], realvol_20=1.5)])
    model = _model(panel)
    _stub_predict(model, monkeypatch, {
        "LOWVOL": (0.96, 3.0, 0.06), "HIGHVOL": (0.96, 3.0, 0.09)})
    monkeypatch.setattr(predict_mod, "tradable_symbols", lambda: None)
    monkeypatch.setattr(predict_mod, "load_config",
                        lambda: type("Cfg", (), {"strategy": {"recovery": {"min_recovery_prob": 0.5}}})())

    out = predict_mod.rank_today(recovery_model=model, n_picks=2, panel=panel)
    # No vol_penalty configured -> pure P/N: HIGHVOL (0.09/3) beats LOWVOL (0.06/3).
    assert list(out["symbol"]) == ["HIGHVOL", "LOWVOL"]
    assert out["vol_penalty"].isna().all()


def test_rebound_rank_empty_without_model(monkeypatch):
    panel = _panel(["AAA"])
    monkeypatch.setattr(predict_mod, "tradable_symbols", lambda: None)
    monkeypatch.setattr(predict_mod, "_try_load_recovery_model", lambda: None)
    out = predict_mod.rank_today(recovery_model=None, n_picks=3, panel=panel)
    assert len(out) == 0
