"""Smoke test of the rebound walk-forward backtest on synthetic data.

Two downtrend tickers on a gently-rising price path (so every buy recovers to
the profit target within a few days). The backtest should run end-to-end and
report the rebound summary (recovery_rate, mean_hold_days, exit reasons)."""
import numpy as np
import pandas as pd
import pytest

from stockpredict.backtest.walk_forward import run as run_backtest
from stockpredict.model.target import recovery_episode
from stockpredict.pricing import profit_threshold


def _make_panel(n_days: int = 900) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    thr = profit_threshold()
    frames = []
    for i, sym in enumerate(["AAA", "BBB"]):
        # Gently rising close so a buy recovers to the profit target in a few days.
        drift = 0.003 + 0.001 * i
        close = 100.0 * (1.0 + drift) ** np.arange(n_days)
        df = pd.DataFrame({"close": close}, index=dates)
        # Force every row to look like a downtrend candidate (the filters read
        # these columns; the close path drives recovery labeling + the exit sim).
        df["symbol"] = sym
        df["rsi_14"] = 40.0
        df["mom_20"] = -0.08
        df["high_prox_20"] = -0.08
        df["adv_vnd_20"] = 5_000_000_000.0
        df["adv_active_days_20"] = 20.0
        df["atr_14"] = 1.0
        df["realvol_20"] = 0.5  # exercises the real (non-short-circuit) vol-penalty path
        rec = recovery_episode(df, thr=thr, max_horizon=60)
        df["target_days_to_recover"] = rec["target_days_to_recover"]
        df["target_recovery_return"] = rec["target_recovery_return"]
        df["target_recovered"] = rec["target_recovered"]
        frames.append(df)
    panel = pd.concat(frames).sort_index()
    panel.index.name = "date"
    return panel


@pytest.mark.timeout(120)
def test_rebound_backtest_runs():
    panel = _make_panel()
    res = run_backtest(panel=panel, start="2020-01-01", end="2021-06-30", top_k=1)
    s = res.summary
    assert s["strategy"] == "rebound"
    assert s["n_trades"] > 0, s
    # On a monotonically rising series every trade recovers to the target.
    assert s["recovery_rate"] > 0.9
    assert s["hit_rate"] > 0.9
    assert "mean_hold_days" in s and s["mean_hold_days"] >= 1
