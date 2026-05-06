"""End-to-end smoke test of the walk-forward backtest on synthetic data.

We construct two synthetic tickers where one (signal-A) has a deterministic
2-day-ahead return that's correlated with a known feature, and the other
(noise-N) is a random walk. The backtest should pick signal-A more often
and produce hit_rate > 0.5.
"""
import numpy as np
import pandas as pd
import pytest

from stockpredict.backtest.walk_forward import run as run_backtest
from stockpredict.dataset import FEATURE_COLS


def _make_panel(seed: int = 7, n_days: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    rows = []

    # Feature values that the model could reasonably exploit. We'll build
    # the panel directly with feature columns + target so we don't depend on
    # the rest of the pipeline.
    for sym, signal_strength in [("AAA", 0.5), ("NNN", 0.0)]:
        feats = {}
        for c in FEATURE_COLS:
            feats[c] = rng.normal(0, 1, size=n_days)
        # Make rsi_14 a real predictor for AAA: low rsi -> higher T+2 return
        rsi_vals = rng.uniform(20, 80, size=n_days)
        feats["rsi_14"] = rsi_vals
        # Plausible liquidity so the filter doesn't reject the row
        feats["adv_vnd_20"] = np.full(n_days, 5_000_000_000.0)
        target = signal_strength * (50 - rsi_vals) / 50.0 * 0.02 + rng.normal(0, 0.005, n_days)
        df = pd.DataFrame(feats, index=dates)
        df["target"] = target
        df["close"] = 20_000.0  # above the min_close_vnd gate
        df["symbol"] = sym
        rows.append(df)

    panel = pd.concat(rows).sort_index()
    panel.index.name = "date"
    return panel


@pytest.mark.timeout(120)
def test_backtest_picks_signal():
    panel = _make_panel()
    res = run_backtest(panel=panel, start="2020-01-01", end="2022-12-31", top_k=1)
    assert res.summary["n_trades"] > 0, res.summary
    # The model should learn the rsi_14 -> target relationship and pick AAA more often.
    counts = res.trades["symbol"].value_counts()
    aaa = counts.get("AAA", 0)
    nnn = counts.get("NNN", 0)
    assert aaa > nnn, f"expected AAA > NNN; got AAA={aaa}, NNN={nnn}"
    # Hit rate on a real signal should beat 50% on raw return (cost may push net below)
    assert res.summary["hit_rate"] > 0.5
