"""Sanity checks for technical indicators on small synthetic series."""
import numpy as np
import pandas as pd

from stockpredict.features.technical import (
    atr,
    high_proximity,
    macd,
    momentum,
    rsi,
    volume_zscore,
)


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_rsi_strict_uptrend_is_100():
    s = _series(np.arange(1, 60, dtype=float))
    r = rsi(s, period=14)
    # in a series of strictly rising prices the RSI converges to 100
    assert r.iloc[-1] > 99.0


def test_rsi_strict_downtrend_is_0():
    s = _series(np.arange(60, 1, -1, dtype=float))
    r = rsi(s, period=14)
    assert r.iloc[-1] < 1.0


def test_momentum_log_ratio():
    s = _series([100.0, 110.0, 121.0, 133.1])
    m = momentum(s, window=2)
    assert np.isclose(m.iloc[2], np.log(121.0 / 100.0))
    assert np.isclose(m.iloc[3], np.log(133.1 / 110.0))


def test_macd_zero_when_flat():
    s = _series([100.0] * 60)
    out = macd(s)
    assert np.isclose(out["macd"].iloc[-1], 0.0)
    assert np.isclose(out["macd_signal"].iloc[-1], 0.0)


def test_atr_positive_with_range():
    df = pd.DataFrame({
        "high": [10, 12, 14, 13, 15],
        "low": [8, 9, 11, 10, 12],
        "close": [9, 11, 13, 12, 14],
    }, index=pd.date_range("2024-01-01", periods=5, freq="B"), dtype=float)
    a = atr(df, period=3)
    assert (a.dropna() > 0).all()


def test_high_proximity_at_high_is_zero():
    s = _series([1, 2, 3, 4, 5, 4, 3, 5, 5, 5])
    p = high_proximity(s, window=5)
    # at the rolling-window high, proximity should be 0
    assert np.isclose(p.iloc[-1], 0.0)


def test_volume_zscore_signs():
    v = _series([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                 100, 100, 100, 100, 100, 100, 100, 100, 500])
    z = volume_zscore(v, window=20)
    # spike on the last bar -> large positive z (4.36 with this exact pattern)
    assert z.iloc[-1] > 4
