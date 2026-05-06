"""Vectorized technical indicators. All functions take a DataFrame with
columns [open, high, low, close, volume] and return a Series aligned to the index."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder's smoothing via EMA with alpha=1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    # Handle degenerate cases explicitly: pure uptrend -> RSI 100, pure downtrend -> 0
    out = pd.Series(np.nan, index=close.index, dtype=float)
    nonzero = avg_loss > 0
    rs = avg_gain[nonzero] / avg_loss[nonzero]
    out[nonzero] = 100.0 - (100.0 / (1.0 + rs))
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    # mask the leading bar (NaN delta)
    out.iloc[0] = np.nan
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": hist})


def momentum(close: pd.Series, window: int) -> pd.Series:
    # Guard against rows where the price is 0 (newly listed, suspended,
    # or data-feed glitch). Replacing with NaN propagates cleanly through
    # the dropna at the end of feature engineering — and silences the
    # cosmetic "divide by zero in log" runtime warning.
    safe = close.replace(0, np.nan)
    return np.log(safe / safe.shift(window))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std(ddof=0)
    return (volume - mean) / std.replace(0, np.nan)


def high_proximity(close: pd.Series, window: int = 20) -> pd.Series:
    """How close current close is to the rolling-window high. 0 = at the high, negative = below."""
    rolling_max = close.rolling(window).max()
    return close / rolling_max - 1.0


def gap(df: pd.DataFrame) -> pd.Series:
    safe_prev_close = df["close"].shift(1).replace(0, np.nan)
    return df["open"] / safe_prev_close - 1.0


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    safe = close.replace(0, np.nan)
    log_ret = np.log(safe / safe.shift(1))
    return log_ret.rolling(window).std(ddof=0) * np.sqrt(252)


def add_all(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Compute all technical features and return df with new columns appended."""
    p = params or {}
    out = df.copy()
    out["rsi_14"] = rsi(df["close"], p.get("rsi_period", 14))
    macd_p = p.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    out = pd.concat([out, macd(df["close"], **macd_p)], axis=1)
    for w in p.get("momentum_windows", [5, 20]):
        out[f"mom_{w}"] = momentum(df["close"], w)
    out["atr_14"] = atr(df, p.get("atr_period", 14))
    out["vol_z_20"] = volume_zscore(df["volume"], p.get("volume_zscore_window", 20))
    out["high_prox_20"] = high_proximity(df["close"], 20)
    out["gap"] = gap(df)
    out["realvol_20"] = realized_vol(df["close"], p.get("realized_vol_window", 20))
    return out
