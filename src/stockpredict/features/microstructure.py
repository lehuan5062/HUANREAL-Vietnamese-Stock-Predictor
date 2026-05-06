"""Liquidity / microstructure features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def adv_vnd(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """20-day average daily traded value (close * volume), in VND."""
    return (df["close"] * df["volume"]).rolling(window).mean()


def intraday_range(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Average normalized intraday range as a slippage proxy."""
    rng = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    return rng.rolling(window).mean()


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["adv_vnd_20"] = adv_vnd(df, 20)
    out["range_20"] = intraday_range(df, 20)
    return out
