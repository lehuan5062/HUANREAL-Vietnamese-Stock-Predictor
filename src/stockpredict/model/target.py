"""Forward-return target aligned to T+2 close (or T+1 open -> T+2 close).

Two targets are produced from each per-symbol OHLCV frame:

* ``target`` — forward return for the trade exit (close-to-close at the
  configured horizon, or next_open-to-close).
* ``target_low`` — next-day low return, ``low[T+1]/close[T] - 1``. Used by
  the quantile "low predictor" head so the user can place a realistic
  limit-buy order at a price below today's close instead of buying at
  today's close. Typically negative; positive only on overnight gap-ups
  where T+1's low sits above T's close.
"""
from __future__ import annotations

import pandas as pd

from ..config import load_config


def forward_return(df: pd.DataFrame, entry: str = "close",
                   exit_offset_days: int = 2) -> pd.Series:
    """Return target indexed by T such that target[T] is the realized return of
    a position that enters at T close (or T+1 open) and exits at T+exit_offset close.

    For entry='close':       target[T] = close[T+k]/close[T] - 1
    For entry='next_open':   target[T] = close[T+k]/open[T+1] - 1
    """
    if entry == "close":
        return df["close"].shift(-exit_offset_days) / df["close"] - 1.0
    if entry == "next_open":
        return df["close"].shift(-exit_offset_days) / df["open"].shift(-1) - 1.0
    raise ValueError(f"unknown entry mode: {entry}")


def next_day_low_return(df: pd.DataFrame) -> pd.Series:
    """Return target indexed by T such that target[T] = low[T+1]/close[T] - 1.

    Drives the quantile "low" head: a quantile regression on this target
    gives a predicted dip the buyer can place as a limit order. Quantile
    alpha controls fill probability (P(fill) ≈ alpha when limit equals the
    alpha-th quantile of next-day low).
    """
    if "low" not in df.columns:
        raise KeyError("'low' column required for next_day_low_return")
    return df["low"].shift(-1) / df["close"] - 1.0


def attach_target(df: pd.DataFrame, entry: str | None = None,
                  exit_offset_days: int | None = None) -> pd.DataFrame:
    cfg = load_config().target
    e = entry or cfg["entry"]
    o = int(exit_offset_days) if exit_offset_days is not None else int(cfg["exit_offset_days"])
    out = df.copy()
    out["target"] = forward_return(df, entry=e, exit_offset_days=o)
    # ``target_low`` rides alongside ``target`` so a single panel build
    # serves both training heads. Rows where either target is NaN are
    # filtered per-head at the dropna step in the trainer.
    if "low" in df.columns:
        out["target_low"] = next_day_low_return(df)
    return out
