"""Forward-return target aligned to T+2 close (or T+1 open -> T+2 close)."""
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


def attach_target(df: pd.DataFrame, entry: str | None = None,
                  exit_offset_days: int | None = None) -> pd.DataFrame:
    cfg = load_config().target
    e = entry or cfg["entry"]
    o = int(exit_offset_days) if exit_offset_days is not None else int(cfg["exit_offset_days"])
    out = df.copy()
    out["target"] = forward_return(df, entry=e, exit_offset_days=o)
    return out
