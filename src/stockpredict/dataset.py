"""Build a model-ready dataset by joining cached OHLCV + features + target across symbols."""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from .config import load_config
from .data.cache import cached_symbols, read_ohlcv
from .features import microstructure, technical
from .filters import has_enough_history
from .model.target import attach_target


FEATURE_COLS = [
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "mom_5", "mom_20",
    "atr_14",
    "vol_z_20",
    "high_prox_20",
    "gap",
    "realvol_20",
    "adv_vnd_20",
    "range_20",
]


def engineer_one(symbol: str, df: pd.DataFrame | None = None,
                 exit_offset_days: int | None = None) -> pd.DataFrame:
    """Compute features + target for a single symbol's OHLCV."""
    if df is None:
        df = read_ohlcv(symbol)
    if df.empty or not has_enough_history(df):
        return pd.DataFrame()
    cfg = load_config().features
    out = technical.add_all(df, cfg)
    out = microstructure.add_all(out)
    out = attach_target(out, exit_offset_days=exit_offset_days)
    out["symbol"] = symbol
    return out


def build_panel(symbols: list[str] | None = None,
                start: str | None = None,
                end: str | None = None,
                require_target: bool = True,
                exit_offset_days: int | None = None) -> pd.DataFrame:
    """Concatenate per-symbol feature frames into a long panel.

    Each row = one (symbol, date). Filters: optional date window, drop rows where
    any required feature is NaN, optionally drop rows without a target.
    """
    syms = symbols or cached_symbols()
    frames = []
    for s in tqdm(syms, desc="engineer", ncols=80):
        df = engineer_one(s, exit_offset_days=exit_offset_days)
        if df.empty:
            continue
        if start is not None:
            df = df[df.index >= pd.to_datetime(start)]
        if end is not None:
            df = df[df.index <= pd.to_datetime(end)]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames)
    # Refine the liquidity gate column to be calendar-aware: count active days
    # over the trailing 20 *market* trading days (treating days the symbol
    # didn't trade as zero volume), now that the panel reveals the full set of
    # market dates. Overwrites the per-symbol row-based fallback from add_all.
    if {"close", "volume", "symbol"}.issubset(panel.columns):
        min_adv = load_config().universe["liquidity_filter"]["min_adv_vnd"]
        panel["adv_active_days_20"] = microstructure.active_days_calendar(
            panel, min_adv, 20)
    panel = panel.dropna(subset=FEATURE_COLS)
    if require_target:
        panel = panel.dropna(subset=["target"])
    return panel.sort_index()
