"""Build a model-ready dataset by joining cached OHLCV + features + target across symbols."""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from .config import load_config
from .data.cache import cached_symbols, read_ohlcv
from .features import microstructure, technical
from .filters import band_break_flags, corporate_action_mask, has_enough_history
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


def _drop_corporate_action_rows(panel: pd.DataFrame,
                                exit_offset_days: int | None) -> pd.DataFrame:
    """Remove rows poisoned by an unadjusted corporate action so the model never
    trains on them — the proper companion to the prediction-time
    ``corporate_action_mask``.

    A band-breaking 1-day move (split / rights / special dividend showing
    through the raw feed) corrupts a row two ways:

    * **Look-back** — it sits inside the feature window, so mom_*/atr_14/rsi_14
      read a phantom crash/spike (``corporate_action_mask`` flags this via
      ``max_abs_ret_20``).
    * **Look-forward** — if it lands in the (t, t+H] target window, the realized
      forward return is a fake move, so the *label* mis-teaches the model.

    We drop a row when either window is contaminated. The latest (prediction)
    rows have no future bars, so they can only be look-back-contaminated — which
    keeps this consistent with the snapshot mask. Disabled when
    ``pricing.corp_action_lookback`` is 0 or the support columns are absent."""
    cfg = load_config()
    lookback = int(cfg.pricing.get("corp_action_lookback", 20))
    if lookback <= 0 or "ret_1d" not in panel.columns or "symbol" not in panel.columns:
        return panel
    # Sort so groupby.shift walks each symbol's bars in date order.
    panel = panel.sort_values("symbol", kind="stable").sort_index(kind="stable")
    # Look-back contamination (per-exchange, over the feature window).
    contaminated = ~corporate_action_mask(panel)
    # Look-forward contamination: a band-break anywhere in (t, t+H] fakes the label.
    horizon = int(exit_offset_days if exit_offset_days is not None
                  else cfg.target["exit_offset_days"])
    brk = band_break_flags(panel)
    g = brk.groupby(panel["symbol"], sort=False)
    for k in range(1, horizon + 1):
        # fill_value=False keeps the shifted series boolean (no NaN -> no object
        # downcast); the tail of each symbol has no future bar to break.
        contaminated = contaminated | g.shift(-k, fill_value=False)
    return panel[~contaminated]


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
    panel = _drop_corporate_action_rows(panel, exit_offset_days)
    panel = panel.dropna(subset=FEATURE_COLS)
    if require_target:
        panel = panel.dropna(subset=["target"])
    # Canonical (date, symbol) row order. A plain ``sort_index()`` orders by
    # date only, leaving same-date rows in concat (i.e. input-symbol) order —
    # so the row layout depended on the order symbols were passed in. Because
    # the LightGBM trainer subsamples rows/features BY POSITION
    # (bagging_fraction / feature_fraction), that made the trained model — and
    # every prediction — depend on incidental symbol ordering (selector order
    # vs warm+stale order vs alphabetical), which is a reproducibility bug.
    # Sorting by symbol first, then a stable sort by date, gives a total order
    # keyed on (date, symbol) that is invariant to input order. Date stays
    # primary, so the temporal train/validation split is unchanged.
    return panel.sort_values("symbol", kind="stable").sort_index(kind="stable")
