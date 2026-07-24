"""Build a model-ready dataset by joining cached OHLCV + features + target across symbols."""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from .config import load_config
from .data.cache import cached_symbols, read_ohlcv
from .features import microstructure, technical
from .filters import corporate_action_mask


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


# Minimum bars needed to compute the rolling technical indicators (longest
# window is the 20-day ones + a buffer) — a data-computability floor, not a
# strategy/liquidity judgment threshold (those are plain columns the LLM
# agent reasons over itself; see selector.eligible_universe).
_MIN_BARS_FOR_FEATURES = 60


def engineer_one(symbol: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute technical/microstructure features for a single symbol's OHLCV."""
    if df is None:
        df = read_ohlcv(symbol)
    if df.empty or len(df) < _MIN_BARS_FOR_FEATURES:
        return pd.DataFrame()
    cfg = load_config().features
    out = technical.add_all(df, cfg)
    out = microstructure.add_all(out)
    out["symbol"] = symbol
    return out


def _drop_corporate_action_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose FEATURES are poisoned by an unadjusted corporate action
    — the training-time companion to the prediction-time
    ``corporate_action_mask``.

    A band-breaking 1-day move (split / rights / special dividend showing
    through the raw feed) inside the look-back feature window makes
    mom_*/atr_14/rsi_14 read a phantom crash/spike, so those rows are dropped.
    Forward (label) contamination needs no window here: ``recovery_episode``
    itself censors a recovery scan at the first future band-break, so a phantom
    jump can never be labeled a bounce. Disabled when
    ``pricing.corp_action_lookback`` is 0 or the support columns are absent."""
    cfg = load_config()
    lookback = int(cfg.pricing.get("corp_action_lookback", 20))
    if lookback <= 0 or "ret_1d" not in panel.columns or "symbol" not in panel.columns:
        return panel
    # Sort so downstream group operations walk each symbol's bars in date order.
    panel = panel.sort_values("symbol", kind="stable").sort_index(kind="stable")
    contaminated = ~corporate_action_mask(panel)
    return panel[~contaminated]


def build_panel(symbols: list[str] | None = None,
                start: str | None = None,
                end: str | None = None,
                require_target: bool = False) -> pd.DataFrame:
    """Concatenate per-symbol feature frames into a long panel.

    Each row = one (symbol, date). Filters: optional date window, drop rows
    where any required feature is NaN. ``require_target`` is accepted for
    backward compatibility with older call sites but is a no-op — there is no
    more ML target to require now that no model is trained.
    """
    syms = symbols or cached_symbols()
    frames = []
    for s in tqdm(syms, desc="engineer", ncols=80):
        df = engineer_one(s)
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
        min_adv = load_config().universe["active_day_vnd_threshold"]
        panel["adv_active_days_20"] = microstructure.active_days_calendar(
            panel, min_adv, 20)
    panel = _drop_corporate_action_rows(panel)
    panel = panel.dropna(subset=FEATURE_COLS)
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
