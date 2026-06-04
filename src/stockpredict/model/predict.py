"""Score the latest cross-section of features and rank tickers by predicted T+2 return.

Two heads are loaded when available:

* ``models/latest.pkl`` — mean head (decides ranking via ``pred_mean``).
* ``models/low_latest.pkl`` — quantile head (predicts ``pred_low``, used
  by ``add_price_suggestions`` to anchor the limit-buy entry below today's
  close). When the low model is missing, ``pred_low`` is omitted from
  the candidates frame and pricing falls back to entry = today's close.
"""
from __future__ import annotations

import pandas as pd

from ..data.universe import tradable_symbols
from ..dataset import FEATURE_COLS, build_panel
from ..filters import liquidity_mask
from ..pricing import add_price_suggestions
from .train import (
    LowQuantileModel,
    TrainedModel,
    latest_low_model_path,
    latest_model_path,
)


def latest_cross_section(panel: pd.DataFrame, on: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return one row per symbol at the most recent date in `panel` (or `on` if given)."""
    if panel.empty:
        return panel
    if on is not None:
        snap = panel[panel.index <= pd.to_datetime(on)]
    else:
        snap = panel
    # take the last row per symbol within the window
    snap = snap.reset_index().sort_values(["symbol", "date"])
    last = snap.groupby("symbol", as_index=False).tail(1).set_index("date")
    return last


def _try_load_low_model() -> LowQuantileModel | None:
    """Return the cached low-quantile model, or None if it hasn't been
    trained yet. Missing pickle is a soft fallback — the caller produces
    candidates without a ``pred_low`` column and pricing reverts to the
    close-anchored entry."""
    p = latest_low_model_path()
    if not p.exists():
        return None
    try:
        return LowQuantileModel.load(p)
    except Exception:
        # Corrupt pickle (interrupted save, schema mismatch) — treat as
        # missing rather than crashing the whole rank pass.
        return None


def rank_today(model: TrainedModel | None = None,
               on: str | pd.Timestamp | None = None,
               top_k: int = 5,
               actionable_only: bool = False,
               panel: pd.DataFrame | None = None,
               units: int | None = None,
               budget_vnd: int | None = None,
               exit_offset_days: int | None = None,
               symbols: list[str] | None = None,
               low_model: LowQuantileModel | None = None) -> pd.DataFrame:
    """Compute predicted return for every eligible symbol on the given date,
    apply the liquidity filter, and return the candidate picks as a DataFrame.

    Two selection modes:

    * ``actionable_only=False`` (legacy) — cut to the ``top_k`` highest
      ``pred_mean`` rows first, then price just those.
    * ``actionable_only=True`` — price the WHOLE scored universe and return
      every row that clears the ``actionable`` gate (``net_reward > 0 &
      rr >= min_rr``), sorted by ``pred_mean`` descending. There is no cap; a
      ticker is no longer hidden just because it ranks below ``top_k`` by
      predicted return.
    """
    if model is None:
        model = TrainedModel.load(latest_model_path())
    if low_model is None:
        low_model = _try_load_low_model()
    if panel is None:
        # require_target=False so we keep the most recent rows even without a known target
        panel = build_panel(symbols=symbols, require_target=False,
                            exit_offset_days=exit_offset_days)
    elif symbols is not None:
        # Caller pre-built the panel but still wants to restrict ranking
        panel = panel[panel["symbol"].astype(str).str.upper().isin(
            {s.upper() for s in symbols}
        )]

    snap = latest_cross_section(panel, on=on)
    if snap.empty:
        return snap

    # Drop tickers that vnstock no longer lists as tradable on HOSE/HNX/UPCOM
    # (DELISTED, etc.). The OHLCV cache keeps a parquet file for every ticker
    # we have ever fetched — including delisted names like HTK — so without
    # this guard a stale cache entry can re-surface as a top pick the user
    # cannot actually buy. ``tradable_symbols()`` returns None when the
    # universe parquet is missing (cold start); in that case we degrade to
    # the pre-existing behaviour rather than wiping the cross-section.
    tradable = tradable_symbols()
    if tradable is not None:
        snap = snap[snap["symbol"].astype(str).str.upper().isin(tradable)]
    if snap.empty:
        return snap

    mask = liquidity_mask(snap)
    snap = snap[mask].copy()
    if snap.empty:
        return snap

    preds = model.predict(snap)
    snap = snap.assign(**preds)
    if low_model is not None:
        # ``pred_low`` is the predicted ``low[T+1]/close[T] - 1`` at the
        # configured quantile alpha. add_price_suggestions consumes this
        # to derive entry_vnd; if absent, it falls back to entry = close.
        snap["pred_low"] = low_model.predict(snap)
        snap["pred_low_alpha"] = float(low_model.alpha)
    snap["rank"] = snap["pred_mean"].rank(ascending=False, method="dense").astype(int)
    ordered = snap.sort_values("pred_mean", ascending=False)
    if actionable_only:
        # Price the WHOLE liquid/tradable universe (both heads already scored it
        # above, so this only adds vectorized pricing math), then return every
        # row that clears the actionable gate — no cap. The predicted return
        # drives the target price (hence the gate) and the pred_mean sort order.
        out = add_price_suggestions(ordered, units=units, budget_vnd=budget_vnd)
        out = out[out["actionable"].fillna(False).astype(bool)]
    else:
        # Legacy path: cut to the top_k by pred_mean first, then price just those.
        out = ordered.head(top_k)
        out = add_price_suggestions(out, units=units, budget_vnd=budget_vnd)
    cols = ["symbol", "close",
            "position_units", "position_value_vnd",
            "entry_vnd", "close_vnd", "entry_limit_pct",
            "target_vnd", "target_low_vnd", "target_high_vnd",
            "stop_vnd", "gross_reward_vnd", "max_loss_vnd",
            "fees_round_trip_vnd", "net_reward_vnd", "net_loss_vnd",
            "rr_ratio", "breakeven_pct", "actionable", "over_budget",
            "pred_mean", "pred_std", "pred_low", "pred_low_alpha", "rank",
            "rsi_14", "mom_5", "mom_20", "vol_z_20", "adv_vnd_20", "atr_14"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index().rename(columns={"date": "as_of"})
