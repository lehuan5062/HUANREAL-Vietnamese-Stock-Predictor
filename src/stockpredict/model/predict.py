"""Score the latest cross-section of features and rank tickers by predicted T+2 return."""
from __future__ import annotations

import pandas as pd

from ..dataset import FEATURE_COLS, build_panel
from ..filters import liquidity_mask
from ..pricing import add_price_suggestions
from .train import TrainedModel, latest_model_path


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


def rank_today(model: TrainedModel | None = None,
               on: str | pd.Timestamp | None = None,
               top_k: int = 5,
               panel: pd.DataFrame | None = None,
               units: int | None = None,
               exit_offset_days: int | None = None,
               symbols: list[str] | None = None) -> pd.DataFrame:
    """Compute predicted return for every eligible symbol on the given date,
    apply the liquidity filter, and return the top_k as a DataFrame."""
    if model is None:
        model = TrainedModel.load(latest_model_path())
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

    mask = liquidity_mask(snap)
    snap = snap[mask].copy()
    if snap.empty:
        return snap

    preds = model.predict(snap)
    snap = snap.assign(**preds)
    snap["rank"] = snap["pred_mean"].rank(ascending=False, method="dense").astype(int)
    out = snap.sort_values("pred_mean", ascending=False).head(top_k)
    out = add_price_suggestions(out, units=units)
    cols = ["symbol", "close",
            "position_units", "position_value_vnd",
            "entry_vnd", "target_vnd", "target_low_vnd", "target_high_vnd",
            "stop_vnd", "gross_reward_vnd", "max_loss_vnd",
            "fees_round_trip_vnd", "net_reward_vnd", "net_loss_vnd",
            "rr_ratio", "breakeven_pct", "actionable",
            "pred_mean", "pred_std", "rank",
            "rsi_14", "mom_5", "mom_20", "vol_z_20", "adv_vnd_20", "atr_14"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index().rename(columns={"date": "as_of"})
