"""Filter the latest cross-section to downtrend names and rank them by the
rebound score = P/N × recovery_probability.

Loads ``models/recovery_latest.pkl`` (the Kaplan-Meier recovery estimator).
When it is absent (cold install) ``rank_today`` returns an empty frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.universe import tradable_symbols
from ..dataset import FEATURE_COLS, build_panel
from ..filters import (ceiling_lock_mask, corporate_action_mask, downtrend_mask,
                       liquidity_mask, overbought_mask, staleness_mask)
from ..pricing import add_recovery_price_suggestions
from .train import RecoveryKMModel, latest_recovery_model_path


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


def _try_load_recovery_model() -> RecoveryKMModel | None:
    """Return the cached Kaplan-Meier recovery head, or None if it hasn't been
    trained yet / the pickle is unreadable. A soft fallback so a cold install
    degrades to an empty pick set with a clear message rather than crashing."""
    p = latest_recovery_model_path()
    if not p.exists():
        return None
    try:
        return RecoveryKMModel.load(p)
    except Exception:
        return None


def rank_today(recovery_model: RecoveryKMModel | None = None,
               on: str | pd.Timestamp | None = None,
               n_picks: int = 5,
               panel: pd.DataFrame | None = None,
               symbols: list[str] | None = None) -> pd.DataFrame:
    """Rebound ranking: score the downtrend-filtered cross-section by
    ``score = P/N * recovery_prob`` (profit per day held, risk-adjusted by the
    eventual-recovery probability) and return EXACTLY ``n_picks`` priced picks.

    The eligible universe is already narrowed to downtrend names by
    ``eligible_universe``, so this only scores + ranks + prices. Returns an empty
    frame when the recovery model is missing (cold install)."""
    if recovery_model is None:
        recovery_model = _try_load_recovery_model()
    if panel is None:
        panel = build_panel(symbols=symbols, require_target=False)
    elif symbols is not None:
        panel = panel[panel["symbol"].astype(str).str.upper().isin(
            {s.upper() for s in symbols})]
    snap = eligible_universe(on=on, panel=panel)
    if snap.empty or recovery_model is None:
        return snap.iloc[0:0]

    preds = recovery_model.predict(snap, history=panel)
    snap = snap.assign(**preds)
    # Healthy-ticker gate: drop names whose estimated recovery probability is
    # below the floor (per-ticker reliability is the dominant signal). This
    # screens out chronic falling knives that would otherwise be held
    # indefinitely under the no-stop / no-cap flexible exit. 0 disables.
    min_prob = float(dict(getattr(load_config(), "strategy", {}) or {})
                     .get("recovery", {}).get("min_recovery_prob", 0.0) or 0.0)
    if min_prob > 0:
        snap = snap[snap["pred_recovery_prob"] >= min_prob].copy()
        if snap.empty:
            return snap.iloc[0:0]
    snap["score"] = (
        (snap["pred_profit"] / snap["pred_days"].clip(lower=1.0))
        * snap["pred_recovery_prob"]
    )
    snap["rank"] = snap["score"].rank(ascending=False, method="dense").astype(int)
    ordered = snap.sort_values("score", ascending=False)
    out = add_recovery_price_suggestions(ordered.head(int(n_picks)))
    cols = ["symbol", "close",
            "close_vnd", "target_vnd",
            "score", "pred_days", "pred_profit", "pred_recovery_prob",
            "gross_reward_vnd", "fees_round_trip_vnd", "net_reward_vnd",
            "breakeven_pct", "below_recovery_bar", "suggested_max_units", "rank",
            "rsi_14", "mom_5", "mom_20", "high_prox_20", "vol_z_20",
            "adv_vnd_20", "atr_14"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index().rename(columns={"date": "as_of"})


def eligible_universe(on: str | pd.Timestamp | None = None,
                      panel: pd.DataFrame | None = None,
                      symbols: list[str] | None = None) -> pd.DataFrame:
    """Return the mechanically-filtered cross-section of tradable names on the
    given date — UNCAPPED, with NO model scoring and NO pricing.

    Applies the same filter cascade ``rank_today`` uses before it scores:
    ``latest_cross_section → tradable_symbols → liquidity_mask →
    ceiling_lock_mask → corporate_action_mask → overbought_mask →
    downtrend_mask``. Loads no model file.

    Used by the LLM-only Claude mode, which hands this whole eligible downtrend
    universe to the LLM to select / rank / price itself; ``rank_today`` calls it
    too so both paths share one filter definition.
    """
    if panel is None:
        # require_target=False so we keep the most recent rows even without a known target
        panel = build_panel(symbols=symbols, require_target=False)
    elif symbols is not None:
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

    # Drop names whose most recent cached bar is stale relative to the pick
    # date: they'd be scored on a months-old close and that close recorded as
    # the entry price (VTS/DCH, July 2026). Print the exclusions — a stale
    # cache is a data problem the user should see, not a silent drop.
    ref = pd.to_datetime(on) if on is not None else pd.Timestamp(snap.index.max())
    fresh = staleness_mask(snap, ref)
    if not fresh.all():
        stale_syms = sorted(snap.loc[~fresh, "symbol"].astype(str))
        shown = (", ".join(stale_syms) if len(stale_syms) <= 20
                 else ", ".join(stale_syms[:20]) + f", ... +{len(stale_syms) - 20} more")
        print(f"[filters] dropped {len(stale_syms)} stale-data candidate(s) "
              f"(last bar too old for {ref.date()}): {shown}")
        snap = snap[fresh].copy()
    if snap.empty:
        return snap

    mask = liquidity_mask(snap)
    snap = snap[mask].copy()
    if snap.empty:
        return snap

    # Drop names locked limit-up: they closed at the daily ceiling, so the buy
    # session opens with a queue and no sellers and a limit-buy can't fill.
    snap = snap[ceiling_lock_mask(snap)].copy()
    if snap.empty:
        return snap

    # Drop names whose recent history holds a band-breaking 1-day move: that's
    # an unadjusted corporate action (split / rights / special dividend), not a
    # real crash, and it poisons mom_*/atr_14/rsi_14 — the model misreads it as
    # an oversold bounce (e.g. VVS's -38% ex-rights gap).
    snap = snap[corporate_action_mask(snap)].copy()
    if snap.empty:
        return snap

    # Drop overbought blow-offs (RSI above the configured cap): off by default
    # (overbought_rsi_max=0); reads a clean RSI post corp-action.
    snap = snap[overbought_mask(snap)].copy()
    if snap.empty:
        return snap

    # Keep only names in a downtrend — the rebound candidates.
    snap = snap[downtrend_mask(snap)].copy()
    return snap
