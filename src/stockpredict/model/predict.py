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

from ..config import load_config
from ..data.universe import tradable_symbols
from ..dataset import FEATURE_COLS, build_panel
from ..filters import ceiling_lock_mask, corporate_action_mask, liquidity_mask
from ..pricing import add_price_suggestions
from .train import (
    RollingEmpiricalQuantileModel,
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


def _try_load_low_model() -> RollingEmpiricalQuantileModel | None:
    """Return the cached low head, or None if it hasn't been trained yet.
    Missing pickle is a soft fallback — the caller produces candidates without
    a ``pred_low`` column and pricing reverts to the close-anchored entry.

    A stale pickle from the previous LightGBM ``LowQuantileModel`` class no
    longer unpickles (the class was removed); that raises here and is treated
    as missing, so entries fall back to close until the user retrains."""
    p = latest_low_model_path()
    if not p.exists():
        return None
    try:
        return RollingEmpiricalQuantileModel.load(p)
    except Exception:
        # Corrupt pickle (interrupted save, schema mismatch) — treat as
        # missing rather than crashing the whole rank pass.
        return None


def rank_today(model: TrainedModel | None = None,
               on: str | pd.Timestamp | None = None,
               n_picks: int = 5,
               panel: pd.DataFrame | None = None,
               exit_offset_days: int | None = None,
               symbols: list[str] | None = None,
               low_model: RollingEmpiricalQuantileModel | None = None) -> pd.DataFrame:
    """Compute predicted return for every eligible symbol on the given date,
    apply the liquidity / tradable / ceiling / glitch filters, and return
    EXACTLY ``n_picks`` picks — the top ``n_picks`` rows by ``pred_mean``,
    priced.

    Selection is exactly-N: the whole scored universe is ranked by
    ``pred_mean`` descending and the top ``n_picks`` are kept, so the implicit
    difficulty (the edge cutoff) floats to whatever admits exactly that count.
    Picks below the break-even quality bar are still returned but carry
    ``below_breakeven=True`` so the caller can warn. If the eligible universe
    holds fewer than ``n_picks`` rows (a tiny cache / heavy --exclude /
    --hose-only), fewer rows are returned — the caller surfaces the shortfall.
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

    preds = model.predict(snap)
    snap = snap.assign(**preds)
    # Sanity guard against unadjusted split / corporate-action artifacts: the
    # mean head is a calibrated 2-day return predictor, so a |pred_mean| far
    # outside its real range is the model extrapolating a huge bounce off a
    # broken price (e.g. mom_20 ≈ -0.9 from a missed split). Drop those rows
    # before ranking/pricing so a data glitch can't become the top — and, under
    # the old gate, the ONLY — actionable pick. Configurable; 0 disables.
    max_abs_pred = float(load_config().pricing.get("max_abs_pred_mean", 0.0))
    if max_abs_pred > 0:
        snap = snap[snap["pred_mean"].abs() <= max_abs_pred].copy()
        if snap.empty:
            return snap
    if low_model is not None:
        # ``pred_low`` is the per-ticker empirical ``low[T+1]/close[T] - 1`` at
        # the configured quantile alpha. add_price_suggestions consumes this to
        # derive entry_vnd; if absent, it falls back to entry = close. Pass the
        # in-memory panel as history so the quantile is computed from the data
        # already loaded (no per-symbol parquet re-reads).
        snap["pred_low"] = low_model.predict(snap, history=panel)
        snap["pred_low_alpha"] = float(low_model.alpha)
    snap["rank"] = snap["pred_mean"].rank(ascending=False, method="dense").astype(int)
    ordered = snap.sort_values("pred_mean", ascending=False)
    # Keep the top n_picks by pred_mean, then price just those. Taking the top
    # N is the same as auto-tuning the edge gate to admit exactly N — the
    # cutoff floats to the Nth pick's score. Pricing flags any of these N that
    # fall below the break-even quality bar (below_breakeven=True).
    out = add_price_suggestions(ordered.head(int(n_picks)))
    cols = ["symbol", "close",
            "entry_vnd", "close_vnd", "entry_limit_pct",
            "target_vnd", "target_low_vnd", "target_high_vnd",
            "stop_vnd", "gross_reward_vnd", "max_loss_vnd",
            "fees_round_trip_vnd", "net_reward_vnd", "net_loss_vnd",
            "rr_ratio", "breakeven_pct", "below_breakeven", "suggested_max_units",
            "pred_mean", "pred_std", "pred_low", "pred_low_alpha", "rank",
            "rsi_14", "mom_5", "mom_20", "vol_z_20", "adv_vnd_20", "atr_14"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index().rename(columns={"date": "as_of"})
