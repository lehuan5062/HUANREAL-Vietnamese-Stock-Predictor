"""Score the latest cross-section of features and rank tickers by predicted T+2 return.

Two heads are loaded when available:

* ``models/latest.pkl`` — mean head (decides ranking via ``pred_mean``).
* ``models/low_latest.pkl`` — quantile head (predicts ``pred_low``, used
  by ``add_price_suggestions`` to anchor the limit-buy entry below today's
  close). When the low model is missing, ``pred_low`` is omitted from
  the candidates frame and pricing falls back to entry = today's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.universe import tradable_symbols
from ..dataset import FEATURE_COLS, build_panel
from ..filters import (ceiling_lock_mask, corporate_action_mask, liquidity_mask,
                       overbought_mask)
from ..pricing import add_price_suggestions
from .train import (
    RollingEmpiricalQuantileModel,
    TrainedModel,
    latest_low_model_path,
    latest_missed_model_path,
    latest_model_path,
)


def _round_trip_cost_fraction(broker: dict) -> float:
    """ACBS round-trip fee as a fraction of trade value: buy + sell commission
    (each with VAT) + sell PIT = ``2*c*(1+v) + p`` (~0.0043). Matches the
    ``breakeven_pct`` pricing computes, used as the cost bar for conviction."""
    c = float(broker.get("commission_pct", 0.15)) / 100.0
    v = float(broker.get("vat_pct", 10)) / 100.0
    p = float(broker.get("pit_pct", 0.10)) / 100.0
    return 2.0 * c * (1.0 + v) + p


def conviction_to_alpha(pred_mean: pd.Series, base_alpha: float, *,
                        cost_fraction: float, min_edge_over_cost: float,
                        weak_mult: float, strong_mult: float, strong_edge: float,
                        hard_min: float, hard_max: float) -> pd.Series:
    """Map each pick's conviction to its entry dip quantile alpha, inversely:
    a strong pick (``pred_mean`` well above the cost bar) gets a SHALLOW dip
    (high alpha, fills easily); a marginal / below-bar pick gets a DEEP dip
    (low alpha, only fills at a bargain).

    ``edge_ratio = pred_mean / (min_edge_over_cost * cost_fraction)`` — 1.0 means
    the pick sits exactly on the break-even bar, where alpha == ``base_alpha``
    (so a flat-base config reproduces today's behavior at the margin). The band
    is expressed as multipliers of ``base_alpha`` so it rescales when
    ``entry_low_alpha`` changes. NaN ``pred_mean`` → base."""
    bar = max(min_edge_over_cost * cost_fraction, 1e-9)
    edge = pred_mean.astype(float) / bar
    xp = [0.0, 1.0, float(strong_edge)]
    fp = [base_alpha * weak_mult, base_alpha, base_alpha * strong_mult]
    alpha = pd.Series(np.interp(edge.to_numpy(), xp, fp), index=pred_mean.index)
    alpha = alpha.where(edge.notna(), base_alpha)
    return alpha.clip(hard_min, hard_max)


def overbought_alpha_penalty(rsi: pd.Series, *, start: float, full: float,
                             mult: float) -> pd.Series:
    """Per-row multiplier (≤ 1.0) that DEEPENS the entry dip the more overbought
    a pick is: 1.0 below ``start`` RSI, ramping linearly down to ``mult`` at/above
    ``full``. Multiplied into the conviction alpha so an overbought name only
    fills on a real pullback. NaN RSI → 1.0 (no penalty); ``mult``=1.0 disables;
    a degenerate ``full<=start`` collapses to a step at ``start``."""
    r = rsi.astype(float)
    if full <= start:
        pen = pd.Series(np.where(r > start, mult, 1.0), index=rsi.index)
    else:
        frac = ((r - start) / (full - start)).clip(0.0, 1.0)
        pen = 1.0 - frac * (1.0 - mult)
    return pen.where(r.notna(), 1.0)


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
               low_model: RollingEmpiricalQuantileModel | None = None,
               model_variant: str = "standard") -> pd.DataFrame:
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
        if str(model_variant) == "missed":
            model = TrainedModel.load(latest_missed_model_path())
        else:
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

    # Drop overbought blow-offs (RSI above the configured cap): a name that's
    # run too far tends to reverse, so buying the top is a poor T+2 entry.
    # Off by default (overbought_rsi_max=0); reads a clean RSI post corp-action.
    snap = snap[overbought_mask(snap)].copy()
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
        # the dip quantile alpha. add_price_suggestions consumes this to derive
        # entry_vnd; if absent, it falls back to entry = close. Pass the
        # in-memory panel as history so the quantile is computed from the data
        # already loaded (no per-symbol parquet re-reads).
        pcfg = load_config().pricing
        if bool(pcfg.get("entry_alpha_couple_conviction", True)):
            # Couple dip-depth to conviction: a strong pick gets a shallow dip
            # (high alpha, fills easily); a weak / below-breakeven pick gets a
            # deep dip (low alpha, only fills at a bargain). Driven by pred_mean
            # only — no circularity with the entry/breakeven computed later.
            broker = dict(load_config().broker) if hasattr(load_config(), "broker") else {}
            alphas = conviction_to_alpha(
                snap["pred_mean"], float(low_model.alpha),
                cost_fraction=_round_trip_cost_fraction(broker),
                min_edge_over_cost=float(pcfg.get("min_edge_over_cost", 1.0)),
                weak_mult=float(pcfg.get("entry_alpha_weak_mult", 0.6)),
                strong_mult=float(pcfg.get("entry_alpha_strong_mult", 1.25)),
                strong_edge=float(pcfg.get("entry_alpha_strong_edge", 3.0)),
                hard_min=float(pcfg.get("entry_alpha_hard_min", 0.05)),
                hard_max=float(pcfg.get("entry_alpha_hard_max", 0.75)),
            )
            # Overbought also hardens the entry: the more overbought a surviving
            # pick, the deeper its dip (lower alpha), so it only fills on a real
            # pullback. mult=1.0 disables. Re-clip after the penalty.
            ob_mult = float(pcfg.get("entry_alpha_overbought_mult", 1.0))
            if ob_mult < 1.0 and "rsi_14" in snap.columns:
                alphas = (alphas * overbought_alpha_penalty(
                    snap["rsi_14"],
                    start=float(pcfg.get("entry_alpha_overbought_start", 60.0)),
                    full=float(pcfg.get("entry_alpha_overbought_full", 85.0)),
                    mult=ob_mult,
                )).clip(float(pcfg.get("entry_alpha_hard_min", 0.05)),
                        float(pcfg.get("entry_alpha_hard_max", 0.75)))
            snap["pred_low"] = low_model.predict(
                snap, history=panel, alphas=alphas.to_numpy())
            snap["pred_low_alpha"] = alphas.to_numpy()
        else:
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
