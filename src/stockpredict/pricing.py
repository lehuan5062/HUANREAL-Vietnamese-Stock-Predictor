"""Translate ML predictions into actionable buy/target/stop prices, on a
per-share basis and net of broker fees.

vnstock prices are in **thousand VND** (e.g. close=15.35 means 15,350 VND).
We expose all suggestion columns in absolute VND (integer) since that's how
Vietnamese traders enter orders in their broker app.

Two entries are surfaced:

* ``entry_vnd`` — the **limit-buy** price the user should place. When a
  ``pred_low`` column is present (produced by the quantile low head),
  it equals ``close * (1 + pred_low) * 1000`` (clipped so we never quote
  a limit above today's close). When ``pred_low`` is absent, this falls
  back to ``close * 1000`` so legacy installs still work.
* ``close_vnd`` — today's close in VND, kept as a reference column so
  the user can compare quoted entry against the close even when the
  limit-prediction shifts ``entry_vnd`` below it.

All risk-reward, fees, and the ``actionable`` gate use ``entry_vnd`` (the
realistic limit price), so what the user sees is exactly the trade they'd
place if the limit fills. All P&L figures are per share — the user decides
their own position size; the gate is size-invariant.

ACBS fee model (default — override in config.yaml):
  buy  cost = trade_value * commission_pct * (1 + vat_pct/100)
  sell cost = trade_value * commission_pct * (1 + vat_pct/100) + trade_value * pit_pct
  total round-trip ~ 0.43% of trade value at ACBS's 0.15% commission.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config


def _broker_costs(buy_value: pd.Series, sell_value: pd.Series, broker: dict
                  ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (buy_fee, sell_fee, total_fee, breakdown_dict) all in VND.

    breakdown is a Series of dicts so the caller can audit the components
    if they want — we only surface the totals on the picks frame to keep it
    readable.
    """
    commission = float(broker["commission_pct"]) / 100.0
    vat = float(broker["vat_pct"]) / 100.0
    pit = float(broker["pit_pct"]) / 100.0
    min_fee = float(broker.get("min_fee_vnd", 0))

    buy_commission = (buy_value * commission).clip(lower=min_fee)
    sell_commission = (sell_value * commission).clip(lower=min_fee)
    buy_vat = buy_commission * vat
    sell_vat = sell_commission * vat
    sell_pit = sell_value * pit

    buy_fee = buy_commission + buy_vat
    sell_fee = sell_commission + sell_vat + sell_pit
    total = buy_fee + sell_fee
    return buy_fee, sell_fee, total, sell_pit  # last is just for audit if needed


def add_price_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    """Append entry / target / stop / fees / net P&L columns to a candidates frame.

    Required input columns (already produced by the feature pipeline):
        close      — today's close in thousand VND
        pred_mean  — predicted forward return (e.g. +0.0017 = +0.17%)
        pred_std   — model dispersion across the seed ensemble
        atr_14     — 14-day ATR in thousand VND

    Optional input column:
        pred_low   — quantile prediction of ``low[T+1]/close[T] - 1``
                     (typically negative). When present, ``entry_vnd``
                     becomes the limit-buy price at that predicted dip
                     and the stop is anchored on the limit (so the
                     stop_atr_mult * ATR risk distance is exact). When
                     absent, ``entry_vnd = close * 1000`` (legacy).

    All P&L is computed PER SHARE — there is no position-size input. Position
    sizing is left entirely to the user; the ``actionable`` gate (and rr_ratio /
    breakeven_pct) is size-invariant, so a per-share view is sufficient. The one
    sizing hint we surface is ``suggested_max_units`` — an *advisory* liquidity
    cap derived from ``adv_vnd_20`` (see below) that never touches the gate.

    Output columns appended (all VND per share, integer where applicable):
        entry_vnd                limit-buy price (= close*(1+pred_low) when present)
        close_vnd                reference: today's close in VND
        entry_limit_pct          predicted dip relative to close (clipped <= 0)
        target_vnd, target_low_vnd, target_high_vnd, stop_vnd
        gross_reward_vnd         target - entry (per share)
        max_loss_vnd             entry - stop (per share)
        fees_round_trip_vnd      buy commission+VAT + sell commission+VAT + sell PIT
        net_reward_vnd           gross_reward - fees   (the headline number)
        net_loss_vnd             max_loss + fees       (worst-case if stopped out)
        rr_ratio                 net_reward / net_loss (≈ target_atr_mult/stop_atr_mult)
        breakeven_pct            price move needed just to cover fees
        actionable               pred_mean >= min_edge_over_cost*breakeven_pct
                                 AND net_reward > 0 AND rr_ratio >= min_rr_ratio.
                                 The take-profit is entry + target_atr_mult*ATR
                                 (NOT close*(1+pred_mean)); pred_mean only drives
                                 ranking and the directional edge gate.
        suggested_max_units      advisory liquidity cap = floor(
                                 max_participation_pct% * adv_vnd_20*1000 / entry_vnd);
                                 null when adv_vnd_20 absent or the cap is disabled.
                                 Informational only — never feeds the actionable gate.
    """
    if df is None or len(df) == 0:
        return df

    cfg = load_config()
    broker = dict(cfg.broker) if hasattr(cfg, "broker") else {}
    pricing_cfg = dict(cfg.pricing) if hasattr(cfg, "pricing") else {}

    stop_mult = float(pricing_cfg.get("stop_atr_mult", 1.5))
    target_mult = float(pricing_cfg.get("target_atr_mult", 2.0))
    min_rr = float(pricing_cfg.get("min_rr_ratio", 0.8))
    min_edge_over_cost = float(pricing_cfg.get("min_edge_over_cost", 1.0))
    max_participation_pct = float(pricing_cfg.get("max_participation_pct", 0.0))

    out = df.copy()

    close_k = out["close"].astype(float)
    pred = out["pred_mean"].astype(float)
    pred_std = out.get("pred_std", pd.Series(0.0, index=out.index)).astype(float)
    atr_k = out.get("atr_14", pd.Series(np.nan, index=out.index)).astype(float)

    # Predicted next-day low return. Default to 0 (entry == close) so the
    # legacy code path — feature frames without a low model — keeps the
    # original behavior. Clip at 0 so we never quote an entry ABOVE close
    # (placing a limit above the market would just buy at market on open
    # and defeats the purpose of a limit-buy).
    pred_low = out.get("pred_low", pd.Series(0.0, index=out.index)).astype(float)
    pred_low_eff = pred_low.fillna(0.0).clip(upper=0.0)

    # Per-share prices in VND. ``entry_v`` is the limit price (= close at
    # the configured dip); ``close_v`` keeps the close-in-VND for display.
    close_v = (close_k * 1000.0).round(0)
    entry_v = (close_k * (1.0 + pred_low_eff) * 1000.0).round(0)
    # Take-profit and stop are BOTH anchored on the LIMIT entry and scaled by
    # ATR(14), so the realized risk-reward (target_atr_mult / stop_atr_mult) is
    # controlled by config rather than hostage to the model's near-zero 2-day
    # return forecast. ``pred_mean`` no longer sets the target price — it only
    # drives ranking and the directional edge gate (see ``actionable`` below).
    # ``target_low/high`` keep a forecast-uncertainty band (±pred_std) around the
    # ATR target so the displayed range still reflects model dispersion.
    entry_k = entry_v / 1000.0
    band_k = pred_std * entry_k
    target_v = ((entry_k + target_mult * atr_k) * 1000.0).round(0)
    target_low_v = ((entry_k + target_mult * atr_k - band_k) * 1000.0).round(0)
    target_high_v = ((entry_k + target_mult * atr_k + band_k) * 1000.0).round(0)
    stop_v = ((entry_k - stop_mult * atr_k) * 1000.0).round(0)

    # Per-share P&L. The user sizes the position themselves; everything below
    # is one share's worth of reward / risk / fees. The actionable gate and
    # rr_ratio are size-invariant, so a per-share view is sufficient.
    gross_reward = target_v - entry_v
    max_loss_units = entry_v - stop_v

    buy_fee, sell_fee, fees_total, _ = _broker_costs(entry_v, target_v, broker)
    net_reward = gross_reward - fees_total
    net_loss = max_loss_units + fees_total

    # rr_ratio: net upside vs net downside, undefined when stop is missing/invalid
    rr = pd.Series(np.nan, index=out.index, dtype=float)
    valid = (max_loss_units > 0) & net_loss.notna()
    rr[valid] = net_reward[valid] / net_loss[valid]

    breakeven_pct = (fees_total / entry_v).round(4)
    # Directional edge gate: the model's predicted forward return must clear
    # ``min_edge_over_cost`` times the round-trip cost (breakeven_pct). With the
    # ATR-scaled target, net_reward is structurally positive and rr is ~constant,
    # so this edge gate — not rr — is what selects how many tickers are
    # actionable. rr >= min_rr remains as a sanity floor (e.g. a degenerate ATR).
    edge_ok = pred >= (min_edge_over_cost * breakeven_pct)
    actionable = edge_ok & (net_reward > 0) & (rr >= min_rr) & valid

    # Advisory liquidity-driven unit cap. adv_vnd_20 is the 20-day average daily
    # traded value in THOUSAND-VND (close-in-thousand-VND * shares), so *1000 to
    # get VND, then participation_pct% of that, divided by the per-share entry,
    # gives the max number of shares that stays within the participation rate.
    # Purely informational; never feeds the actionable gate. Null when ADV is
    # missing or the cap is disabled (max_participation_pct <= 0).
    max_units = pd.Series(pd.NA, index=out.index, dtype="Float64")
    if max_participation_pct > 0 and "adv_vnd_20" in out.columns:
        adv_vnd = out["adv_vnd_20"].astype(float) * 1000.0
        budget = (max_participation_pct / 100.0) * adv_vnd
        units = (budget / entry_v).where((entry_v > 0) & adv_vnd.notna())
        max_units = np.floor(units)

    out["entry_vnd"] = entry_v.astype("Int64")
    out["close_vnd"] = close_v.astype("Int64")
    out["entry_limit_pct"] = pred_low_eff.round(6)
    out["target_vnd"] = target_v.astype("Int64")
    out["target_low_vnd"] = target_low_v.astype("Int64")
    out["target_high_vnd"] = target_high_v.astype("Int64")
    out["stop_vnd"] = stop_v.astype("Int64")
    out["gross_reward_vnd"] = gross_reward.round(0).astype("Int64")
    out["max_loss_vnd"] = max_loss_units.round(0).astype("Int64")
    out["fees_round_trip_vnd"] = fees_total.round(0).astype("Int64")
    out["net_reward_vnd"] = net_reward.round(0).astype("Int64")
    out["net_loss_vnd"] = net_loss.round(0).astype("Int64")
    out["rr_ratio"] = rr.round(2)
    out["breakeven_pct"] = breakeven_pct
    out["actionable"] = actionable
    out["suggested_max_units"] = pd.array(max_units, dtype="Int64")
    return out


def add_adjusted_price_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    """Append a PARALLEL set of ``adj_*`` columns derived from LLM-supplied
    entry / target overrides — **without** touching any of the mechanical
    columns produced by :func:`add_price_suggestions`.

    Motivation: the mechanical ``entry_vnd`` is a per-ticker dip limit that the
    news stage cannot move. On a news-driven melt-up (e.g. a macro catalyst
    lifting the whole market), that dip never comes and the limit never fills.
    This lets the Claude / Gemini news stage quote its OWN entry and target —
    informed by the catalyst — and surfaces the matching risk-reward so the
    user can compare the news-adjusted trade against the mechanical one
    side by side.

    Inputs (optional, in VND per share; produced by the news parsers):
        adj_entry_vnd   — limit/entry price the LLM wants placed. Unlike the
                          mechanical entry, this is NOT clipped at the close:
                          on a strong catalyst the LLM may quote an entry ABOVE
                          today's close to guarantee a fill.
        adj_target_vnd  — exit target the LLM wants.

    When either is missing / NaN for a row, that row's ``adj_*`` outputs mirror
    the mechanical values (``entry_vnd`` / ``target_vnd``), so the columns are
    always fully populated and an un-adjusted pick reads identically to its
    mechanical twin.

    Output columns appended (all VND per share unless noted), each the
    ``adj_`` twin of a mechanical column:
        adj_entry_vnd, adj_target_vnd  (echoed back, NaN-filled to mechanical)
        adj_stop_vnd                   stop anchored on the adjusted entry
        adj_gross_reward_vnd, adj_max_loss_vnd
        adj_fees_round_trip_vnd
        adj_net_reward_vnd, adj_net_loss_vnd
        adj_rr_ratio
        adj_breakeven_pct
        adj_actionable
    """
    if df is None or len(df) == 0:
        return df

    cfg = load_config()
    broker = dict(cfg.broker) if hasattr(cfg, "broker") else {}
    pricing_cfg = dict(cfg.pricing) if hasattr(cfg, "pricing") else {}
    stop_mult = float(pricing_cfg.get("stop_atr_mult", 1.5))
    min_rr = float(pricing_cfg.get("min_rr_ratio", 0.8))

    out = df.copy()

    # Mechanical anchors to fall back on (these must already exist; if a
    # legacy frame lacks them, default to NaN so the adj columns are NaN too).
    mech_entry = out.get("entry_vnd", pd.Series(np.nan, index=out.index)).astype("float64")
    mech_target = out.get("target_vnd", pd.Series(np.nan, index=out.index)).astype("float64")
    atr_k = out.get("atr_14", pd.Series(np.nan, index=out.index)).astype(float)

    # LLM overrides — when absent or NaN, mirror the mechanical price.
    adj_entry = out.get("adj_entry_vnd", pd.Series(np.nan, index=out.index)).astype("float64")
    adj_target = out.get("adj_target_vnd", pd.Series(np.nan, index=out.index)).astype("float64")
    adj_entry_v = adj_entry.where(adj_entry.notna(), mech_entry).round(0)
    adj_target_v = adj_target.where(adj_target.notna(), mech_target).round(0)

    # Stop anchored on the ADJUSTED entry, same rule as the mechanical stop:
    # risk distance is exactly stop_atr_mult * ATR below the entry.
    adj_entry_k = adj_entry_v / 1000.0
    adj_stop_v = ((adj_entry_k - stop_mult * atr_k) * 1000.0).round(0)

    adj_gross_reward = adj_target_v - adj_entry_v
    adj_max_loss = adj_entry_v - adj_stop_v

    _, _, adj_fees_total, _ = _broker_costs(adj_entry_v, adj_target_v, broker)
    adj_net_reward = adj_gross_reward - adj_fees_total
    adj_net_loss = adj_max_loss + adj_fees_total

    adj_rr = pd.Series(np.nan, index=out.index, dtype=float)
    valid = (adj_max_loss > 0) & adj_net_loss.notna()
    adj_rr[valid] = adj_net_reward[valid] / adj_net_loss[valid]

    adj_breakeven_pct = (adj_fees_total / adj_entry_v).round(4)
    adj_actionable = (adj_net_reward > 0) & (adj_rr >= min_rr) & valid

    out["adj_entry_vnd"] = adj_entry_v.astype("Int64")
    out["adj_target_vnd"] = adj_target_v.astype("Int64")
    out["adj_stop_vnd"] = adj_stop_v.astype("Int64")
    out["adj_gross_reward_vnd"] = adj_gross_reward.round(0).astype("Int64")
    out["adj_max_loss_vnd"] = adj_max_loss.round(0).astype("Int64")
    out["adj_fees_round_trip_vnd"] = adj_fees_total.round(0).astype("Int64")
    out["adj_net_reward_vnd"] = adj_net_reward.round(0).astype("Int64")
    out["adj_net_loss_vnd"] = adj_net_loss.round(0).astype("Int64")
    out["adj_rr_ratio"] = adj_rr.round(2)
    out["adj_breakeven_pct"] = adj_breakeven_pct
    out["adj_actionable"] = adj_actionable
    return out
