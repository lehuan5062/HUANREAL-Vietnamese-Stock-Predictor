"""Translate rebound predictions into buy / target prices, per share and net of
broker fees.

vnstock prices are in **thousand VND** (e.g. close=15.35 means 15,350 VND). We
expose all suggestion columns in absolute VND (integer) since that's how
Vietnamese traders enter orders in their broker app.

The rebound trade buys at the close (``close_vnd`` — there is no entry-price
prediction) and sells at ``target_vnd = close × (1 + pred_profit)``; the exit is
flexible (hold until the target). All P&L figures are per share — the user
decides their own position size.

ACBS fee model (default — override in config.yaml):
  buy  cost = trade_value * commission_pct * (1 + vat_pct/100)
  sell cost = trade_value * commission_pct * (1 + vat_pct/100) + trade_value * pit_pct
  total round-trip ~ 0.43% of trade value at ACBS's 0.15% commission.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config


def round_trip_cost_fraction(broker: dict | None = None) -> float:
    """ACBS round-trip fee as a fraction of trade value: buy + sell commission
    (each with VAT) + sell PIT = ``2*c*(1+v) + p`` (~0.0043).

    This is the single shared cost definition used by the pricing math, the
    conviction bar in ranking, and the rebound ``profit_threshold`` (so
    "profitable" always means "clears the same round-trip cost the pricing
    charges"). ``broker`` defaults to the configured broker block."""
    if broker is None:
        cfg = load_config()
        broker = dict(cfg.broker) if hasattr(cfg, "broker") else {}
    c = float(broker.get("commission_pct", 0.15)) / 100.0
    v = float(broker.get("vat_pct", 10)) / 100.0
    p = float(broker.get("pit_pct", 0.10)) / 100.0
    return 2.0 * c * (1.0 + v) + p


def profit_threshold(broker: dict | None = None, margin: float | None = None) -> float:
    """The rebound "profitable point" return bar: round-trip cost + a margin.

    A future close clears this bar when ``close[T+k]/close[T] - 1 >=`` this value
    — i.e. the position is profitable *after* fees plus ``margin``. ``margin``
    defaults to ``strategy.recovery.profit_margin`` in config."""
    if margin is None:
        cfg = load_config()
        strat = dict(getattr(cfg, "strategy", {}) or {})
        recovery = dict(strat.get("recovery", {}) or {})
        margin = float(recovery.get("profit_margin", 0.005))
    return round_trip_cost_fraction(broker) + float(margin)


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


def add_recovery_price_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    """Append rebound-strategy price / economics columns to a candidates frame.

    Buys at today's close and holds until the position first turns profitable
    (flexible exit — no ATR stop, no fixed horizon). Required input columns:
    ``close``, ``pred_profit`` (P), ``pred_days`` (N), ``pred_recovery_prob``.

    Output columns appended (VND per share where applicable):
        close_vnd            the buy price = today's close (no entry-price
                             prediction — you buy at the close)
        target_vnd           close * (1 + pred_profit) — the profit target
        score                P/N * recovery_prob — the ranking objective
        gross_reward_vnd, fees_round_trip_vnd, net_reward_vnd, breakeven_pct
        below_recovery_bar   True when the pick fails the quality bar
        suggested_max_units  advisory liquidity cap
    """
    if df is None or len(df) == 0:
        return df

    cfg = load_config()
    broker = dict(cfg.broker) if hasattr(cfg, "broker") else {}
    pricing_cfg = dict(cfg.pricing) if hasattr(cfg, "pricing") else {}
    strat = dict(getattr(cfg, "strategy", {}) or {})
    recovery_cfg = dict(strat.get("recovery", {}) or {})
    min_recovery_prob = float(recovery_cfg.get("min_recovery_prob", 0.0) or 0.0)
    max_participation_pct = float(pricing_cfg.get("max_participation_pct", 0.0))
    thr = profit_threshold(broker)

    out = df.copy()
    close_k = out["close"].astype(float)
    pred_profit = out.get("pred_profit", pd.Series(np.nan, index=out.index)).astype(float)
    pred_days = out.get("pred_days", pd.Series(np.nan, index=out.index)).astype(float)
    pred_prob = out.get("pred_recovery_prob", pd.Series(np.nan, index=out.index)).astype(float)

    # Buy at the close — there is NO entry-price prediction (the dip/limit head
    # was removed), so ``close_vnd`` is the single buy price.
    close_v = (close_k * 1000.0).round(0)
    target_v = (close_v * (1.0 + pred_profit.clip(lower=0.0))).round(0)
    if "score" in out.columns:
        # Caller (e.g. rank_today) already ranked by its own score -- possibly
        # volatility-penalized -- before calling here; preserve it rather than
        # silently recomputing a different (unpenalized) number under the same
        # column name. LLM-only picks (modes/claude.py::finalize_llm) never
        # carry a pre-existing "score", so they still get it computed below.
        score = out["score"].astype(float)
    else:
        # LLM-only picks carry no statistical recovery probability (the LLM's
        # DROP/selection vetting IS its probability judgement) — treat a missing
        # prob as 1.0 so score reduces to plain P/N and the prob gate is skipped.
        score = (pred_profit / pred_days.clip(lower=1.0)) * pred_prob.fillna(1.0)

    gross_reward = target_v - close_v
    _, _, fees_total, _ = _broker_costs(close_v, target_v, broker)
    net_reward = gross_reward - fees_total
    breakeven_pct = (fees_total / close_v).round(4)

    below_bar = (
        (pred_prob.fillna(1.0) < min_recovery_prob)
        | (net_reward <= 0)
        | (pred_profit <= thr)
    ).fillna(True)

    max_units = pd.Series(pd.NA, index=out.index, dtype="Float64")
    if max_participation_pct > 0 and "adv_vnd_20" in out.columns:
        adv_vnd = out["adv_vnd_20"].astype(float) * 1000.0
        budget = (max_participation_pct / 100.0) * adv_vnd
        units = (budget / close_v).where((close_v > 0) & adv_vnd.notna())
        max_units = np.floor(units)

    out["close_vnd"] = close_v.astype("Int64")
    out["target_vnd"] = target_v.astype("Int64")
    out["score"] = score.round(6)
    out["gross_reward_vnd"] = gross_reward.round(0).astype("Int64")
    out["fees_round_trip_vnd"] = fees_total.round(0).astype("Int64")
    out["net_reward_vnd"] = net_reward.round(0).astype("Int64")
    out["breakeven_pct"] = breakeven_pct
    out["below_recovery_bar"] = below_bar
    out["suggested_max_units"] = pd.array(max_units, dtype="Int64")
    return out


def add_adjusted_price_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    """Append a PARALLEL set of ``adj_*`` columns derived from LLM-supplied
    entry / target overrides — **without** touching the mechanical columns
    from :func:`add_recovery_price_suggestions`.

    Motivation: the mechanical buy is today's close and the target is
    ``close × (1 + pred_profit)``. If the LLM's research says a name will gap up
    or down on a catalyst, it can quote its OWN entry / target here, and this
    surfaces the matching economics so the user can compare side by side.

    Inputs (optional, in VND per share; produced by the news parsers):
        adj_entry_vnd   — buy price the LLM wants placed.
        adj_target_vnd  — exit target the LLM wants.

    When either is missing / NaN for a row, that row's ``adj_*`` outputs mirror
    the mechanical values (``close_vnd`` / ``target_vnd``), so the columns are
    always fully populated and an un-adjusted pick reads identically.

    Output columns appended (all VND per share). The rebound trade has NO
    stop-loss (flexible hold-until-target exit), so there is no adj stop / rr:
        adj_entry_vnd, adj_target_vnd  (echoed back, NaN-filled to close/target)
        adj_gross_reward_vnd
        adj_fees_round_trip_vnd
        adj_net_reward_vnd
        adj_breakeven_pct
    """
    if df is None or len(df) == 0:
        return df

    cfg = load_config()
    broker = dict(cfg.broker) if hasattr(cfg, "broker") else {}

    out = df.copy()

    # Mechanical anchors: the buy price is close_vnd (no entry-price prediction),
    # the target is target_vnd. When the LLM leaves adj_* blank, mirror these.
    mech_entry = out.get("close_vnd", pd.Series(np.nan, index=out.index)).astype("float64")
    mech_target = out.get("target_vnd", pd.Series(np.nan, index=out.index)).astype("float64")

    adj_entry = out.get("adj_entry_vnd", pd.Series(np.nan, index=out.index)).astype("float64")
    adj_target = out.get("adj_target_vnd", pd.Series(np.nan, index=out.index)).astype("float64")
    adj_entry_v = adj_entry.where(adj_entry.notna(), mech_entry).round(0)
    adj_target_v = adj_target.where(adj_target.notna(), mech_target).round(0)

    adj_gross_reward = adj_target_v - adj_entry_v
    _, _, adj_fees_total, _ = _broker_costs(adj_entry_v, adj_target_v, broker)
    adj_net_reward = adj_gross_reward - adj_fees_total
    adj_breakeven_pct = (adj_fees_total / adj_entry_v).round(4)

    out["adj_entry_vnd"] = adj_entry_v.astype("Int64")
    out["adj_target_vnd"] = adj_target_v.astype("Int64")
    out["adj_gross_reward_vnd"] = adj_gross_reward.round(0).astype("Int64")
    out["adj_fees_round_trip_vnd"] = adj_fees_total.round(0).astype("Int64")
    out["adj_net_reward_vnd"] = adj_net_reward.round(0).astype("Int64")
    out["adj_breakeven_pct"] = adj_breakeven_pct
    return out
