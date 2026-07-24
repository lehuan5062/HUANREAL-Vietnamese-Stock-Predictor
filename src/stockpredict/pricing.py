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
    """The rebound/momentum "profitable point" return bar: round-trip cost + a
    margin.

    A future close clears this bar when ``close[T+k]/close[T] - 1 >=`` this value
    — i.e. the position is profitable *after* fees plus ``margin``. ``margin``
    defaults to ``pricing.profit_margin`` in config."""
    if margin is None:
        cfg = load_config()
        pricing_cfg = dict(cfg.pricing) if hasattr(cfg, "pricing") else {}
        margin = float(pricing_cfg.get("profit_margin", 0.005))
    return round_trip_cost_fraction(broker) + float(margin)


def settle_days() -> int:
    """Earliest tradable exit day (T+2 settlement) — shares bought today can't
    be sold before this many trading days later. See ``pricing.settle_days``."""
    cfg = load_config()
    pricing_cfg = dict(cfg.pricing) if hasattr(cfg, "pricing") else {}
    return int(pricing_cfg.get("settle_days", 2))


def resolve_exit(future_close, entry: float, thr: float,
                 min_hold_days: int = 1) -> dict | None:
    """Resolve a momentum/rebound trade's exit by walking its forward close
    path (plain price arithmetic — no model involved).

    ``future_close`` is the sequence of closes AFTER the entry bar, in
    chronological order (offset k = 1, 2, ...). The trade exits on the first day
    at or after ``min_hold_days`` whose close clears the profitable point
    (``close >= entry*(1 + thr)``) — ``min_hold_days`` embargoes earlier days
    that aren't actually sellable (T+2 settlement); pass ``settle_days()`` to
    match live pricing.

    Returns ``{"reason": "recovery", "k" (1-based day), "exit_close"}`` for that
    day, or ``None`` when the trade is still open (never recovered within the
    available data).

    A single-session close-to-close move larger than the widest exchange band
    (UPCOM 15%) is physically impossible without a corporate action, so the
    scan censors (returns ``None``, leaving the trade open) at such a bar
    rather than mistaking a phantom jump for a real recovery."""
    import numpy as _np
    _BAND_BREAK = 0.15
    fc = _np.asarray(future_close, dtype=float)
    n = fc.size
    if n == 0 or entry <= 0:
        return None
    recov = entry * (1.0 + float(thr))
    prev = float(entry)
    for k in range(1, n + 1):
        c = float(fc[k - 1])
        if prev > 0 and abs(c / prev - 1.0) > _BAND_BREAK:
            return None
        prev = c
        if k >= max(1, int(min_hold_days)) and c >= recov:
            return {"reason": "recovery", "k": k, "exit_close": c}
    return None


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
    """Append momentum/rebound price / economics columns to a candidates frame.

    Shared by the momentum and rebound modes: buys at today's close and holds
    until the position first turns profitable (flexible exit — no ATR stop, no
    fixed horizon). Required input columns: ``close``, ``pred_profit`` (P),
    ``pred_days`` (N) — both supplied by the LLM agent's own research; there is
    no statistical recovery-probability model any more (the agent's selection
    judgement stands in for it), so ``score`` reduces to plain P/N.

    Output columns appended (VND per share where applicable):
        close_vnd            the buy price = today's close (no entry-price
                             prediction — you buy at the close)
        target_vnd           close * (1 + pred_profit) — the profit target
        score                P/N — the ranking objective
        gross_reward_vnd, fees_round_trip_vnd, net_reward_vnd, breakeven_pct
        below_recovery_bar   True when the pick fails the quality bar (net
                             reward <= 0 or P doesn't clear the round-trip cost)
    """
    if df is None or len(df) == 0:
        return df

    cfg = load_config()
    broker = dict(cfg.broker) if hasattr(cfg, "broker") else {}
    thr = profit_threshold(broker)

    out = df.copy()
    close_k = out["close"].astype(float)
    pred_profit = out.get("pred_profit", pd.Series(np.nan, index=out.index)).astype(float)
    pred_days = out.get("pred_days", pd.Series(np.nan, index=out.index)).astype(float)

    # Buy at the close — there is NO entry-price prediction, so ``close_vnd``
    # is the single buy price.
    close_v = (close_k * 1000.0).round(0)
    target_v = (close_v * (1.0 + pred_profit.clip(lower=0.0))).round(0)
    score = (pred_profit / pred_days.clip(lower=1.0))

    gross_reward = target_v - close_v
    _, _, fees_total, _ = _broker_costs(close_v, target_v, broker)
    net_reward = gross_reward - fees_total
    breakeven_pct = (fees_total / close_v).round(4)

    below_bar = (
        (net_reward <= 0)
        | (pred_profit <= thr)
    ).fillna(True)

    out["close_vnd"] = close_v.astype("Int64")
    out["target_vnd"] = target_v.astype("Int64")
    out["score"] = score.round(6)
    out["gross_reward_vnd"] = gross_reward.round(0).astype("Int64")
    out["fees_round_trip_vnd"] = fees_total.round(0).astype("Int64")
    out["net_reward_vnd"] = net_reward.round(0).astype("Int64")
    out["breakeven_pct"] = breakeven_pct
    out["below_recovery_bar"] = below_bar
    return out


def add_dividend_price_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    """Append dividend-strategy price columns to a candidates frame.

    The dividend mode is a HOLD, not a swing trade: buy at today's close, no
    profit target, no stop-loss, no T+N. Required input column: ``close``.

    Output columns appended (VND per share):
        close_vnd           the buy price = today's close
        fees_buy_vnd         one-sided buy commission + VAT (no sell leg yet —
                             the position is a hold, not a round trip)
    """
    if df is None or len(df) == 0:
        return df

    cfg = load_config()
    broker = dict(cfg.broker) if hasattr(cfg, "broker") else {}
    commission = float(broker.get("commission_pct", 0.15)) / 100.0
    vat = float(broker.get("vat_pct", 10)) / 100.0
    min_fee = float(broker.get("min_fee_vnd", 0))

    out = df.copy()
    close_v = (out["close"].astype(float) * 1000.0).round(0)
    buy_commission = (close_v * commission).clip(lower=min_fee)
    buy_fee = (buy_commission * (1.0 + vat)).round(0)

    out["close_vnd"] = close_v.astype("Int64")
    out["fees_buy_vnd"] = buy_fee.astype("Int64")
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
