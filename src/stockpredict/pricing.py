"""Translate ML predictions into actionable buy/target/stop prices, sized for
the user's actual trade (default 100 units) and net of broker fees.

vnstock prices are in **thousand VND** (e.g. close=15.35 means 15,350 VND).
We expose all suggestion columns in absolute VND (integer) since that's how
Vietnamese traders enter orders in their broker app.

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


def add_price_suggestions(df: pd.DataFrame, units: int | None = None) -> pd.DataFrame:
    """Append entry / target / stop / fees / net P&L columns to a candidates frame.

    Required input columns (already produced by the feature pipeline):
        close      — today's close in thousand VND
        pred_mean  — predicted T+2 return (e.g. +0.0017 = +0.17%)
        pred_std   — model dispersion across the seed ensemble
        atr_14     — 14-day ATR in thousand VND

    Output columns appended (all VND, integer where applicable):
        position_units, position_value_vnd
        entry_vnd, target_vnd, target_low_vnd, target_high_vnd, stop_vnd
        gross_reward_vnd      target - entry, scaled by position_units
        max_loss_vnd          entry - stop, scaled by position_units
        fees_round_trip_vnd   buy commission+VAT + sell commission+VAT + sell PIT
        net_reward_vnd        gross_reward - fees   (the headline number)
        net_loss_vnd          max_loss + fees       (worst-case if stopped out)
        rr_ratio              net_reward / net_loss
        breakeven_pct         price move needed just to cover fees
        actionable            net_reward > 0 AND rr_ratio >= min_rr_ratio
    """
    if df is None or len(df) == 0:
        return df

    cfg = load_config()
    broker = dict(cfg.broker) if hasattr(cfg, "broker") else {}
    pricing_cfg = dict(cfg.pricing) if hasattr(cfg, "pricing") else {}

    pos_units = int(units) if units is not None else int(broker.get("default_position_units", 100))
    lot = int(broker.get("lot_size", 100))
    # ACBS / VN exchange rule: minimum order is one lot (100 shares).
    # Round down to nearest whole lot, but never below `lot`.
    pos_units = max(lot, (pos_units // lot) * lot)
    stop_mult = float(pricing_cfg.get("stop_atr_mult", 1.5))
    min_rr = float(pricing_cfg.get("min_rr_ratio", 0.8))

    out = df.copy()

    close_k = out["close"].astype(float)
    pred = out["pred_mean"].astype(float)
    pred_std = out.get("pred_std", pd.Series(0.0, index=out.index)).astype(float)
    atr_k = out.get("atr_14", pd.Series(np.nan, index=out.index)).astype(float)

    # Per-share prices in VND
    entry_v = (close_k * 1000.0).round(0)
    target_v = (close_k * (1.0 + pred) * 1000.0).round(0)
    target_low_v = (close_k * (1.0 + pred - pred_std) * 1000.0).round(0)
    target_high_v = (close_k * (1.0 + pred + pred_std) * 1000.0).round(0)
    stop_v = ((close_k - stop_mult * atr_k) * 1000.0).round(0)

    # Position-level P&L (sized at pos_units)
    position_v = entry_v * pos_units
    target_position_v = target_v * pos_units
    gross_reward = (target_v - entry_v) * pos_units
    max_loss_units = (entry_v - stop_v) * pos_units

    buy_fee, sell_fee, fees_total, _ = _broker_costs(position_v, target_position_v, broker)
    net_reward = gross_reward - fees_total
    net_loss = max_loss_units + fees_total

    # rr_ratio: net upside vs net downside, undefined when stop is missing/invalid
    rr = pd.Series(np.nan, index=out.index, dtype=float)
    valid = (max_loss_units > 0) & net_loss.notna()
    rr[valid] = net_reward[valid] / net_loss[valid]

    breakeven_pct = (fees_total / position_v).round(4)
    actionable = (net_reward > 0) & (rr >= min_rr) & valid

    out["position_units"] = pos_units
    out["position_value_vnd"] = position_v.round(0).astype("Int64")
    out["entry_vnd"] = entry_v.astype("Int64")
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
    return out
