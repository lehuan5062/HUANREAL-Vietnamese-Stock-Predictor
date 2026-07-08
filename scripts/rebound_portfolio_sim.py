"""Realistic event-driven portfolio simulation of the rebound strategy.

Unlike the walk-forward backtest (which just compounds average per-trade
returns into a fantasy equity curve with no cash account), this models the
user's ACTUAL trading rules:

  * No fixed starting capital. Seed by buying 100 shares of the #1 pick on each
    of the first `SEED_DAYS` (3) trading days -> `SEED_DAYS` rolling positions.
  * Keep up to `SLOTS` (= SEED_DAYS) positions. When one exits, its freed slot
    is refilled with the top-ranked pick NOT already held, using available cash.
  * Board-lot rules: buy only in multiples of 100 shares, minimum 100. If cash
    can't cover 100 shares, the user TOPS UP (capital injected) to reach 100.
  * Each position is capped at `suggested_max_units` (the liquidity cap), rounded
    down to a 100-lot.
  * Exit at recovery (hold until the profit target; no stop, no cap).

Because cash is injected on demand, the headline return is a money-weighted
IRR on all injected capital, with the final book marked at the last close.
Round-trip broker cost (config broker, ~0.43%) is charged per trade.

Assumptions (state + adjust as needed):
  * Buy and sell both at the day's CLOSE (same bar the model scores on).
  * When a slot frees, all currently-idle cash is allocated to that one slot
    (floor 100, cap suggested_max_units); leftover cash waits for the next slot.
  * "#1 pick" = highest-score eligible pick that day not already held.

Usage:  python scripts/rebound_portfolio_sim.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from stockpredict.backtest.walk_forward import _rebound_filters
from stockpredict.config import load_config
from stockpredict.dataset import build_panel
from stockpredict.model.target import resolve_exit
from stockpredict.model.train import train_recovery
from stockpredict.pricing import (add_recovery_price_suggestions,
                                  round_trip_cost_fraction)

START = "2024-01-01"
SEED_DAYS = 3
SLOTS = 3


def _daily_candidates(panel: pd.DataFrame, start: str) -> tuple[dict, dict]:
    """Return (per_day, paths):
      per_day[date] = list of candidate dicts (score-ranked) with
        {symbol, close, max_units, next_open, next_low,
         exit_date, exit_price, exit_reason}
      paths[symbol] = (index_np, open_np, low_np, close_np) forward price path.
    ``next_open`` / ``next_low`` are the NEXT trading day's bar (None when the
    signal day is the last bar) — used by execution models that place the buy
    order the following morning. Mirrors the walk-forward training/scoring loop."""
    cfg = load_config()
    bt = cfg.backtest
    start = pd.to_datetime(start)
    end = panel.index.max()
    train_years = int(bt["train_window_years"])
    oos_months = int(bt["oos_window_months"])
    step_months = int(bt["step_months"])
    from stockpredict.pricing import profit_threshold
    thr = profit_threshold()
    min_prob = float(dict(cfg.get("strategy", {}) or {})
                     .get("recovery", {}).get("min_recovery_prob", 0.0) or 0.0)

    paths = {}
    for sym, g in panel.groupby("symbol"):
        gi = g.sort_index()
        paths[str(sym)] = (gi.index.to_numpy(),
                           gi["open"].astype(float).to_numpy(),
                           gi["low"].astype(float).to_numpy(),
                           gi["close"].astype(float).to_numpy())

    def _exit_for(symbol, date, entry):
        idx_arr, _open_arr, _low_arr, close_arr = paths.get(
            symbol, (None, None, None, None))
        if idx_arr is None or entry <= 0:
            return (None, None, "open")
        fut = np.where(idx_arr > np.datetime64(pd.Timestamp(date)))[0]
        if fut.size == 0:
            return (None, None, "open")
        fc = close_arr[fut]
        ex = resolve_exit(fc, entry, thr)
        if ex is None:
            return (pd.Timestamp(idx_arr[fut[-1]]), float(fc[-1]), "open")
        return (pd.Timestamp(idx_arr[fut[ex["k"] - 1]]), float(ex["exit_close"]), ex["reason"])

    def _next_bar(symbol, date):
        idx_arr, open_arr, low_arr, _close_arr = paths.get(
            symbol, (None, None, None, None))
        if idx_arr is None:
            return (None, None)
        fut = np.where(idx_arr > np.datetime64(pd.Timestamp(date)))[0]
        if fut.size == 0:
            return (None, None)
        j = int(fut[0])
        return (float(open_arr[j]), float(low_arr[j]))

    anchors = []
    a = start
    while a < end:
        anchors.append(a)
        a = a + pd.DateOffset(months=step_months)

    per_day: dict = {}
    for anchor in anchors:
        tr_start = anchor - pd.DateOffset(years=train_years)
        tr = panel[(panel.index >= tr_start) & (panel.index < anchor)]
        if tr.empty:
            continue
        oos_end = min(anchor + pd.DateOffset(months=oos_months), end)
        oos = panel[(panel.index >= anchor) & (panel.index < oos_end)]
        if oos.empty:
            continue
        try:
            model = train_recovery(tr)
        except ValueError:
            continue
        for date, day_slice in oos.groupby(level=0):
            day = _rebound_filters(day_slice.copy())
            if day.empty:
                continue
            preds = model.predict(day)
            day = day.assign(**preds)
            if min_prob > 0:
                day = day[day["pred_recovery_prob"] >= min_prob]
                if day.empty:
                    continue
            day["score"] = ((day["pred_profit"] / day["pred_days"].clip(lower=1.0))
                            * day["pred_recovery_prob"])
            day = day.sort_values("score", ascending=False)
            priced = add_recovery_price_suggestions(day)
            cands = []
            for sym, r in priced.iterrows():
                symbol = str(r["symbol"])
                entry = float(r["close"])
                mu = r.get("suggested_max_units")
                mu = int(mu) if pd.notna(mu) else None
                ed, ep, er = _exit_for(symbol, date, entry)
                nxt_open, nxt_low = _next_bar(symbol, date)
                pred_days = r.get("pred_days")
                pred_days = float(pred_days) if pd.notna(pred_days) else None
                cands.append({"symbol": symbol, "close": entry, "max_units": mu,
                              "next_open": nxt_open, "next_low": nxt_low,
                              "exit_date": ed, "exit_price": ep, "exit_reason": er,
                              "pred_days": pred_days})
            per_day[pd.Timestamp(date)] = cands
    return per_day, paths


def _lot(shares: float) -> int:
    return int(shares // 100 * 100)


def simulate() -> dict:
    panel = build_panel(require_target=True)
    per_day, paths = _daily_candidates(panel, START)
    days = sorted(per_day.keys())
    rt = round_trip_cost_fraction()

    cash = 0.0            # thousand-VND
    injected = 0.0        # total capital supplied
    positions = {}        # symbol -> {shares, entry, exit_date, exit_price, reason}
    cashflows = []        # (date, amount) : negative = injection, positive = final value
    trades = []
    equity = []           # (date, book_value)

    def close_on(date):
        # map symbol->close for marking the book on `date`
        return {c["symbol"]: c["close"] for c in per_day.get(date, [])}

    seeded = 0
    for i, date in enumerate(days):
        # 1) EXITS due today.
        for sym in list(positions.keys()):
            p = positions[sym]
            if p["exit_date"] is not None and pd.Timestamp(p["exit_date"]) == date:
                proceeds = p["shares"] * p["exit_price"] - rt * p["shares"] * p["entry"]
                cash += proceeds
                trades.append({"symbol": sym, "entry": p["entry"], "exit": p["exit_price"],
                               "shares": p["shares"], "reason": p["reason"],
                               "ret": p["exit_price"] / p["entry"] - 1 - rt})
                del positions[sym]

        cands = [c for c in per_day.get(date, []) if c["symbol"] not in positions]

        # 2) SEED: first SEED_DAYS trading days -> buy 100 of that day's top pick.
        if seeded < SEED_DAYS and cands:
            c = cands[0]
            cost = 100 * c["close"]
            if cash < cost:
                injected += cost - cash
                cashflows.append((date, -(cost - cash)))
                cash = cost
            cash -= cost
            positions[c["symbol"]] = {"shares": 100, "entry": c["close"],
                                      "exit_date": c["exit_date"], "exit_price": c["exit_price"],
                                      "reason": c["exit_reason"]}
            seeded += 1
            cands = cands[1:]

        # 3) REFILL free slots with top unheld picks, reinvesting cash (+ top-up to floor).
        elif seeded >= SEED_DAYS:
            ci = 0
            while len(positions) < SLOTS and ci < len(cands):
                c = cands[ci]; ci += 1
                price = c["close"]
                # shares affordable with idle cash, else top up to the 100 floor.
                aff = _lot(cash / price) if price > 0 else 0
                shares = max(aff, 100)
                if c["max_units"]:
                    shares = min(shares, _lot(c["max_units"]))
                    shares = max(shares, 100)
                cost = shares * price
                if cash < cost:
                    injected += cost - cash
                    cashflows.append((date, -(cost - cash)))
                    cash = cost
                cash -= cost
                positions[c["symbol"]] = {"shares": shares, "entry": price,
                                          "exit_date": c["exit_date"], "exit_price": c["exit_price"],
                                          "reason": c["exit_reason"]}

        # 4) mark the book (cash + open positions at today's close where known).
        cmap = close_on(date)
        book = cash + sum(p["shares"] * cmap.get(sym, p["entry"]) for sym, p in positions.items())
        equity.append((date, book))

    # Liquidate remaining book at last close for the final money-weighted flow.
    last_date = days[-1]
    cmap = close_on(last_date)
    final_val = cash + sum(p["shares"] * cmap.get(sym, p["entry"]) for sym, p in positions.items())
    cashflows.append((last_date, final_val))

    # Money-weighted IRR (daily), annualized. Bisection on NPV.
    t0 = days[0]
    def npv(rate):
        return sum(amt / (1 + rate) ** ((d - t0).days) for d, amt in cashflows)
    lo, hi = -0.9, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    daily_irr = (lo + hi) / 2
    ann_irr = (1 + daily_irr) ** 365 - 1

    eq = pd.Series({d: v for d, v in equity}).sort_index()
    peak = eq.cummax()
    max_dd = float((eq / peak - 1).min())
    profit = final_val - injected
    tdf = pd.DataFrame(trades)
    span_days = (days[-1] - days[0]).days

    return {
        "span": f"{days[0].date()}..{days[-1].date()} (~{span_days/365:.1f}y)",
        "n_trades": len(trades),
        "win_rate": float((tdf["ret"] > 0).mean()) if len(tdf) else float("nan"),
        "total_capital_injected_VND": injected * 1000,
        "final_value_VND": final_val * 1000,
        "total_profit_VND": profit * 1000,
        "profit_on_injected": profit / injected if injected else float("nan"),
        "annualized_IRR": ann_irr,
        "book_max_drawdown": max_dd,
        "exit_mix": tdf["reason"].value_counts().to_dict() if len(tdf) else {},
    }


if __name__ == "__main__":
    import json
    res = simulate()
    print(json.dumps(res, indent=2, default=str))
