"""Portfolio simulation: buy the daily #1 rebound pick, held tickers INCLUDED.

Held tickers are NOT excluded from the candidate pool — so if the #1-ranked
pick is a name you already own, you buy it again (add to the position).

Rules:
  * BUY DAILY   — every trading day, target that day's #1-ranked rebound pick,
                  **even if already held**.
  * EXECUTION   — pre-open limit at the signal day's close, next morning.
                  Filled only if next day's low <= signal close.
  * FLOOR 100   — always buy ≥100 shares; inject capital if needed.
  * REINVEST    — proceeds from sells recycled into new buys first.
  * LOTS OF 100 — buy multiples of 100 (ACBS board lot).
  * LIQUIDITY   — never buy more than the pick's suggested_max_units.
  * EXIT        — T+2 minimum hold, then sell on first profitable close.

Entry price = min(next_open, signal_close) if next_low <= signal_close; else no fill.
Fees explicit at each leg. Unsold at end = loss.

    python -m scripts.rebound_sim_include_held
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scripts.rebound_portfolio_sim import _daily_candidates
from stockpredict.config import load_config
from stockpredict.pricing import profit_threshold

START = "2024-01-01"
SETTLE_DAYS = 2


def _fees():
    b = dict(load_config().broker)
    c = float(b["commission_pct"]) / 100.0
    v = float(b["vat_pct"]) / 100.0
    p = float(b["pit_pct"]) / 100.0
    return (lambda val: val * c * (1 + v),
            lambda val: val * c * (1 + v) + val * p)


def _lot(sh):
    return int(sh // 100 * 100)


def _build_data(start=START, end=None):
    from stockpredict.dataset import build_panel
    panel = build_panel(require_target=True)
    per_day, paths = _daily_candidates(panel, start, end)
    return per_day, paths, sorted(per_day.keys())


def simulate(data=None, start=START, end=None):
    """Simulate with hardcoded limit_next_day execution.

    No sub-modes — this sim has ONE rule set. ``start``/``end`` bound the
    backtest window when ``data`` is not supplied (default = the module START
    → last bar, i.e. identical to prior behavior).
    """
    per_day, paths, days = data if data is not None else _build_data(start, end)
    # IRR anchors: annualize over the FULL backtest window, not the first-to-
    # last-trade span. A config that only trades a few days out of the window
    # kept capital standing ready the whole time; annualizing a one-day round
    # trip over its own span produced absurd IRRs (observed: 57,340 = 5.7M%).
    # Clamped so the anchors always at least cover the traded days.
    w_start = pd.Timestamp(start) if start is not None else None
    w_end = pd.Timestamp(end) if end is not None else None
    if not days:
        # No anchor produced any candidate day in this window/config combo
        # (e.g. too little training data, or gates filtered every candidate
        # out). Degenerate result, not a crash.
        return {
            "span": f"{start}..{end} (no eligible days)",
            "signals": 0, "missed_fills_gap_up": 0, "fill_rate": float("nan"),
            "total_trades": 0, "sold_at_profit": 0, "unsold_counted_as_loss": 0,
            "win_rate": float("nan"), "mean_hold_days_winners": float("nan"),
            "total_capital_injected_VND": 0.0, "total_fees_paid_VND": 0.0,
            "final_value_VND": 0.0, "total_profit_VND": 0.0,
            "profit_on_injected": float("nan"), "annualized_IRR": float("nan"),
            "book_max_drawdown": float("nan"),
        }
    thr = profit_threshold()
    buy_fee, sell_fee = _fees()

    cash = 0.0
    injected = 0.0
    fees_paid = 0.0
    n_signals = 0
    n_missed_fills = 0
    positions = []
    cashflows = []
    trades = []
    equity = []

    def bar_on(date, sym):
        idx_arr, _o, _l, close_arr = paths[sym]
        j = int(np.searchsorted(idx_arr, np.datetime64(pd.Timestamp(date))))
        if j < len(idx_arr) and idx_arr[j] == np.datetime64(pd.Timestamp(date)):
            return j, float(close_arr[j])
        return None, None

    def next_bar_index(date, sym):
        idx_arr = paths[sym][0]
        fut = np.where(idx_arr > np.datetime64(pd.Timestamp(date)))[0]
        return int(fut[0]) if fut.size else None

    for date in days:
        # 1) SELLS — from T+2 after entry, exit on first profitable close.
        still = []
        for pos in positions:
            j, c_t = bar_on(date, pos["sym"])
            if j is not None and j >= pos["entry_j"] + SETTLE_DAYS \
                    and c_t >= pos["entry"] * (1.0 + thr):
                val = pos["shares"] * c_t
                fee = sell_fee(val)
                cash += val - fee
                fees_paid += fee
                trades.append({"sym": pos["sym"], "entry": pos["entry"],
                               "exit": c_t, "shares": pos["shares"],
                               "hold": j - pos["entry_j"],
                               "ret": c_t / pos["entry"] - 1.0,
                               "reason": "recovery"})
            else:
                still.append(pos)
        positions = still

        # 2) BUY — day's #1 pick, limit_next_day execution.
        cands = per_day.get(date, [])
        if cands:
            c = cands[0]
            sym, signal_close, mu = c["symbol"], c["close"], c["max_units"]
            n_signals += 1

            nxt_open, nxt_low = c.get("next_open"), c.get("next_low")
            entry = None
            entry_j = None

            if nxt_open is None:
                pass  # no next bar yet
            elif nxt_low > signal_close:
                n_missed_fills += 1  # gap-up: no fill
            else:
                entry = min(nxt_open, signal_close)
                entry_j = next_bar_index(date, sym)
                entry_date = pd.Timestamp(paths[sym][0][entry_j])

            if entry is not None and entry > 0 and entry_j is not None:
                aff = _lot(cash / entry)
                shares = max(aff, 100)
                if mu:
                    shares = min(shares, max(_lot(mu), 100))
                cost = shares * entry
                total = cost + buy_fee(cost)
                if cash < total:
                    injected += total - cash
                    cashflows.append((date, -(total - cash)))
                    cash = total
                cash -= total
                fees_paid += buy_fee(cost)
                positions.append({"sym": sym, "shares": shares, "entry": entry,
                                  "entry_j": entry_j, "entry_date": entry_date})

        # Mark the book.
        book = cash + sum(p["shares"] * (bar_on(date, p["sym"])[1] or p["entry"])
                          for p in positions)
        equity.append((date, book))

    # End: force-close unsold positions.
    n_wins = len(trades)
    last = days[-1]
    liq = 0.0
    for p in positions:
        _, c_t = bar_on(last, p["sym"])
        c_t = c_t or p["entry"]
        val = p["shares"] * c_t
        liq += val - sell_fee(val)
        trades.append({"sym": p["sym"], "entry": p["entry"], "exit": c_t,
                       "shares": p["shares"], "hold": (last - p["entry_date"]).days,
                       "ret": c_t / p["entry"] - 1.0, "reason": "unsold_loss"})
    n_forced = len(positions)
    final_val = cash + liq
    cashflows.append((last, final_val))

    # Money-weighted IRR (daily), annualized over the backtest window.
    t0 = min(w_start, days[0]) if w_start is not None else days[0]
    t_final = max(w_end, days[-1]) if w_end is not None else days[-1]
    # Final valuation sits at the window end: after the last trade the book is
    # cash earning nothing, which correctly dilutes the annualized rate.
    cashflows[-1] = (t_final, cashflows[-1][1])
    def npv(r):
        return sum(a / (1 + r) ** ((d - t0).days) for d, a in cashflows)
    lo, hi = -0.99, 2.0
    for _ in range(200):
        mid = (lo + hi) / 2
        (lo, hi) = (mid, hi) if npv(mid) > 0 else (lo, mid)
    ann = (1 + (lo + hi) / 2) ** 365 - 1

    eq = pd.Series({d: v for d, v in equity}).sort_index()
    max_dd = float((eq / eq.cummax() - 1).min())
    tdf = pd.DataFrame(trades)
    total = n_wins + n_forced
    return {
        "span": f"{days[0].date()}..{days[-1].date()} (~{(days[-1]-days[0]).days/365:.1f}y)",
        # Marker read by rebound_config_suggest: records WITHOUT it were
        # annualized over the traded span (pre-fix) and get filtered when short.
        "irr_anchor": "window",
        "signals": n_signals,
        "missed_fills_gap_up": n_missed_fills,
        "fill_rate": (1 - n_missed_fills / n_signals) if n_signals else float("nan"),
        "total_trades": total,
        "sold_at_profit": n_wins,
        "unsold_counted_as_loss": n_forced,
        "win_rate": (n_wins / total) if total else float("nan"),
        "mean_hold_days_winners": float(tdf.loc[tdf.get("reason").ne("unsold_loss"), "hold"].mean()) if n_wins else float("nan"),
        "total_capital_injected_VND": injected * 1000,
        "total_fees_paid_VND": fees_paid * 1000,
        "final_value_VND": final_val * 1000,
        "total_profit_VND": (final_val - injected) * 1000,
        "profit_on_injected": (final_val - injected) / injected if injected else float("nan"),
        "annualized_IRR": ann,
        "book_max_drawdown": max_dd,
    }


if __name__ == "__main__":
    import json
    data = _build_data()
    result = simulate(data=data)
    print(json.dumps(result, indent=2, default=str))
