"""Portfolio simulation: exit on the model's PREDICTED exit day/price.

Same buy side as rebound_sim_include_held (buys the daily #1 pick, held
tickers included), but a different EXIT rule: instead of selling whenever
profitable from T+2 onwards, this sells on the exact day the model predicted
(exit_date) at the predicted price (exit_price). It's a diagnostic for whether
the model's day/price predictions are any good, independent of the
opportunistic sell-whenever-profitable heuristic the other sims use.

    python -m scripts.rebound_sim_predicted_exit
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
    return (lambda val: val * c * (1 + v),                 # buy fee
            lambda val: val * c * (1 + v) + val * p)       # sell fee


def _lot(sh):
    return int(sh // 100 * 100)


def _build_data():
    from stockpredict.dataset import build_panel
    panel = build_panel(require_target=True)
    per_day, paths = _daily_candidates(panel, START)
    return per_day, paths, sorted(per_day.keys())


def simulate_strict_exit(data=None):
    """Exit strictly on the predicted exit_date at the predicted exit_price."""
    per_day, paths, days = data if data is not None else _build_data()
    if not days:
        return {
            "execution": "strict_exit_on_predicted_day", "span": "no eligible days",
            "signals": 0, "missed_fills_gap_up": 0, "fill_rate": float("nan"),
            "total_trades": 0, "predicted_day_exits": 0, "unsold_counted_as_loss": 0,
            "win_rate": float("nan"), "mean_hold_days": float("nan"),
            "total_capital_injected_VND": 0.0, "total_fees_paid_VND": 0.0,
            "final_value_VND": 0.0, "total_profit_VND": 0.0,
            "profit_on_injected": float("nan"), "annualized_IRR": float("nan"),
            "book_max_drawdown": float("nan"),
        }
    buy_fee, sell_fee = _fees()

    cash = 0.0
    injected = 0.0
    fees_paid = 0.0
    n_signals = 0
    n_missed_fills = 0
    n_predicted_exits = 0
    n_early_exits = 0  # exits that hit before predicted day
    positions = []      # {sym, shares, entry, entry_j, entry_date, exit_date, exit_price}
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
        # 1) EXITS — only on the predicted exit_date, at the predicted exit_price.
        still = []
        for pos in positions:
            if date == pos["exit_date"]:
                # Try to exit on the predicted day at the predicted price
                exit_price = pos["exit_price"]
                val = pos["shares"] * exit_price
                fee = sell_fee(val)
                cash += val - fee
                fees_paid += fee
                trades.append({"sym": pos["sym"], "entry": pos["entry"],
                               "exit": exit_price, "shares": pos["shares"],
                               "hold": (date - pos["entry_date"]).days,
                               "ret": exit_price / pos["entry"] - 1.0,
                               "reason": "predicted_exit"})
                n_predicted_exits += 1
            else:
                still.append(pos)
        positions = still

        # 2) BUY the day's #1 pick.
        cands = per_day.get(date, [])
        if cands:
            c = cands[0]
            sym, signal_close, mu = c["symbol"], c["close"], c["max_units"]
            exit_date = c.get("exit_date")  # Predicted exit date
            exit_price = c.get("exit_price")  # Predicted exit price
            n_signals += 1

            entry = None
            entry_j = None
            nxt_open, nxt_low = c.get("next_open"), c.get("next_low")
            if nxt_open is None:
                pass
            elif nxt_low > signal_close:
                n_missed_fills += 1
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
                                      "entry_j": entry_j, "entry_date": entry_date,
                                      "exit_date": exit_date, "exit_price": exit_price})

        # Mark the book
        book = cash + sum(p["shares"] * (bar_on(date, p["sym"])[1] or p["entry"])
                          for p in positions)
        equity.append((date, book))

    # End of run: anything still open is force-closed at last close as a loss.
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

    # Money-weighted IRR
    t0 = days[0]
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
    total = n_predicted_exits + n_forced
    return {
        "execution": "strict_exit_on_predicted_day",
        "span": f"{days[0].date()}..{days[-1].date()} (~{(days[-1]-days[0]).days/365:.1f}y)",
        "signals": n_signals,
        "missed_fills_gap_up": n_missed_fills,
        "fill_rate": (1 - n_missed_fills / n_signals) if n_signals else float("nan"),
        "total_trades": total,
        "predicted_day_exits": n_predicted_exits,
        "unsold_counted_as_loss": n_forced,
        "win_rate": (n_predicted_exits / total) if total else float("nan"),
        "mean_hold_days": float(tdf.loc[tdf.get("reason").ne("unsold_loss"), "hold"].mean()) if n_predicted_exits else float("nan"),
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
    out = simulate_strict_exit(data=data)
    print(json.dumps(out, indent=2, default=str))
