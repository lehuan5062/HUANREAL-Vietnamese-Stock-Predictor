"""FINAL event-driven portfolio simulation of the rebound strategy.

Trading rules (as specified by the user):
  * BUY DAILY   — every trading day, target that day's #1-ranked rebound pick.
  * EXECUTION   — the order is placed the NEXT morning BEFORE the open
                  ("lệnh trước giờ") as a LIMIT at the signal day's close:
                    - next day's low  > limit  ->  NO FILL (gap-up, no trade)
                    - next day opens <= limit  ->  filled at the OPEN (ATO
                      price improvement)
                    - otherwise                ->  filled at the limit
                  i.e. filled iff low[T+1] <= close[T];
                       entry = min(open[T+1], close[T]).
  * FLOOR 100   — always buy at least 100 shares; if cash can't cover 100,
                  the user tops up (capital injected) to reach the floor.
  * REINVEST    — proceeds from sells are recycled into new buys first.
  * LOTS OF 100 — buy only in multiples of 100 shares (ACBS board lot).
  * LIQUIDITY   — never buy more than the pick's `suggested_max_units`.
  * EXIT        — you cannot sell before T+2 from the ENTRY day (settlement);
                  from then on, SELL IMMEDIATELY on the first close that is
                  profitable net of BOTH fees (close/entry - 1 >= the
                  profit_threshold = round-trip fee + margin).

Fees are charged explicitly at each leg:
  buy_fee  = value * commission*(1+VAT)
  sell_fee = value * commission*(1+VAT) + value*PIT
The same ACBS parameters drive the prediction code's profit_threshold, so the
sim's sell rule and the model's "profitable point" agree by construction.

Anything still open at the end is force-closed at the last close and counted
as a LOSS, so the win rate is honest. Headline is a money-weighted IRR on all
injected capital.

Run compares the realistic limit-fill execution against a lookahead baseline
(fill at the signal close — NOT achievable, since the signal is computed from
that close and the close is only known after the market closes) and against
FOMO (chase the next open). The realistic limit is the true ceiling:

    python -m scripts.rebound_final_sim
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
SETTLE_DAYS = 2   # T+2: earliest sellable session after the ENTRY day


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


def simulate(execution: str = "limit_next_day", data=None):
    """``execution``:
      * ``"limit_next_day"`` (the user's real method) — pre-open limit at the
        signal close, next day; may not fill (see module docstring).
      * ``"signal_close"`` — LOOKAHEAD BASELINE, not achievable. Fills at the
        signal day's close unconditionally. The signal is COMPUTED from that
        close (RSI/momentum/rank all need it), and the close is only known
        after the market closes — so you can never trade at the very close you
        predicted on. This model quietly assumes you both know the close AND
        trade at it, which is impossible in time. Kept only to measure how
        much the execution constraint costs (the gap to the real limit model).
      * ``"market_next_open"`` — FOMO: unconditional market/ATO buy at the
        next day's open, paying whatever it gapped to. Kept as a cautionary
        comparison: chasing the open destroys the small per-trade edge
        (measured ~1.3%/yr vs 17.8% for the real limit; the ~36% signal-close
        figure is the unachievable lookahead baseline above)."""
    per_day, paths, days = data if data is not None else _build_data()
    thr = profit_threshold()
    buy_fee, sell_fee = _fees()

    cash = 0.0          # thousand-VND
    injected = 0.0
    fees_paid = 0.0
    n_signals = 0
    n_missed_fills = 0
    positions = []      # {sym, shares, entry, entry_j, entry_date}
    cashflows = []      # (date, signed amount): <0 injection, >0 final value
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
        # 1) SELLS — from T+2 after the entry day, exit on first profitable close.
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

        # 2) BUY the day's #1 pick under the configured execution model.
        cands = per_day.get(date, [])
        if cands:
            c = cands[0]
            sym, signal_close, mu = c["symbol"], c["close"], c["max_units"]
            n_signals += 1
            entry = None
            entry_j = None
            if execution == "limit_next_day":
                nxt_open, nxt_low = c.get("next_open"), c.get("next_low")
                if nxt_open is None:
                    pass                        # no next bar yet — no order
                elif nxt_low > signal_close:
                    n_missed_fills += 1         # gap-up never came back: no fill
                else:
                    entry = min(nxt_open, signal_close)
                    entry_j = next_bar_index(date, sym)
                    entry_date = pd.Timestamp(paths[sym][0][entry_j])
            elif execution == "market_next_open":
                nxt_open = c.get("next_open")
                if nxt_open is not None and nxt_open > 0:
                    entry = nxt_open           # FOMO: pay the open, always fill
                    entry_j = next_bar_index(date, sym)
                    entry_date = pd.Timestamp(paths[sym][0][entry_j])
            elif execution == "signal_close":
                entry_j, _ = bar_on(date, sym)
                entry = signal_close if entry_j is not None else None
                entry_date = date
            else:
                raise ValueError(f"unknown execution model: {execution}")

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

        # mark the book
        book = cash + sum(p["shares"] * (bar_on(date, p["sym"])[1] or p["entry"])
                          for p in positions)
        equity.append((date, book))

    # End of run: everything still open counts as a LOSS — by the sell rule it
    # was never profitable-after-fees. Force-close at the last close.
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

    # Money-weighted IRR (daily), annualized, via bisection on NPV.
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
    total = n_wins + n_forced
    return {
        "execution": execution,
        "span": f"{days[0].date()}..{days[-1].date()} (~{(days[-1]-days[0]).days/365:.1f}y)",
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
    out = {
        "REALISTIC (pre-open limit at signal close)": simulate("limit_next_day", data=data),
        "LOOKAHEAD BASELINE (fill at signal close — NOT achievable)": simulate("signal_close", data=data),
        "FOMO (market order at next open — cautionary)": simulate("market_next_open", data=data),
    }
    print(json.dumps(out, indent=2, default=str))
