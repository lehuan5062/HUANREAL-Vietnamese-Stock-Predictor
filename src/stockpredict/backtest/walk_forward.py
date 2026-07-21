"""Walk-forward backtest of the rebound strategy (flexible exit).

Loop:
  for each anchor in [start, start + step, ...]:
      train the recovery estimator on [anchor - train_window, anchor)
      for each trading day t in [anchor, anchor + oos_window):
          filter to downtrend names, rank by P/N score, take top_k
          buy at close[t], hold until the close first clears the profit target
          (recovery) — or a stop / time cap if configured — record net return
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config, reports_dir
from ..dataset import build_panel
from ..filters import (ceiling_lock_mask, corporate_action_mask, downtrend_mask,
                       liquidity_mask, overbought_mask)
from ..model.predict import rebound_score
from ..model.target import resolve_exit, settle_days
from ..model.train import train_recovery
from ..pricing import profit_threshold


@dataclass
class BacktestResult:
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)


def _annualize_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std(ddof=0) == 0 or len(returns) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=0))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def run(panel: pd.DataFrame | None = None,
        start: str | None = None,
        end: str | None = None,
        top_k: int | None = None) -> BacktestResult:
    """Walk-forward backtest of the rebound strategy (flexible exit)."""
    cfg = load_config()
    bt = cfg.backtest
    start = pd.to_datetime(start or bt["start"])
    end = pd.to_datetime(end or dt.date.today().isoformat())
    top_k = top_k or int(bt["top_k"])
    train_years = int(bt["train_window_years"])
    oos_months = int(bt["oos_window_months"])
    step_months = int(bt["step_months"])
    cost = float(bt["cost_bps"]) / 10_000.0

    if panel is None:
        panel = build_panel(require_target=True)
    panel = panel.copy()
    panel = panel[panel.index <= end]

    return _run_rebound(panel, start, end, top_k, train_years,
                        oos_months, step_months, cost)


def _rebound_filters(day: pd.DataFrame) -> pd.DataFrame:
    """Apply the same gate cascade as ``eligible_universe`` to a single-date
    slice: liquidity → ceiling-lock → corporate-action → overbought → downtrend."""
    for mask in (liquidity_mask, ceiling_lock_mask, corporate_action_mask,
                 overbought_mask, downtrend_mask):
        if day.empty:
            return day
        day = day[mask(day)]
    return day


def _run_rebound(panel: pd.DataFrame, start, end, top_k, train_years,
                 oos_months, step_months, cost) -> BacktestResult:
    """Walk-forward backtest of the rebound strategy with a FLEXIBLE exit:
    buy the top-K by P/N score at the day's close, then sell on the first future
    day whose close first clears the profit threshold (recovery). A pick that
    never recovers before the data edge is marked open (not recovered) and
    marked-to-market at the last available close.

    CAVEAT (label lookahead): the recovery target labels are computed on each
    symbol's full price series, so the KM head trained at an anchor can peek
    slightly past it. OOS returns here are therefore mildly optimistic; treat
    the rebound-vs-legacy comparison as indicative, not exact."""
    thr = profit_threshold()
    cfg = load_config()
    min_prob = float(dict(getattr(cfg, "strategy", {}) or {})
                     .get("recovery", {}).get("min_recovery_prob", 0.0) or 0.0)

    # Per-symbol forward close path (sorted), for the flexible-exit sim.
    paths: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, g in panel.groupby("symbol"):
        gi = g.sort_index()
        paths[str(sym)] = (gi.index.to_numpy(), gi["close"].astype(float).to_numpy())

    anchors = []
    a = start
    while a < end:
        anchors.append(a)
        a = a + pd.DateOffset(months=step_months)

    trades_rows: list[dict] = []
    for anchor in anchors:
        train_start = anchor - pd.DateOffset(years=train_years)
        train_panel = panel[(panel.index >= train_start) & (panel.index < anchor)]
        if train_panel.empty:
            continue
        oos_end = min(anchor + pd.DateOffset(months=oos_months), end)
        oos_panel = panel[(panel.index >= anchor) & (panel.index < oos_end)]
        if oos_panel.empty:
            continue
        try:
            model = train_recovery(train_panel)
        except ValueError:
            continue

        for date, day_slice in oos_panel.groupby(level=0):
            day = _rebound_filters(day_slice.copy())
            if day.empty:
                continue
            preds = model.predict(day)
            day = day.assign(**preds)
            if min_prob > 0:
                day = day[day["pred_recovery_prob"] >= min_prob]
                if day.empty:
                    continue
            day["score"], _ = rebound_score(day, cfg)
            picks = day.sort_values("score", ascending=False).head(top_k)
            for sym, row in picks.iterrows():
                symbol = str(row.get("symbol", sym))
                entry = float(row["close"])
                idx_arr, close_arr = paths.get(symbol, (None, None))
                if idx_arr is None or entry <= 0:
                    continue
                d0 = np.datetime64(pd.Timestamp(date))
                fut = np.where(idx_arr > d0)[0]
                if fut.size == 0:
                    continue  # no forward bar to exit on
                fut_close = close_arr[fut]
                ex = resolve_exit(fut_close, entry, thr, min_hold_days=settle_days())
                if ex is not None:
                    reason = ex["reason"]
                    hold_days = int(ex["k"])
                    exit_close = float(ex["exit_close"])
                else:
                    # Still open at the data edge — mark-to-market at last close.
                    reason = "open"
                    hold_days = int(fut.size)
                    exit_close = float(fut_close[-1])
                recovered = reason == "recovery"
                gross = exit_close / entry - 1.0
                net = gross - cost
                trades_rows.append({
                    "date": date, "symbol": symbol,
                    "score": float(row["score"]),
                    "pred_days": float(row["pred_days"]),
                    "pred_profit": float(row["pred_profit"]),
                    "recovered": recovered,
                    "exit_reason": reason,
                    "hold_days": hold_days,
                    "actual": gross, "net": net,
                    "net_per_day": net / max(hold_days, 1),
                })

    trades = pd.DataFrame(trades_rows)
    if trades.empty:
        return BacktestResult(summary={"n_trades": 0, "warning": "no trades",
                                       "strategy": "rebound"})

    daily = (trades.groupby("date")["net"].mean().rename("daily_ret")
             .to_frame().sort_index())
    daily["equity"] = (1.0 + daily["daily_ret"]).cumprod()
    mean_hold = float(trades["hold_days"].mean())
    sharpe = _annualize_sharpe(daily["daily_ret"],
                               periods_per_year=252 / max(mean_hold, 1))

    reason_counts = trades["exit_reason"].value_counts().to_dict()
    n = len(trades)
    summary = {
        "strategy": "rebound",
        "n_trades": int(n),
        "recovery_rate": float(trades["recovered"].mean()),
        "open_rate": float(reason_counts.get("open", 0) / n),
        "exit_reasons": {k: int(v) for k, v in reason_counts.items()},
        "hit_rate": float((trades["actual"] > 0).mean()),
        "hit_rate_net": float((trades["net"] > 0).mean()),
        "mean_return": float(trades["actual"].mean()),
        "mean_return_net": float(trades["net"].mean()),
        "median_return": float(trades["actual"].median()),
        "mean_hold_days": mean_hold,
        "median_hold_days": float(trades["hold_days"].median()),
        "net_return_per_day": float(trades["net_per_day"].mean()),
        "sharpe_net": sharpe,
        "max_drawdown": _max_drawdown(daily["equity"]),
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "top_k": top_k,
        "cost_bps": int(round(cost * 10_000)),
        "note": ("recovery labels computed on full series -> OOS returns are "
                 "mildly optimistic (train-label lookahead near each anchor)"),
    }
    return BacktestResult(trades=trades, equity=daily, summary=summary)


def write_report(result: BacktestResult, name: str | None = None) -> Path:
    """Persist a markdown + parquet report; matplotlib equity curve PNG when available."""
    name = name or f"backtest_{dt.date.today().isoformat()}"
    out_dir = reports_dir() / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not result.trades.empty:
        result.trades.to_parquet(out_dir / "trades.parquet", index=False)
        result.equity.to_parquet(out_dir / "equity.parquet")

    plot_path = out_dir / "equity.png"
    try:
        if not result.equity.empty:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            result.equity["equity"].plot(ax=ax)
            ax.set_title("Equity curve (top-K basket, net of cost)")
            ax.set_ylabel("growth of $1")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
    except Exception:
        plot_path = None

    summary_md = ["# Backtest summary", ""]
    for k, v in result.summary.items():
        summary_md.append(f"- **{k}**: {v}")
    if plot_path and plot_path.exists():
        summary_md.append("")
        summary_md.append(f"![equity]({plot_path.name})")
    (out_dir / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")
    return out_dir
