"""Walk-forward backtest of the top-K daily picks.

Loop:
  for each anchor in [start, start + step, ...]:
      train on [anchor - train_window, anchor)
      for each trading day t in [anchor, anchor + oos_window):
          score all eligible symbols
          buy top_k at close[t], sell at close[t+exit_offset]
          record realized return (minus cost)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config, reports_dir
from ..dataset import build_panel
from ..filters import liquidity_mask
from ..model.train import train


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
        top_k: int | None = None,
        weights_fn=None) -> BacktestResult:
    """Walk-forward backtest. ``weights_fn`` (optional) maps a train panel to
    per-row sample weights, so the missed-winners retrain variant can be A/B'd
    against the standard fit; ``None`` = standard training."""
    cfg = load_config()
    bt = cfg.backtest
    start = pd.to_datetime(start or bt["start"])
    end = pd.to_datetime(end or dt.date.today().isoformat())
    top_k = top_k or int(bt["top_k"])
    train_years = int(bt["train_window_years"])
    oos_months = int(bt["oos_window_months"])
    step_months = int(bt["step_months"])
    cost = float(bt["cost_bps"]) / 10_000.0
    exit_off = int(cfg.target["exit_offset_days"])

    if panel is None:
        panel = build_panel(require_target=True)
    panel = panel.copy()
    panel = panel[panel.index <= end]

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

        model = train(train_panel,
                      weights=(weights_fn(train_panel) if weights_fn else None))

        # apply the liquidity filter on each trading day, score, take top_k
        for date, day_slice in oos_panel.groupby(level=0):
            day = day_slice.copy()
            mask = liquidity_mask(day)
            day = day[mask]
            if day.empty:
                continue
            preds = model.predict(day)
            day = day.assign(**preds)
            picks = day.sort_values("pred_mean", ascending=False).head(top_k)
            for sym, row in picks.iterrows():
                actual = row["target"]
                trades_rows.append({
                    "date": date,
                    "symbol": row.get("symbol", sym),
                    "pred_mean": row["pred_mean"],
                    "actual": actual,
                    "net": actual - cost,
                })

    trades = pd.DataFrame(trades_rows)
    if trades.empty:
        return BacktestResult(summary={"n_trades": 0, "warning": "no trades"})

    daily = (
        trades.groupby("date")["net"]
        .mean()
        .rename("daily_ret")
        .to_frame()
        .sort_index()
    )
    daily["equity"] = (1.0 + daily["daily_ret"]).cumprod()

    # Sharpe note: each "day" represents a held position spanning exit_off days,
    # so independent observations land roughly every exit_off days. Discount accordingly.
    sharpe = _annualize_sharpe(daily["daily_ret"], periods_per_year=252 / max(exit_off, 1))

    summary = {
        "n_trades": int(len(trades)),
        "hit_rate": float((trades["actual"] > 0).mean()),
        "hit_rate_net": float((trades["net"] > 0).mean()),
        "mean_return": float(trades["actual"].mean()),
        "mean_return_net": float(trades["net"].mean()),
        "median_return": float(trades["actual"].median()),
        "sharpe_net": sharpe,
        "max_drawdown": _max_drawdown(daily["equity"]),
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "top_k": top_k,
        "exit_offset_days": exit_off,
        "cost_bps": int(bt["cost_bps"]),
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
