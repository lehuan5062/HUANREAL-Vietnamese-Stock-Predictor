"""Mode A: pure ML + technical filter, output top-K."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from ..model.predict import rank_today
from ..picks_meta import actionable_suffix, annotate_best
from ..tracking import effective_today_for_trading, record, run_signature


def run(max_picks: int | None = None, on: str | None = None,
        units: int | None = None,
        budget_vnd: int | None = None,
        exit_offset_days: int | None = None,
        symbols: list[str] | None = None,
        hose_only: bool = False,
        include_etfs: bool = True,
        exclude: list[str] | None = None) -> tuple[pd.DataFrame, Path]:
    picks = rank_today(actionable_only=True, max_picks=max_picks, on=on,
                       units=units, budget_vnd=budget_vnd,
                       exit_offset_days=exit_offset_days, symbols=symbols)
    picks = annotate_best(picks)
    if on is not None:
        today_ts = pd.Timestamp(on)
    else:
        today_ts = effective_today_for_trading()
    today = today_ts.strftime("%Y-%m-%d")

    cfg = load_config()
    eff_max_picks = max_picks if max_picks is not None else int(
        cfg.get("report", {}).get("max_picks", 20))
    eff_units = None if budget_vnd is not None else (
        int(units) if units is not None
        else int(cfg.broker.get("default_position_units", 100))
    )
    eff_horizon = int(exit_offset_days) if exit_offset_days is not None else int(
        cfg.target["exit_offset_days"]
    )
    sig = run_signature(mode="base", exit_offset_days=eff_horizon,
                        units=eff_units, budget_vnd=budget_vnd, hose_only=hose_only,
                        include_etfs=include_etfs, exclude=exclude)
    out = reports_dir() / f"picks_{today}_{sig}{actionable_suffix(picks)}.json"
    excl_list = sorted({s.upper() for s in (exclude or [])})
    payload = {
        "as_of": today,
        "mode": "base",
        "exit_offset_days": eff_horizon,
        "sizing_mode": "budget" if budget_vnd is not None else "units",
        "units": eff_units,
        "budget_vnd": budget_vnd,
        "hose_only": hose_only,
        "include_etfs": include_etfs,
        "exclude": excl_list,
        "run_signature": sig,
        "selection": "actionable_only",
        "max_picks": eff_max_picks,
        "n_actionable": int(len(picks)),
        "picks": json.loads(picks.to_json(orient="records", date_format="iso")),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    record(picks, mode="base", as_of=today_ts,
           exit_offset_days=eff_horizon, units=eff_units, budget_vnd=budget_vnd,
           hose_only=hose_only, include_etfs=include_etfs, exclude=excl_list)
    return picks, out
