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


def run(top_k: int = 5, on: str | None = None,
        units: int | None = None,
        exit_offset_days: int | None = None,
        symbols: list[str] | None = None,
        hose_only: bool = False,
        include_etfs: bool = True) -> tuple[pd.DataFrame, Path]:
    picks = rank_today(top_k=top_k, on=on, units=units,
                       exit_offset_days=exit_offset_days, symbols=symbols)
    picks = annotate_best(picks)
    if on is not None:
        today_ts = pd.Timestamp(on)
    else:
        today_ts = effective_today_for_trading()
    today = today_ts.strftime("%Y-%m-%d")

    cfg = load_config()
    eff_units = int(units) if units is not None else int(
        cfg.broker.get("default_position_units", 100)
    )
    eff_horizon = int(exit_offset_days) if exit_offset_days is not None else int(
        cfg.target["exit_offset_days"]
    )
    sig = run_signature(mode="base", exit_offset_days=eff_horizon,
                        units=eff_units, hose_only=hose_only,
                        include_etfs=include_etfs)
    out = reports_dir() / f"picks_{today}_{sig}{actionable_suffix(picks)}.json"
    payload = {
        "as_of": today,
        "mode": "base",
        "exit_offset_days": eff_horizon,
        "units": eff_units,
        "hose_only": hose_only,
        "include_etfs": include_etfs,
        "run_signature": sig,
        "top_k": top_k,
        "picks": json.loads(picks.to_json(orient="records", date_format="iso")),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    record(picks, mode="base", as_of=today_ts,
           exit_offset_days=eff_horizon, units=eff_units, hose_only=hose_only,
           include_etfs=include_etfs)
    return picks, out
