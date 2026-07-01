"""Mode A: pure ML + technical filter, output top-K."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from ..model.predict import rank_today
from ..picks_meta import picks_suffix
from ..tracking import effective_today_for_trading, record, run_signature


def run(on: str | None = None,
        exit_offset_days: int | None = None,
        n_picks: int | None = None,
        symbols: list[str] | None = None,
        hose_only: bool = False,
        include_etfs: bool = True,
        exclude: list[str] | None = None) -> tuple[pd.DataFrame, Path]:
    cfg = load_config()
    requested_n = int(n_picks) if n_picks else int(cfg.pricing.get("default_picks", 5))
    picks = rank_today(n_picks=requested_n, on=on,
                       exit_offset_days=exit_offset_days, symbols=symbols)
    if on is not None:
        today_ts = pd.Timestamp(on)
    else:
        today_ts = effective_today_for_trading()
    today = today_ts.strftime("%Y-%m-%d")

    eff_horizon = int(exit_offset_days) if exit_offset_days is not None else int(
        cfg.target["exit_offset_days"]
    )
    sig = run_signature(mode="base", exit_offset_days=eff_horizon,
                        hose_only=hose_only,
                        include_etfs=include_etfs, exclude=exclude)
    out = reports_dir() / f"picks_{today}_{sig}{picks_suffix(picks)}.json"
    excl_list = sorted({s.upper() for s in (exclude or [])})
    n_below = int(picks["below_recovery_bar"].fillna(True).sum()) if "below_recovery_bar" in picks.columns else 0
    payload = {
        "as_of": today,
        "mode": "base",
        "exit_offset_days": eff_horizon,
        "hose_only": hose_only,
        "include_etfs": include_etfs,
        "exclude": excl_list,
        "run_signature": sig,
        "selection": "top_n",
        "requested_picks": requested_n,
        "n_picks": int(len(picks)),
        "n_below_breakeven": n_below,
        "picks": json.loads(picks.to_json(orient="records", date_format="iso")),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    record(picks, mode="base", as_of=today_ts,
           exit_offset_days=eff_horizon,
           hose_only=hose_only, include_etfs=include_etfs, exclude=excl_list)
    return picks, out
