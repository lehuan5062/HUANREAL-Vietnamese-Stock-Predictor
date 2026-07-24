"""Momentum (short-term trend-following) mode: 100% LLM-agent-driven.

The whole mechanically-gated universe (staleness / ceiling-lock / corporate-
action) is handed to the agent, unranked. The agent selects, researches, and
for each pick predicts N (trading days to a profitable exit) and P (the
profit at that exit). Finalize computes ``score = P / N``, ranks by it, and
prices via ``pricing.add_recovery_price_suggestions`` (buy at close, target =
close × (1 + P), no stop — hold until the target).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..news.llm_plan_runner import parse_llm_plan, write_llm_plan
from ..pricing import add_recovery_price_suggestions
from ..selector import eligible_universe
from ..tracking import run_signature
from .common import (default_n_picks, emit_universe_meta, read_candidates_sidecar,
                     read_meta, resolve_on_date, write_picks_json)

MODE = "momentum"


def run(on: str | None = None, n_picks: int | None = None,
       symbols: list[str] | None = None, hose_only: bool = False,
       include_etfs: bool = True, exclude: list[str] | None = None
       ) -> tuple[pd.DataFrame, Path]:
    """Emit the rebound plan markdown. Returns (universe_df, plan_path)."""
    requested_n = default_n_picks(n_picks)
    universe = eligible_universe(on=on, symbols=symbols)
    on_date = resolve_on_date(on)
    excl_list = sorted({s.upper() for s in (exclude or [])})
    sig = run_signature(mode=MODE, hose_only=hose_only,
                        include_etfs=include_etfs, exclude=excl_list)
    plan_path = write_llm_plan(MODE, universe, on=on_date, run_signature=sig,
                              n_picks=requested_n)
    emit_universe_meta(plan_path, universe, method="llm_only",
                       n_picks=requested_n, hose_only=hose_only,
                       include_etfs=include_etfs, exclude=excl_list, sig=sig)
    return universe, plan_path


def finalize(plan_path: str | Path) -> tuple[pd.DataFrame, Path]:
    plan_path = Path(plan_path)
    scored = parse_llm_plan(plan_path)
    if scored.empty:
        raise RuntimeError(f"no picks parsed from {plan_path} — fill the Results table")

    dropped = scored[scored["dropped"]]
    if not dropped.empty:
        print(f"[momentum] DROP: excluding {len(dropped)} ticker(s): "
              f"{', '.join(dropped['symbol'].tolist())}")
    scored = scored[~scored["dropped"]].drop(columns=["dropped"])
    if scored.empty:
        raise RuntimeError("all picks dropped")

    bad = scored[scored["pred_days"].isna() | (scored["pred_days"] < 1)
                | scored["pred_profit"].isna() | (scored["pred_profit"] <= 0)]
    if not bad.empty:
        print(f"[momentum] WARNING: dropping {len(bad)} pick(s) with a missing/"
              f"invalid N_days or P: {', '.join(bad['symbol'].tolist())}")
    scored = scored.drop(bad.index)
    if scored.empty:
        raise RuntimeError("no picks with a valid N_days and P")

    universe = read_candidates_sidecar(plan_path)
    if universe is not None:
        ref_cols = [c for c in ["symbol", "close", "rsi_14", "mom_5", "mom_20",
                                "high_prox_20", "adv_vnd_20", "organ_name",
                                "instrument_type"]
                   if c in universe.columns]
        merged = scored.merge(universe[ref_cols], on="symbol", how="left")
    else:
        merged = scored

    merged = add_recovery_price_suggestions(merged)
    merged = merged.sort_values("score", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1

    meta = read_meta(plan_path)
    out, sig, _ = write_picks_json(MODE, merged, plan_path, meta)
    return merged, out
