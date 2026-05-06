"""Mode B: Claude does the news re-rank.

Two paths:
  - **autonomous**: ANTHROPIC_API_KEY is set -> call the Anthropic API with
    web_search server tool, get back a JSON of news scores, re-rank.
  - **interactive**: no API key -> emit a markdown plan that an in-session
    Claude (Claude Code / Cowork) reads via WebFetch and fills, then run
    `claude-finalize` to re-rank.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from ..envfile import load as load_env
from ..model.predict import rank_today
from ..news.claude_runner import parse_plan, write_plan
from ..picks_meta import annotate_best
from ..tracking import effective_today_for_trading, run_signature


def _autonomous_available() -> bool:
    load_env()  # populate env from .env if present
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def run(top_k_final: int = 5, on: str | None = None,
        units: int | None = None,
        exit_offset_days: int | None = None,
        symbols: list[str] | None = None,
        hose_only: bool = False) -> tuple[pd.DataFrame, Path, str]:
    """Run the best-available path. Returns (picks_df, output_path, mode_tag)
    where mode_tag is 'autonomous' or 'interactive'."""
    if _autonomous_available():
        try:
            return _run_autonomous(top_k_final=top_k_final, on=on, units=units,
                                   exit_offset_days=exit_offset_days,
                                   symbols=symbols, hose_only=hose_only)
        except Exception as e:
            print(f"[claude] autonomous mode failed ({e}); falling back to plan emit.")
    candidates, plan_path = emit_plan(top_k_final=top_k_final, on=on, units=units,
                                      exit_offset_days=exit_offset_days,
                                      symbols=symbols, hose_only=hose_only)
    return candidates, plan_path, "interactive"


def emit_plan(top_k_final: int = 5, on: str | None = None,
              units: int | None = None,
              exit_offset_days: int | None = None,
              symbols: list[str] | None = None,
              hose_only: bool = False) -> tuple[pd.DataFrame, Path]:
    cfg = load_config().modes["claude"]
    pool = int(cfg["candidate_pool"])
    candidates = rank_today(top_k=pool, on=on, units=units,
                            exit_offset_days=exit_offset_days, symbols=symbols)
    if on is not None:
        on_date = dt.date.fromisoformat(on)
    else:
        on_date = effective_today_for_trading().date()

    full_cfg = load_config()
    eff_units = int(units) if units is not None else int(
        full_cfg.broker.get("default_position_units", 100)
    )
    eff_horizon = int(exit_offset_days) if exit_offset_days is not None else int(
        full_cfg.target["exit_offset_days"]
    )
    sig = run_signature(mode="claude", exit_offset_days=eff_horizon,
                        units=eff_units, hose_only=hose_only)

    plan_path = write_plan(candidates, on=on_date,
                           run_signature=sig,
                           current_horizon=eff_horizon,
                           current_signature=sig)
    # Sidecar parquet so `finalize` can recover pricing (entry / target / stop)
    # and other feature columns that aren't in the markdown score table.
    sidecar = plan_path.with_suffix(".candidates.parquet")
    candidates.to_parquet(sidecar, index=False)
    # Sidecar metadata so `finalize` knows the horizon / units / hose-only
    # used at emit time.
    meta_path = plan_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps({
            "exit_offset_days": eff_horizon,
            "units": eff_units,
            "hose_only": hose_only,
            "run_signature": sig,
        }, indent=2),
        encoding="utf-8",
    )
    return candidates, plan_path


def _run_autonomous(top_k_final: int, on: str | None,
                    units: int | None = None,
                    exit_offset_days: int | None = None,
                    symbols: list[str] | None = None,
                    hose_only: bool = False) -> tuple[pd.DataFrame, Path, str]:
    from ..news import claude_api

    cfg = load_config().modes["claude"]
    pool = int(cfg["candidate_pool"])
    candidates = rank_today(top_k=pool, on=on, units=units,
                            exit_offset_days=exit_offset_days, symbols=symbols)
    if candidates.empty:
        raise RuntimeError("no candidates from ML stage")

    if on is not None:
        today_ts = pd.Timestamp(on)
    else:
        today_ts = effective_today_for_trading()
    today = today_ts.strftime("%Y-%m-%d")

    full_cfg = load_config()
    eff_units = int(units) if units is not None else int(
        full_cfg.broker.get("default_position_units", 100)
    )
    eff_horizon = int(exit_offset_days) if exit_offset_days is not None else int(
        full_cfg.target["exit_offset_days"]
    )
    sig = run_signature(mode="claude", exit_offset_days=eff_horizon,
                        units=eff_units, hose_only=hose_only)

    print(f"[claude] calling Anthropic API to score {len(candidates)} candidates...")
    scored = claude_api.score(candidates, date=today,
                              current_horizon=eff_horizon,
                              current_signature=sig)
    merged = claude_api.merge(candidates, scored).head(top_k_final)
    merged = annotate_best(merged)

    out = reports_dir() / f"picks_claude_{today}_{sig}.json"
    payload = {
        "as_of": today,
        "mode": "claude_autonomous",
        "exit_offset_days": eff_horizon,
        "units": eff_units,
        "hose_only": hose_only,
        "run_signature": sig,
        "global_summary": scored.get("global_summary", ""),
        "weight": float(cfg["news_weight"]),
        "picks": json.loads(merged.to_json(orient="records")),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from ..tracking import record
    record(merged, mode="claude", as_of=today_ts,
           exit_offset_days=eff_horizon, units=eff_units, hose_only=hose_only)
    return merged, out, "autonomous"


def finalize(plan_path: str | Path, top_k_final: int = 5) -> tuple[pd.DataFrame, Path]:
    from ..news.claude_runner import DROP_SENTINEL

    cfg = load_config().modes["claude"]
    weight = float(cfg["news_weight"])
    plan_path = Path(plan_path)
    scored = parse_plan(plan_path)
    if scored.empty:
        raise RuntimeError(f"no scores parsed from {plan_path}")

    # Hard-DROP override: excluded entirely regardless of ML signal.
    dropped = scored[scored["news_score"] == DROP_SENTINEL]
    if not dropped.empty:
        print(f"[claude] DROP override: excluding {len(dropped)} ticker(s): "
              f"{', '.join(dropped['symbol'].tolist())}")
    scored = scored[scored["news_score"] != DROP_SENTINEL].copy()
    if scored.empty:
        raise RuntimeError("all candidates dropped")

    # Recover pricing columns (entry/target/stop/rr_ratio) from the sidecar
    # parquet that emit_plan saved alongside the markdown plan, and bring
    # in the explanation columns (business, dimensions, drivers, key_news,
    # dimensions_cited) from parse_plan. dimensions_cited rides through to
    # `record()` so the ledger can later aggregate hit-rate by dimension.
    explain_cols = ["symbol", "news_score", "business", "dimensions",
                    "drivers", "key_news", "dimensions_cited"]
    explain_cols = [c for c in explain_cols if c in scored.columns]
    sidecar = plan_path.with_suffix(".candidates.parquet")
    if sidecar.exists():
        candidates = pd.read_parquet(sidecar)
        merged = candidates.merge(scored[explain_cols], on="symbol", how="inner")
    else:
        # Older plans without sidecar: fall back to the bare score table.
        merged = scored
    merged["adjusted"] = merged["pred_mean"] * (1.0 + weight * merged["news_score"])
    merged = merged.sort_values("adjusted", ascending=False).head(top_k_final).reset_index(drop=True)
    merged = annotate_best(merged)

    # Recover horizon / units / hose_only from the sidecar metadata (if present).
    meta_path = plan_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    exit_off = meta.get("exit_offset_days")
    eff_units = meta.get("units")
    eff_hose = bool(meta.get("hose_only", False))
    sig = meta.get("run_signature") or run_signature(
        mode="claude",
        exit_offset_days=int(exit_off or 2),
        units=int(eff_units or 100),
        hose_only=eff_hose,
    )

    today_ts = effective_today_for_trading()
    today = today_ts.strftime("%Y-%m-%d")
    out = reports_dir() / f"picks_claude_{today}_{sig}.json"
    payload = {
        "as_of": today,
        "mode": "claude",
        "exit_offset_days": exit_off,
        "units": eff_units,
        "hose_only": eff_hose,
        "run_signature": sig,
        "plan_file": str(plan_path),
        "weight": weight,
        "picks": json.loads(merged.to_json(orient="records")),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from ..tracking import record
    record(merged, mode="claude", as_of=today_ts,
           exit_offset_days=exit_off, units=eff_units, hose_only=eff_hose)
    return merged, out
