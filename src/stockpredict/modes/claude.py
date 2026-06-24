"""Mode B: Claude does the news re-rank.

Emit a markdown plan that an in-session Claude (Claude Code / Cowork) reads
via WebFetch and fills, then run `claude-finalize` to re-rank.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from ..model.predict import rank_today
from ..news.claude_runner import parse_plan, write_plan
from ..picks_meta import picks_suffix
from ..tracking import effective_today_for_trading, run_signature


def run(on: str | None = None,
        exit_offset_days: int | None = None,
        n_picks: int | None = None,
        symbols: list[str] | None = None,
        hose_only: bool = False,
        include_etfs: bool = True,
        exclude: list[str] | None = None) -> tuple[pd.DataFrame, Path, str]:
    """Emit the interactive plan. Returns (candidates_df, plan_path, 'interactive')."""
    candidates, plan_path = emit_plan(on=on,
                                      exit_offset_days=exit_offset_days,
                                      n_picks=n_picks,
                                      symbols=symbols, hose_only=hose_only,
                                      include_etfs=include_etfs,
                                      exclude=exclude)
    return candidates, plan_path, "interactive"


def emit_plan(on: str | None = None,
              exit_offset_days: int | None = None,
              n_picks: int | None = None,
              symbols: list[str] | None = None,
              hose_only: bool = False,
              include_etfs: bool = True,
              exclude: list[str] | None = None) -> tuple[pd.DataFrame, Path]:
    full_cfg = load_config()
    requested_n = int(n_picks) if n_picks else int(full_cfg.pricing.get("default_picks", 5))
    candidates = rank_today(n_picks=requested_n, on=on,
                            exit_offset_days=exit_offset_days, symbols=symbols)
    if on is not None:
        on_date = dt.date.fromisoformat(on)
    else:
        on_date = effective_today_for_trading().date()

    eff_horizon = int(exit_offset_days) if exit_offset_days is not None else int(
        full_cfg.target["exit_offset_days"]
    )
    excl_list = sorted({s.upper() for s in (exclude or [])})
    sig = run_signature(mode="claude", exit_offset_days=eff_horizon,
                        hose_only=hose_only,
                        include_etfs=include_etfs, exclude=excl_list)

    plan_path = write_plan(candidates, on=on_date,
                           run_signature=sig,
                           current_horizon=eff_horizon,
                           current_signature=sig)
    # Sidecar parquet so `finalize` can recover pricing (entry / target / stop)
    # and other feature columns that aren't in the markdown score table.
    sidecar = plan_path.with_suffix(".candidates.parquet")
    candidates.to_parquet(sidecar, index=False)
    # Sidecar metadata so `finalize` knows the horizon / hose-only /
    # include-etfs / exclude used at emit time.
    meta_path = plan_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps({
            "exit_offset_days": eff_horizon,
            "n_picks": requested_n,
            "hose_only": hose_only,
            "include_etfs": include_etfs,
            "exclude": excl_list,
            "run_signature": sig,
        }, indent=2),
        encoding="utf-8",
    )
    return candidates, plan_path


def finalize(plan_path: str | Path) -> tuple[pd.DataFrame, Path]:
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
                    "drivers", "key_news", "dimensions_cited",
                    "adj_entry_vnd", "adj_target_vnd"]
    explain_cols = [c for c in explain_cols if c in scored.columns]
    sidecar = plan_path.with_suffix(".candidates.parquet")
    if sidecar.exists():
        candidates = pd.read_parquet(sidecar)
        merged = candidates.merge(scored[explain_cols], on="symbol", how="inner")
    else:
        # Older plans without sidecar: fall back to the bare score table.
        merged = scored
    merged["adjusted"] = merged["pred_mean"] * (1.0 + weight * merged["news_score"])
    # Parallel news-adjusted entry/target economics (adj_* columns). Purely
    # additive — the mechanical entry/target/rr columns are untouched.
    from ..pricing import add_adjusted_price_suggestions
    merged = add_adjusted_price_suggestions(merged)
    # News re-orders the SAME N candidates emitted upstream; it never adds or
    # drops names, so there's no re-slicing here — just re-rank by the
    # news-adjusted score.
    merged = merged.sort_values("adjusted", ascending=False).reset_index(drop=True)

    # Recover horizon / hose_only from the sidecar metadata (if present).
    meta_path = plan_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    exit_off = meta.get("exit_offset_days")
    eff_hose = bool(meta.get("hose_only", False))
    # Legacy meta files (pre-ETF) lack `include_etfs`; default to True so
    # the recovered signature matches what the original emit_plan produced.
    eff_etfs = bool(meta.get("include_etfs", True))
    # Legacy meta files (pre-exclude) lack `exclude`; default to [].
    eff_excl = list(meta.get("exclude") or [])
    sig = meta.get("run_signature") or run_signature(
        mode="claude",
        exit_offset_days=int(exit_off or 2),
        hose_only=eff_hose,
        include_etfs=eff_etfs,
        exclude=eff_excl,
    )

    requested_n = meta.get("n_picks")
    n_below = int(merged["below_breakeven"].fillna(True).sum()) if "below_breakeven" in merged.columns else 0
    today_ts = effective_today_for_trading()
    today = today_ts.strftime("%Y-%m-%d")
    out = reports_dir() / f"picks_claude_{today}_{sig}{picks_suffix(merged)}.json"
    payload = {
        "as_of": today,
        "mode": "claude",
        "exit_offset_days": exit_off,
        "hose_only": eff_hose,
        "include_etfs": eff_etfs,
        "exclude": eff_excl,
        "run_signature": sig,
        "selection": "top_n",
        "requested_picks": requested_n,
        "n_picks": int(len(merged)),
        "n_below_breakeven": n_below,
        "plan_file": str(plan_path),
        "weight": weight,
        "picks": json.loads(merged.to_json(orient="records")),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from ..tracking import record
    record(merged, mode="claude", as_of=today_ts,
           exit_offset_days=exit_off,
           hose_only=eff_hose, include_etfs=eff_etfs, exclude=eff_excl)
    return merged, out
