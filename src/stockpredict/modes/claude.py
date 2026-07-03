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
        n_picks: int | None = None,
        symbols: list[str] | None = None,
        hose_only: bool = False,
        include_etfs: bool = True,
        exclude: list[str] | None = None,
        union_missed: bool = False,
        llm_only: bool = False) -> tuple[pd.DataFrame, Path, str]:
    """Emit the interactive plan. Returns (candidates_df, plan_path, tag).

    ``llm_only`` switches to the no-ML path: the whole eligible universe (no
    ``pred_mean``) is handed to the LLM, which selects / ranks / prices itself.
    """
    if llm_only:
        universe, plan_path = emit_llm_plan(on=on,
                                            n_picks=n_picks,
                                            symbols=symbols, hose_only=hose_only,
                                            include_etfs=include_etfs,
                                            exclude=exclude)
        return universe, plan_path, "interactive-llm"
    candidates, plan_path = emit_plan(on=on,
                                      n_picks=n_picks,
                                      symbols=symbols, hose_only=hose_only,
                                      include_etfs=include_etfs,
                                      exclude=exclude, union_missed=union_missed)
    return candidates, plan_path, "interactive"


def emit_llm_plan(on: str | None = None,
                  n_picks: int | None = None,
                  symbols: list[str] | None = None,
                  hose_only: bool = False,
                  include_etfs: bool = True,
                  exclude: list[str] | None = None) -> tuple[pd.DataFrame, Path]:
    """LLM-only emit: hand the WHOLE eligible universe (no ML scoring) to the
    LLM. Writes a `claude_llm_plan_*` markdown + sidecars and returns
    (universe_df, plan_path)."""
    from ..model.predict import eligible_universe
    from ..news.claude_llm_runner import write_llm_plan

    full_cfg = load_config()
    requested_n = int(n_picks) if n_picks else int(full_cfg.pricing.get("default_picks", 5))
    universe = eligible_universe(on=on, symbols=symbols)
    if on is not None:
        on_date = dt.date.fromisoformat(on)
    else:
        on_date = effective_today_for_trading().date()
    excl_list = sorted({s.upper() for s in (exclude or [])})
    sig = run_signature(mode="claude_llm", hose_only=hose_only,
                        include_etfs=include_etfs, exclude=excl_list)
    plan_path = write_llm_plan(universe, on=on_date, run_signature=sig,
                               n_picks=requested_n)
    sidecar = plan_path.with_suffix(".candidates.parquet")
    universe.to_parquet(sidecar, index=False)
    meta_path = plan_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps({
            "method": "llm_only",
            "n_picks": requested_n,
            "hose_only": hose_only,
            "include_etfs": include_etfs,
            "exclude": excl_list,
            "run_signature": sig,
        }, indent=2),
        encoding="utf-8",
    )
    return universe, plan_path


def emit_plan(on: str | None = None,
              n_picks: int | None = None,
              symbols: list[str] | None = None,
              hose_only: bool = False,
              include_etfs: bool = True,
              exclude: list[str] | None = None,
              union_missed: bool = False) -> tuple[pd.DataFrame, Path]:
    full_cfg = load_config()
    requested_n = int(n_picks) if n_picks else int(full_cfg.pricing.get("default_picks", 5))
    candidates = rank_today(n_picks=requested_n, on=on, symbols=symbols)
    ab_verdict = None
    if on is not None:
        on_date = dt.date.fromisoformat(on)
    else:
        on_date = effective_today_for_trading().date()

    excl_list = sorted({s.upper() for s in (exclude or [])})
    sig = run_signature(mode="claude", hose_only=hose_only,
                        include_etfs=include_etfs, exclude=excl_list)

    plan_path = write_plan(candidates, on=on_date,
                           run_signature=sig,
                           current_signature=sig,
                           ab_verdict=ab_verdict)
    # Sidecar parquet so `finalize` can recover pricing (entry / target / stop)
    # and other feature columns that aren't in the markdown score table.
    sidecar = plan_path.with_suffix(".candidates.parquet")
    candidates.to_parquet(sidecar, index=False)
    # Sidecar metadata so `finalize` knows the horizon / hose-only /
    # include-etfs / exclude used at emit time.
    meta_path = plan_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps({
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
    # Nudge the P/N ``score`` by the LLM's news vetting.
    merged["adjusted"] = merged["score"] * (1.0 + weight * merged["news_score"])
    # Parallel news-adjusted entry/target economics (adj_* columns). Purely
    # additive — the mechanical entry/target/rr columns are untouched.
    from ..pricing import add_adjusted_price_suggestions
    merged = add_adjusted_price_suggestions(merged)
    merged = merged.sort_values("adjusted", ascending=False).reset_index(drop=True)

    # Recover horizon / hose_only from the sidecar metadata (if present).
    meta_path = plan_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    eff_hose = bool(meta.get("hose_only", False))
    eff_etfs = bool(meta.get("include_etfs", True))
    eff_excl = list(meta.get("exclude") or [])
    sig = meta.get("run_signature") or run_signature(
        mode="claude",
        hose_only=eff_hose,
        include_etfs=eff_etfs,
        exclude=eff_excl,
    )

    requested_n = meta.get("n_picks")
    # The emitted candidates may be a UNION (standard + missed) larger than N,
    # so after the LLM's news re-rank, keep the top N by adjusted score.
    if requested_n and len(merged) > int(requested_n):
        merged = merged.head(int(requested_n)).reset_index(drop=True)
    bar_col = ("below_recovery_bar" if "below_recovery_bar" in merged.columns
               else "below_breakeven")
    n_below = int(merged[bar_col].fillna(True).sum()) if bar_col in merged.columns else 0
    today_ts = effective_today_for_trading()
    today = today_ts.strftime("%Y-%m-%d")
    out = reports_dir() / f"picks_claude_{today}_{sig}{picks_suffix(merged)}.json"
    payload = {
        "as_of": today,
        "mode": "claude",
        "hose_only": eff_hose,
        "include_etfs": eff_etfs,
        "exclude": eff_excl,
        "run_signature": sig,
        "selection": "top_n",
        "requested_picks": requested_n,
        "n_picks": int(len(merged)),
        "n_below_recovery_bar": n_below,
        "plan_file": str(plan_path),
        "weight": weight,
        "picks": json.loads(merged.to_json(orient="records")),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from ..tracking import record
    record(merged, mode="claude", as_of=today_ts,
           hose_only=eff_hose, include_etfs=eff_etfs, exclude=eff_excl)
    return merged, out


def finalize_llm(plan_path: str | Path) -> tuple[pd.DataFrame, Path]:
    """Finalize an LLM-only plan: the LLM predicted N (``pred_days``) and P
    (``pred_profit``) per pick; compute ``score = P / N``, rank by it, and price
    through the SAME recovery pricing the base/hybrid modes use (buy at close,
    target = close × (1 + P), no stop). Writes a ``picks_claude_llm_*`` JSON in
    the same format and records under ``mode='claude_llm'``."""
    from ..news.claude_llm_runner import parse_llm_plan
    from ..pricing import add_recovery_price_suggestions

    plan_path = Path(plan_path)
    scored = parse_llm_plan(plan_path)
    if scored.empty:
        raise RuntimeError(f"no picks parsed from {plan_path} — fill the Results table")

    dropped = scored[scored["dropped"]]
    if not dropped.empty:
        print(f"[claude-llm] DROP: excluding {len(dropped)} ticker(s): "
              f"{', '.join(dropped['symbol'].tolist())}")
    scored = scored[~scored["dropped"]].drop(columns=["dropped"])
    if scored.empty:
        raise RuntimeError("all picks dropped")

    # Every pick must carry a valid N and P — they drive the score, the target
    # price, and the ledger's per-pick target_date.
    bad = scored[scored["pred_days"].isna() | (scored["pred_days"] < 1)
                 | scored["pred_profit"].isna() | (scored["pred_profit"] <= 0)]
    if not bad.empty:
        print(f"[claude-llm] WARNING: dropping {len(bad)} pick(s) with a missing/"
              f"invalid N_days or P: {', '.join(bad['symbol'].tolist())}")
    scored = scored.drop(bad.index)
    if scored.empty:
        raise RuntimeError("no picks with a valid N_days and P")

    # Recover reference columns from the universe sidecar — the buy price and
    # economics need the close.
    sidecar = plan_path.with_suffix(".candidates.parquet")
    if sidecar.exists():
        universe = pd.read_parquet(sidecar)
        ref_cols = [c for c in ["symbol", "close", "rsi_14", "mom_5", "mom_20",
                                "high_prox_20", "adv_vnd_20", "organ_name",
                                "instrument_type"]
                    if c in universe.columns]
        merged = scored.merge(universe[ref_cols], on="symbol", how="left")
    else:
        merged = scored

    # Same pricing path as base/hybrid. No pred_recovery_prob column here — the
    # LLM's selection vetting stands in for it, so score reduces to P/N.
    merged = add_recovery_price_suggestions(merged)
    merged = merged.sort_values("score", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1

    # Recover run params / signature from the meta sidecar.
    meta_path = plan_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    eff_hose = bool(meta.get("hose_only", False))
    eff_etfs = bool(meta.get("include_etfs", True))
    eff_excl = list(meta.get("exclude") or [])
    sig = meta.get("run_signature") or run_signature(
        mode="claude_llm",
        hose_only=eff_hose, include_etfs=eff_etfs, exclude=eff_excl)

    requested_n = meta.get("n_picks")
    if requested_n and len(merged) > int(requested_n):
        merged = merged.head(int(requested_n)).reset_index(drop=True)
    bar_col = ("below_recovery_bar" if "below_recovery_bar" in merged.columns
               else "below_breakeven")
    n_below = int(merged[bar_col].fillna(True).sum()) if bar_col in merged.columns else 0
    today_ts = effective_today_for_trading()
    today = today_ts.strftime("%Y-%m-%d")
    out = reports_dir() / f"picks_claude_llm_{today}_{sig}{picks_suffix(merged)}.json"
    payload = {
        "as_of": today,
        "mode": "claude_llm",
        "method": "llm_only",
        "hose_only": eff_hose,
        "include_etfs": eff_etfs,
        "exclude": eff_excl,
        "run_signature": sig,
        "selection": "llm_pick",
        "requested_picks": requested_n,
        "n_picks": int(len(merged)),
        "n_below_recovery_bar": n_below,
        "plan_file": str(plan_path),
        "picks": json.loads(merged.to_json(orient="records")),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from ..tracking import record
    record(merged, mode="claude_llm", as_of=today_ts,
           hose_only=eff_hose, include_etfs=eff_etfs, exclude=eff_excl)
    return merged, out
