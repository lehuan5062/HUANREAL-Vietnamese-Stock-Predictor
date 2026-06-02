"""Mode C: emit a self-contained prompt for the user to paste into Gemini Chat
on the web (with browsing). Gemini does the news re-rank externally; the local
program does not call any Gemini API and does not pass past-performance feedback
(Gemini Chat on the web cannot update the local ledger).

Two-step flow:
  1. `predict --mode gemini` writes a prompt + a `.candidates.parquet` sidecar.
  2. User pastes Gemini's JSON response into `reports/gemini_response_<date>.json`.
  3. `gemini-finalize <prompt-path>` reads the response, merges with the sidecar,
     produces the final explained picks JSON.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from ..model.predict import rank_today
from ..news.gemini_prompt import write_prompt
from ..news.gemini_response import merge_response, parse_response
from ..picks_meta import actionable_suffix, annotate_best
from ..tracking import effective_today_for_trading, run_signature


def run(max_picks: int | None = None, on: str | None = None,
        units: int | None = None,
        budget_vnd: int | None = None,
        exit_offset_days: int | None = None,
        symbols: list[str] | None = None,
        hose_only: bool = False,
        include_etfs: bool = True,
        exclude: list[str] | None = None) -> tuple[pd.DataFrame, Path, str]:
    """Always emits a prompt file; returns (candidates, prompt_path, 'prompt-only')."""
    candidates, prompt_path = emit_prompt(max_picks=max_picks, on=on, units=units,
                                          budget_vnd=budget_vnd,
                                          exit_offset_days=exit_offset_days,
                                          symbols=symbols, hose_only=hose_only,
                                          include_etfs=include_etfs,
                                          exclude=exclude)
    return candidates, prompt_path, "prompt-only"


def emit_prompt(max_picks: int | None = None, on: str | None = None,
                units: int | None = None,
                budget_vnd: int | None = None,
                exit_offset_days: int | None = None,
                symbols: list[str] | None = None,
                hose_only: bool = False,
                include_etfs: bool = True,
                exclude: list[str] | None = None) -> tuple[pd.DataFrame, Path]:
    candidates = rank_today(actionable_only=True, max_picks=max_picks, on=on,
                            units=units, budget_vnd=budget_vnd,
                            exit_offset_days=exit_offset_days, symbols=symbols)
    if on:
        on_date = dt.date.fromisoformat(on)
    else:
        on_date = effective_today_for_trading().date()

    full_cfg = load_config()
    eff_units = None if budget_vnd is not None else (
        int(units) if units is not None
        else int(full_cfg.broker.get("default_position_units", 100))
    )
    eff_horizon = int(exit_offset_days) if exit_offset_days is not None else int(
        full_cfg.target["exit_offset_days"]
    )
    excl_list = sorted({s.upper() for s in (exclude or [])})
    sig = run_signature(mode="gemini", exit_offset_days=eff_horizon,
                        units=eff_units, budget_vnd=budget_vnd, hose_only=hose_only,
                        include_etfs=include_etfs, exclude=excl_list)

    # write_prompt currently uses the date in the filename; we suffix with sig
    # and the actionable tickers so a directory listing surfaces them at a glance.
    path = write_prompt(candidates, on=on_date, exit_offset_days=eff_horizon)
    full_suffix = f"_{sig}{actionable_suffix(candidates)}"
    sig_path = path.with_name(path.stem + full_suffix + path.suffix)
    if sig_path != path:
        path.replace(sig_path)
        path = sig_path
    # Sidecars so `gemini-finalize` can recover pricing + run params.
    sidecar = path.with_suffix(".candidates.parquet")
    candidates.to_parquet(sidecar, index=False)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps({
            "exit_offset_days": eff_horizon,
            "sizing_mode": "budget" if budget_vnd is not None else "units",
            "units": eff_units,
            "budget_vnd": budget_vnd,
            "hose_only": hose_only,
            "include_etfs": include_etfs,
            "exclude": excl_list,
            "run_signature": sig,
        }, indent=2),
        encoding="utf-8",
    )
    return candidates, path


def finalize(prompt_path: str | Path,
             response_path: str | Path | None = None,
             max_picks: int | None = None) -> tuple[pd.DataFrame, Path]:
    """Merge Gemini's JSON response back into the saved candidates and produce
    the final explained top-K picks JSON.

    `prompt_path` is the txt file that emit_prompt wrote.
    `response_path` is where the user saved Gemini's JSON output. If omitted
    we look for `reports/gemini_response_<date>.json` next to the prompt.
    """
    prompt_path = Path(prompt_path)
    if response_path is None:
        # Default path: same date stamp as the prompt
        stem = prompt_path.stem  # e.g. gemini_prompt_2026-05-05
        date_part = stem.replace("gemini_prompt_", "")
        response_path = reports_dir() / f"gemini_response_{date_part}.json"
    response_path = Path(response_path)
    if not response_path.exists():
        raise FileNotFoundError(
            f"Gemini response not found at {response_path}. Save Gemini's JSON "
            f"output to that path first."
        )

    cfg = load_config().modes["gemini"]
    weight = float(cfg["news_weight"])

    raw = response_path.read_text(encoding="utf-8")
    response = parse_response(raw)

    sidecar = prompt_path.with_suffix(".candidates.parquet")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"candidates sidecar not found at {sidecar} — was the prompt produced "
            f"by a current build? Re-run `predict --mode gemini` first."
        )
    candidates = pd.read_parquet(sidecar)

    merged = merge_response(candidates, response)
    if merged.empty:
        raise RuntimeError("all candidates dropped or unmatched")

    merged["adjusted"] = merged["pred_mean"] * (1.0 + weight * merged["news_score"])
    merged = merged.sort_values("adjusted", ascending=False).reset_index(drop=True)
    # The candidate set is already the capped actionable set from emit time;
    # honor an explicit --top override here as a manual trim, otherwise list all.
    if max_picks is not None:
        merged = merged.head(int(max_picks)).reset_index(drop=True)
    merged = annotate_best(merged)

    # Recover horizon from the sidecar metadata.
    exit_off = None
    meta_path = prompt_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            exit_off = json.loads(meta_path.read_text(encoding="utf-8")).get("exit_offset_days")
        except Exception:
            exit_off = None

    # Pull units / hose_only / include_etfs / exclude / signature from sidecar meta if present.
    eff_units = None
    eff_budget = None
    sizing_mode = "units"
    eff_hose = False
    # Legacy meta files (pre-ETF) lack `include_etfs`; default to True so
    # the recovered signature matches what the original emit_prompt produced.
    eff_etfs = True
    # Legacy meta files (pre-exclude) lack `exclude`; default to [].
    eff_excl: list[str] = []
    sig = None
    meta_path = prompt_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            eff_units = meta.get("units")
            eff_budget = meta.get("budget_vnd")
            sizing_mode = meta.get("sizing_mode") or (
                "budget" if eff_budget is not None else "units")
            eff_hose = bool(meta.get("hose_only", False))
            eff_etfs = bool(meta.get("include_etfs", True))
            eff_excl = list(meta.get("exclude") or [])
            sig = meta.get("run_signature")
        except Exception:
            pass
    if sig is None:
        sig = run_signature(mode="gemini",
                            exit_offset_days=int(exit_off or 2),
                            units=(None if eff_budget is not None else int(eff_units or 100)),
                            budget_vnd=eff_budget,
                            hose_only=eff_hose,
                            include_etfs=eff_etfs,
                            exclude=eff_excl)

    today_ts = effective_today_for_trading()
    today = today_ts.strftime("%Y-%m-%d")
    out = reports_dir() / f"picks_gemini_{today}_{sig}{actionable_suffix(merged)}.json"
    payload = {
        "as_of": today,
        "mode": "gemini",
        "exit_offset_days": exit_off,
        "sizing_mode": sizing_mode,
        "units": eff_units,
        "budget_vnd": eff_budget,
        "hose_only": eff_hose,
        "include_etfs": eff_etfs,
        "exclude": eff_excl,
        "run_signature": sig,
        "selection": "actionable_only",
        "n_actionable": int(len(merged)),
        "prompt_file": str(prompt_path),
        "response_file": str(response_path),
        "global_summary": response.get("global_summary", ""),
        "weight": weight,
        "picks": json.loads(merged.to_json(orient="records")),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from ..tracking import record
    record(merged, mode="gemini", as_of=today_ts,
           exit_offset_days=exit_off, units=eff_units, budget_vnd=eff_budget,
           hose_only=eff_hose, include_etfs=eff_etfs, exclude=eff_excl)
    return merged, out
