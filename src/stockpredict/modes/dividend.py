"""Dividend (long-term hold) mode: 100% LLM-agent-driven.

Unlike momentum/rebound, dividend numbers are NOT delegated to the agent's
web search — ``data.dividends`` is a deterministic fetcher (KBS/VCI, same
vnai-bypass technique as OHLCV) that computes real yield/payout-history
columns. The agent's job is purely to VET the sustainability of those real
numbers (earnings coverage, governance, dilution risk, sector stability) and
predict an ``expected_hold_years`` + confidence. There is no N/P/target — a
dividend pick is a hold, not a swing trade, so pricing is buy-at-close only
(see ``pricing.add_dividend_price_suggestions``).
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

from ..config import reports_dir
from ..data.dividends import dividend_summary
from ..news.llm_plan_runner import (_extract_dimension_tags, _extract_findings_list,
                                    _extract_step, _split_per_ticker_sections)
from ..news.sources import global_urls, vn_urls
from ..pricing import add_dividend_price_suggestions
from ..selector import eligible_universe
from ..tracking import run_signature
from .common import (default_n_picks, emit_universe_meta, read_candidates_sidecar,
                     read_meta, resolve_on_date, write_picks_json)

MODE = "dividend"


def _enrich_with_dividend_data(universe: pd.DataFrame, on_date: dt.date) -> pd.DataFrame:
    """Merge in the deterministic dividend-history columns for every symbol in
    the eligible universe: ``dividend_yield_ttm``, ``years_paid_consecutive``,
    ``last_ex_date``, ``payout_trend``, ``n_dividend_events``."""
    rows = []
    for _, r in universe.iterrows():
        sym = str(r["symbol"]).upper()
        summary = dividend_summary(sym, close_vnd_thousand=r.get("close"), as_of=on_date)
        rows.append({"symbol": sym, **summary})
    div_df = pd.DataFrame(rows)
    return universe.merge(div_df, on="symbol", how="left")


def run(on: str | None = None, n_picks: int | None = None,
       symbols: list[str] | None = None, hose_only: bool = False,
       include_etfs: bool = True, exclude: list[str] | None = None
       ) -> tuple[pd.DataFrame, Path]:
    """Fetch the eligible universe + dividend history, emit the dividend plan
    markdown. Returns (universe_df, plan_path)."""
    requested_n = default_n_picks(n_picks)
    universe = eligible_universe(on=on, symbols=symbols)
    on_date = resolve_on_date(on)
    universe = _enrich_with_dividend_data(universe, on_date)

    excl_list = sorted({s.upper() for s in (exclude or [])})
    sig = run_signature(mode=MODE, hose_only=hose_only,
                        include_etfs=include_etfs, exclude=excl_list)
    plan_path = _write_dividend_plan(universe, on=on_date, run_signature=sig,
                                     n_picks=requested_n)
    emit_universe_meta(plan_path, universe, method="llm_only",
                       n_picks=requested_n, hose_only=hose_only,
                       include_etfs=include_etfs, exclude=excl_list, sig=sig)
    return universe, plan_path


def _write_dividend_plan(universe: pd.DataFrame, on: dt.date, run_signature: str,
                         n_picks: int) -> Path:
    out_dir = reports_dir()
    path = out_dir / f"dividend_plan_{on.isoformat()}_{run_signature}.md"

    from ..news.company_info import enrich
    universe = enrich(universe)

    lines = [
        f"# Dividend pick plan — {on.isoformat()}",
        "",
        "## Method — vet payout sustainability (no ML ranking)",
        "",
        "This is the **dividend** (long-term hold) strategy: a HOLD, not a",
        "swing trade — buy at close, no profit target, no stop-loss, no fixed",
        "exit day. The yield/payout numbers below come from a DETERMINISTIC",
        "fetcher (real VCI corporate-events data, not your web search) — your",
        "job is to VET whether the payout is sustainable, not to find the",
        "numbers yourself.",
        "",
        f"1. **Select** the best **{int(n_picks)}** name(s) from the universe",
        "   table below, using the real dividend_yield_ttm / "
        "years_paid_consecutive / payout_trend columns as your starting point.",
        "2. **Research sustainability**: earnings coverage (can the company",
        "   afford this payout from FCF/earnings, not debt?), governance/audit",
        "   flags, dilution risk (is it really a stock dividend disguised as a",
        "   cash one, or issuing new shares to fund the payout?), sector",
        "   stability, and any signs the payout is about to be cut.",
        "3. **Predict** `expected_hold_years` (how many years you'd expect to",
        "   hold this for the dividend thesis to play out) and a `confidence`",
        "   (`low` / `med` / `high`) per pick.",
        "4. For each chosen name, write a `### TICKER — Company` section",
        "   documenting the business, dimensions researched, and findings —",
        "   then fill the results table at the bottom.",
        "",
        "**Hard override**: if you find a delisting / trading halt / bankruptcy",
        "filing / an imminent dividend CUT, do NOT pick the name (or write",
        "`DROP` in its results row).",
        "",
        "## Global / macro context (read once)",
        "",
        "Scan for major global shocks (wars, sanctions/tariffs, sharp oil/gold/",
        "USD-VND moves) and note the VN-Index's broad trend — a dividend hold",
        "cares less about short-term index moves than momentum/rebound do, but",
        "a sector-wide shock (e.g. a rate move hitting bank dividends) still",
        "matters.",
        "",
    ]
    for name, url in global_urls().items():
        lines.append(f"- [{name}]({url})")
    lines += [
        "",
        "## Universe (UNRANKED — the full mechanically-gated set + real dividend data)",
        "",
        "| symbol | company | close | dividend_yield_ttm | years_paid_consecutive | "
        "payout_trend | last_ex_date | type |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in universe.iterrows():
        sym = str(row["symbol"]).upper()
        organ = (row.get("organ_name", "") or "").replace("|", "/")
        close = row.get("close", float("nan"))
        yld = row.get("dividend_yield_ttm", float("nan"))
        years = row.get("years_paid_consecutive", 0)
        trend = row.get("payout_trend", "unknown")
        last_ex = row.get("last_ex_date", "") or ""
        rtype = str(row.get("instrument_type", "STOCK") or "STOCK").upper()
        close_s = f"{close:.0f}" if pd.notna(close) else ""
        yld_s = f"{yld:.2%}" if pd.notna(yld) else ""
        lines.append(f"| {sym} | {organ} | {close_s} | {yld_s} | {years} | "
                     f"{trend} | {last_ex} | {rtype} |")

    lines += [
        "",
        "## Per-pick research sections",
        "",
        "For EACH name you choose, add a section in this exact format. Use",
        "WebFetch / WebSearch, cross-check at least 2 sources, tag each finding",
        "with `[dimension]`.",
        "",
        "Seed sources you can reuse per ticker (replace TICKER):",
        "",
    ]
    for name, url in vn_urls("TICKER").items():
        lines.append(f"- [{name}]({url})")

    lines += [
        "",
        "```",
        "### TICKER  —  Company name",
        "",
        "**Step 1 — Business**: one line on what the company does.",
        "- ",
        "",
        "**Step 2 — Research dimensions**: the 3-7 sustainability drivers you",
        "judged matter for THIS ticker's payout (earnings coverage, governance,",
        "dilution, sector cycle, ...).",
        "- ",
        "",
        "**Step 4 — Findings** (one bullet per dimension, tagged `[dimension-name]`,",
        "with dates + sources):",
        "- ",
        "```",
        "",
        "## Results — fill this with your chosen picks",
        "",
        "`expected_hold_years` >= 0.5; `confidence` in {low, med, high}. Write",
        "`DROP` in `expected_hold_years` to exclude a row you listed.",
        "",
        "| rank | symbol | expected_hold_years | confidence |",
        "| --- | --- | --- | --- |",
    ]
    for i in range(int(n_picks)):
        lines.append(f"| {i + 1} |  |  |  |")
    lines += [
        "",
        "When done, run:",
        f"  `python -m stockpredict.cli finalize reports/{path.name}`",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_SYM_RE = re.compile(r"^[A-Z0-9]{2,8}$")
_NUM_RE = re.compile(r"^[+\-]?\d+(?:\.\d+)?$")
_CONF_MAP = {"low": 0.33, "med": 0.66, "medium": 0.66, "high": 1.0}


def _parse_result_row(line: str):
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    if len(cells) < 4:
        return None
    sym = cells[1].upper()
    if not _SYM_RE.match(sym):
        return None
    hold_str = cells[2].upper()
    if hold_str == "DROP":
        return sym, True, float("nan"), float("nan")
    if not _NUM_RE.match(hold_str):
        return None
    hold_years = float(hold_str)
    conf_str = cells[3].strip().lower()
    confidence = _CONF_MAP.get(conf_str, float("nan"))
    return sym, False, hold_years, confidence


def parse_dividend_plan(path: str | Path) -> pd.DataFrame:
    """Read the filled dividend plan and return DataFrame[symbol, dropped,
    expected_hold_years, confidence, business, dimensions, key_news,
    dimensions_cited]."""
    text = Path(path).read_text(encoding="utf-8")
    sections = _split_per_ticker_sections(text)
    rows = []
    in_results = False
    seen: set[str] = set()
    for line in text.splitlines():
        if line.strip().startswith("## Results"):
            in_results = True
            continue
        if not in_results:
            continue
        parsed = _parse_result_row(line)
        if not parsed:
            continue
        sym, dropped, hold_years, confidence = parsed
        if sym in seen:
            continue
        seen.add(sym)
        sec = sections.get(sym, "")
        findings = _extract_findings_list(sec)
        rows.append({
            "symbol": sym,
            "dropped": dropped,
            "expected_hold_years": hold_years,
            "confidence": confidence,
            "business": _extract_step(sec, "Business"),
            "dimensions": _extract_step(sec, "Research dimensions"),
            "key_news": findings,
            "dimensions_cited": ",".join(_extract_dimension_tags(findings)),
        })
    return pd.DataFrame(rows)


def finalize(plan_path: str | Path) -> tuple[pd.DataFrame, Path]:
    plan_path = Path(plan_path)
    scored = parse_dividend_plan(plan_path)
    if scored.empty:
        raise RuntimeError(f"no picks parsed from {plan_path} — fill the Results table")

    dropped = scored[scored["dropped"]]
    if not dropped.empty:
        print(f"[dividend] DROP: excluding {len(dropped)} ticker(s): "
              f"{', '.join(dropped['symbol'].tolist())}")
    scored = scored[~scored["dropped"]].drop(columns=["dropped"])
    if scored.empty:
        raise RuntimeError("all picks dropped")

    bad = scored[scored["expected_hold_years"].isna() | (scored["expected_hold_years"] <= 0)]
    if not bad.empty:
        print(f"[dividend] WARNING: dropping {len(bad)} pick(s) with a missing/"
              f"invalid expected_hold_years: {', '.join(bad['symbol'].tolist())}")
    scored = scored.drop(bad.index)
    if scored.empty:
        raise RuntimeError("no picks with a valid expected_hold_years")

    universe = read_candidates_sidecar(plan_path)
    if universe is not None:
        ref_cols = [c for c in ["symbol", "close", "dividend_yield_ttm",
                                "years_paid_consecutive", "payout_trend",
                                "last_ex_date", "organ_name", "instrument_type"]
                   if c in universe.columns]
        merged = scored.merge(universe[ref_cols], on="symbol", how="left")
    else:
        merged = scored

    merged = add_dividend_price_suggestions(merged)
    # score = yield-weighted sustainability: dividend_yield_ttm * confidence.
    yld = merged.get("dividend_yield_ttm", pd.Series(float("nan"), index=merged.index)).astype(float)
    conf = merged.get("confidence", pd.Series(float("nan"), index=merged.index)).astype(float)
    merged["score"] = (yld.fillna(0.0) * conf.fillna(0.5)).round(6)
    merged = merged.sort_values("score", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1
    # No T+2/T+N target for a hold — the ledger's exit resolution doesn't
    # apply here; this flag is purely informational for downstream tools.
    merged["below_recovery_bar"] = False

    meta = read_meta(plan_path)
    out, sig, _ = write_picks_json(MODE, merged, plan_path, meta,
                                   extra={"weight": None})
    return merged, out
