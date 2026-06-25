"""Build a markdown plan for the LLM-ONLY Claude mode.

Unlike the hybrid plan (``claude_runner.write_plan``), there is NO ML mean head
here: the candidates are the WHOLE mechanically-filtered universe (uncapped, no
``pred_mean``), and the in-session Claude does the entire job — select which
names to buy, rank them by its own conviction, and set the entry / target / stop
prices itself from its research.

Workflow:
  1. ``predict.eligible_universe`` produces the filtered universe (no scoring)
  2. ``write_llm_plan`` emits a universe reference table + instructions + an
     empty results table the LLM fills with its chosen picks and prices
  3. Claude researches, selects, ranks, prices (WebSearch / WebFetch)
  4. ``parse_llm_plan`` reads the filled results table + per-pick sections
  5. ``modes/claude.py.finalize_llm`` ranks by conviction (no ML multiplier),
     applies the LLM prices, and writes the picks JSON.

The per-ticker section format (``### TICKER — Company`` with Step 1 / 2 / 4) and
the ``[dimension-tag]`` convention are deliberately identical to the hybrid plan
so the same parsing helpers in ``claude_runner`` are reused.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

from ..config import reports_dir
from .claude_runner import (
    DROP_SENTINEL,
    _extract_dimension_tags,
    _extract_findings_list,
    _extract_step,
    _parse_price_cell,
    _split_per_ticker_sections,
)
from .sources import global_urls, vn_urls


def write_llm_plan(universe: pd.DataFrame, on: dt.date | None = None,
                   run_signature: str | None = None,
                   current_horizon: int | None = None,
                   n_picks: int = 5) -> Path:
    """Emit the LLM-only markdown plan. ``universe`` is the full eligible
    cross-section (any of [symbol, close, rsi_14, mom_20, instrument_type]);
    NO ``pred_mean`` / pricing columns are required. ``n_picks`` is how many
    names the LLM should ultimately surface."""
    on = on or dt.date.today()
    out_dir = reports_dir()
    if run_signature:
        path = out_dir / f"claude_llm_plan_{on.isoformat()}_{run_signature}.md"
    else:
        path = out_dir / f"claude_llm_plan_{on.isoformat()}.md"

    from .company_info import enrich
    universe = enrich(universe)

    horizon_txt = (f"~{int(current_horizon)} trading day(s)"
                   if current_horizon is not None else "the holding window")

    lines = [
        f"# Claude LLM-only pick plan — {on.isoformat()}",
        "",
        "## Method — YOU pick everything (no ML ranking)",
        "",
        "This is the **LLM-only** path: there is **no machine-learning model** in",
        "the loop. The table below is the ENTIRE mechanically-eligible universe",
        "(liquidity / tradable / ceiling / corporate-action filtered) — it is NOT",
        "ranked or pre-scored. Your job is to do the whole pick:",
        "",
        f"1. **Select** the best **{int(n_picks)}** name(s) for a T+2 trade from the",
        "   universe table, using your own research — fundamentals, news, sector,",
        "   macro, technicals. You may research as many candidates as you need.",
        "2. **Rank** your chosen names by a numeric **conviction** score (higher =",
        "   stronger). The final picks JSON is ordered by this score.",
        "3. **Price each pick yourself** — set `entry_vnd`, `target_vnd`, `stop_vnd`",
        "   in VND PER SHARE based on your research (support/resistance, ATR-sized",
        "   risk, the catalyst). There is no mechanical limit-dip here; the entry",
        "   MAY sit above today's close if a catalyst warrants guaranteeing a fill.",
        "4. For each chosen name, write a `### TICKER — Company` section (template",
        "   below) documenting the business, the dimensions you researched, and",
        "   your findings — then fill the results table at the bottom.",
        "",
        "**Hard override**: if you find a delisting / trading halt / bankruptcy",
        "filing, do NOT pick the name (or write `DROP` in its conviction cell).",
        "",
        "## Global / macro context (read once, before picking)",
        "",
        "**Major-conflict / geopolitical check.** Scan for major global conflicts "
        "or geopolitical shocks breaking today (wars, ceasefires, sanctions / "
        "tariffs, oil-supply or shipping disruptions, sharp oil / gold / USD-VND "
        "moves). A market-wide catalyst can move the whole VN-Index and specific "
        "sectors regardless of any single company; carry it into every pick.",
        "",
        f"**VN-Index trend call.** Research where the VN-Index is likely headed over "
        f"{horizon_txt}: recent trend / momentum, position vs 50/200-day MAs, "
        "breadth, foreign net buy/sell, scheduled macro events. State an explicit "
        "UP / SIDEWAYS / DOWN view with confidence and let it tilt every pick "
        "(be conservative in a likely-DOWN tape; remember entries may need to be "
        "higher in a melt-up).",
        "",
    ]
    for name, url in global_urls().items():
        lines.append(f"- [{name}]({url})")
    lines += [
        "",
        "## Universe (UNRANKED — the full eligible set)",
        "",
        "| symbol | company | close | rsi_14 | mom_20 | type |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in universe.iterrows():
        sym = str(row["symbol"]).upper()
        organ = (row.get("organ_name", "") or "").replace("|", "/")
        close = row.get("close", float("nan"))
        rsi = row.get("rsi_14", float("nan"))
        mom20 = row.get("mom_20", float("nan"))
        rtype = str(row.get("instrument_type", "STOCK") or "STOCK").upper()
        close_s = f"{close:.0f}" if pd.notna(close) else ""
        rsi_s = f"{rsi:.0f}" if pd.notna(rsi) else ""
        mom_s = f"{mom20:+.3f}" if pd.notna(mom20) else ""
        lines.append(f"| {sym} | {organ} | {close_s} | {rsi_s} | {mom_s} | {rtype} |")

    lines += [
        "",
        "## Per-pick research sections",
        "",
        "For EACH name you choose, add a section in this exact format (the",
        "finalize parser keys on the `### TICKER` heading and the `**Step N —`",
        "labels). Seed URLs per ticker are listed — use WebFetch / WebSearch,",
        "cross-check at least 2 sources, and tag each finding with `[dimension]`.",
        "",
        "```",
        "### TICKER  —  Company name",
        "",
        "**Step 1 — Business**: one line on what the company does.",
        "- ",
        "",
        "**Step 2 — Research dimensions**: the 3-7 drivers you judged matter for",
        "THIS ticker on a T+2 horizon.",
        "- ",
        "",
        "**Step 4 — Findings** (one bullet per dimension, tagged `[dimension-name]`,",
        "with dates + sources):",
        "- ",
        "```",
        "",
        "Seed sources you can reuse per ticker (replace TICKER):",
        "",
    ]
    # Show the seed-URL shape once using a placeholder ticker.
    for name, url in vn_urls("TICKER").items():
        lines.append(f"- [{name}]({url})")

    lines += [
        "",
        "## Results — fill this with your chosen picks",
        "",
        "One row per pick, ordered however you like (the finalize step re-sorts by",
        "`conviction`, highest first). `conviction` is any positive number on a",
        "consistent scale; `entry_vnd` / `target_vnd` / `stop_vnd` are VND per",
        "share. Write `DROP` in `conviction` to exclude a row you listed.",
        "",
        "| rank | symbol | conviction | entry_vnd | target_vnd | stop_vnd |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for i in range(int(n_picks)):
        lines.append(f"| {i + 1} |  |  |  |  |  |")
    lines += [
        "",
        "When done, run:",
        f"  `python -m stockpredict.cli claude-finalize reports/{path.name}`",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_SYM_RE = re.compile(r"^[A-Z0-9]{2,8}$")
_NUM_RE = re.compile(r"^[+\-]?\d+(?:\.\d+)?$")


def _parse_result_row(line: str):
    """Parse one results-table row: ``| rank | symbol | conviction | entry |
    target | stop |``. Returns (symbol, conviction|DROP_SENTINEL, entry, target,
    stop) or None for non-data rows (header / separator / blank cells)."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    if len(cells) < 6:
        return None
    sym = cells[1].upper()
    if not _SYM_RE.match(sym):
        return None
    conv_str = cells[2].upper()
    if conv_str == "DROP":
        conviction = float(DROP_SENTINEL)
    elif _NUM_RE.match(conv_str):
        conviction = float(conv_str)
    else:
        return None
    entry = _parse_price_cell(cells[3])
    target = _parse_price_cell(cells[4])
    stop = _parse_price_cell(cells[5])
    return sym, conviction, entry, target, stop


def parse_llm_plan(path: str | Path) -> pd.DataFrame:
    """Read the filled LLM-only plan and return DataFrame[symbol, conviction,
    entry_vnd, target_vnd, stop_vnd, business, dimensions, key_news,
    dimensions_cited]. ``conviction == DROP_SENTINEL`` marks a dropped row."""
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
        sym, conviction, entry, target, stop = parsed
        if sym in seen:
            continue
        seen.add(sym)
        sec = sections.get(sym, "")
        findings = _extract_findings_list(sec)
        rows.append({
            "symbol": sym,
            "conviction": conviction,
            "entry_vnd": entry,
            "target_vnd": target,
            "stop_vnd": stop,
            "business": _extract_step(sec, "Business"),
            "dimensions": _extract_step(sec, "Research dimensions")
                          or _extract_step(sec, "Key drivers"),
            "key_news": findings,
            "dimensions_cited": ",".join(_extract_dimension_tags(findings)),
        })
    return pd.DataFrame(rows)
