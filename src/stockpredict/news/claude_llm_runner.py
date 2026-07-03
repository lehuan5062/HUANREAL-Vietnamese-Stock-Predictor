"""Build a markdown plan for the LLM-ONLY Claude mode.

Unlike the hybrid plan (``claude_runner.write_plan``), there is NO statistical
model here: the candidates are the WHOLE mechanically-filtered downtrend universe
(uncapped), and the in-session Claude does the whole job — select which names to
buy and, for each, predict **N** (trading days to bounce back to profit) and
**P** (the profit at that bounce, as a return fraction). Finalize then computes
``score = P / N`` and ranks by it — the SAME objective the base/hybrid modes use
(theirs is P/N × recovery_prob; the LLM's selection vetting stands in for the
probability). As in the rest of the rebound pipeline, you BUY AT TODAY'S CLOSE,
the target is ``close × (1 + P)``, and there is NO stop-loss (hold until the
target).

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
    _extract_dimension_tags,
    _extract_findings_list,
    _extract_step,
    _parse_price_cell,
    _split_per_ticker_sections,
)
from .sources import global_urls, vn_urls


def write_llm_plan(universe: pd.DataFrame, on: dt.date | None = None,
                   run_signature: str | None = None,
                   n_picks: int = 5) -> Path:
    """Emit the LLM-only markdown plan. ``universe`` is the full eligible
    downtrend cross-section (any of [symbol, close, rsi_14, mom_20,
    instrument_type]); no pricing columns are required. ``n_picks`` is how many
    names the LLM should ultimately surface."""
    on = on or dt.date.today()
    out_dir = reports_dir()
    if run_signature:
        path = out_dir / f"claude_llm_plan_{on.isoformat()}_{run_signature}.md"
    else:
        path = out_dir / f"claude_llm_plan_{on.isoformat()}.md"

    from .company_info import enrich
    universe = enrich(universe)

    horizon_txt = "the expected holding window (a few days to a couple of weeks)"

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
        f"1. **Select** the best **{int(n_picks)}** name(s) to rebound from the",
        "   universe table, using your own research — fundamentals, news, sector,",
        "   macro, technicals. You may research as many candidates as you need.",
        "2. **Predict N and P** for each chosen name, from your research:",
        "   `N_days` = expected TRADING days until it bounces back to a profitable",
        "   point; `P` = the expected profit at that point, as a decimal return",
        "   fraction (e.g. `0.05` = +5%; `5%` also accepted). P must clear the",
        "   round-trip fee bar (~0.95%) or the pick is flagged weak.",
        "3. Finalize computes `score = P / N` (profit per day held) and **ranks",
        "   your picks by it** — same objective as the base/hybrid modes. You BUY",
        "   AT TODAY'S CLOSE (the `close` column, ×1000 for VND); the sell target",
        "   is `close × (1 + P)`. No entry price, no stop — hold until the target.",
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
        "labels). Use WebFetch / WebSearch, cross-check at least 2 sources, and",
        "tag each finding with `[dimension]`.",
        "",
        "Seed sources you can reuse per ticker (replace TICKER):",
        "",
    ]
    # Show the seed-URL shape once using a placeholder ticker. NOTE: this list
    # must sit ABOVE the section template — the LLM appends its `### TICKER`
    # sections after the template, and the section splitter accumulates until
    # the next `## ` heading, so any stray bullets between the sections and
    # `## Results` would leak into the last ticker's findings.
    for name, url in vn_urls("TICKER").items():
        lines.append(f"- [{name}]({url})")

    lines += [
        "",
        "Section template — append your filled sections directly below it,",
        "immediately before `## Results`:",
        "",
        "```",
        "### TICKER  —  Company name",
        "",
        "**Step 1 — Business**: one line on what the company does.",
        "- ",
        "",
        "**Step 2 — Research dimensions**: the 3-7 drivers you judged matter for",
        "THIS ticker's rebound.",
        "- ",
        "",
        "**Step 4 — Findings** (one bullet per dimension, tagged `[dimension-name]`,",
        "with dates + sources):",
        "- ",
        "```",
        "",
        "## Results — fill this with your chosen picks",
        "",
        "One row per pick, ordered however you like (the finalize step computes",
        "`score = P / N` and re-sorts, highest first). `N_days` = expected trading",
        "days to the bounce (>= 1); `P` = expected profit as a decimal fraction",
        "(`0.05` = +5%; a `5%` cell is also accepted). You buy at the close; the",
        "target is `close × (1 + P)`; no stop. Write `DROP` in `N_days` to",
        "exclude a row you listed.",
        "",
        "| rank | symbol | N_days | P |",
        "| --- | --- | --- | --- |",
    ]
    for i in range(int(n_picks)):
        lines.append(f"| {i + 1} |  |  |  |")
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


def _parse_profit_cell(cell: str) -> float:
    """Parse the P cell into a decimal return fraction. Accepts ``0.05`` or
    ``5%`` (percent form divided by 100). Blank / unparseable → NaN."""
    s = (cell or "").strip().replace(",", "")
    if not s or s == "-":
        return float("nan")
    pct = s.endswith("%")
    if pct:
        s = s[:-1].strip()
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return v / 100.0 if pct else v


def _parse_result_row(line: str):
    """Parse one results-table row: ``| rank | symbol | N_days | P |``.
    Returns (symbol, dropped, pred_days, pred_profit) or None for non-data rows
    (header / separator / blank cells). ``DROP`` in the N_days cell marks the
    row excluded."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    if len(cells) < 4:
        return None
    sym = cells[1].upper()
    if not _SYM_RE.match(sym):
        return None
    n_str = cells[2].upper()
    if n_str == "DROP":
        return sym, True, float("nan"), float("nan")
    if not _NUM_RE.match(n_str):
        return None
    pred_days = float(n_str)
    pred_profit = _parse_profit_cell(cells[3])
    return sym, False, pred_days, pred_profit


def parse_llm_plan(path: str | Path) -> pd.DataFrame:
    """Read the filled LLM-only plan and return DataFrame[symbol, dropped,
    pred_days, pred_profit, business, dimensions, key_news, dimensions_cited].
    Entry is the close and the target is ``close × (1 + pred_profit)``, both set
    at finalize; there is no stop. ``dropped=True`` marks a DROP row."""
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
        sym, dropped, pred_days, pred_profit = parsed
        if sym in seen:
            continue
        seen.add(sym)
        sec = sections.get(sym, "")
        findings = _extract_findings_list(sec)
        rows.append({
            "symbol": sym,
            "dropped": dropped,
            "pred_days": pred_days,
            "pred_profit": pred_profit,
            "business": _extract_step(sec, "Business"),
            "dimensions": _extract_step(sec, "Research dimensions")
                          or _extract_step(sec, "Key drivers"),
            "key_news": findings,
            "dimensions_cited": ",".join(_extract_dimension_tags(findings)),
        })
    return pd.DataFrame(rows)
