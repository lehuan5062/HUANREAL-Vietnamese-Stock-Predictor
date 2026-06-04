"""Build a markdown news-research plan that an in-session Claude can execute via WebFetch.

Workflow:
  1. base ML stage selects the actionable candidates
  2. write_plan() produces a markdown checklist with per-ticker URLs
     and a +1 / 0 / -1 sentiment rubric
  3. Claude (running this session) fills in the rubric using WebFetch
  4. parse_plan() reads the filled markdown and returns adjusted scores
  5. modes/claude.py.finalize() re-ranks: adjusted = ml * (1 + news_weight * news)
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from .sources import global_urls, vn_urls


def write_plan(candidates: pd.DataFrame, on: dt.date | None = None,
               run_signature: str | None = None,
               current_horizon: int | None = None,
               current_signature: str | None = None) -> Path:
    """Emit the markdown plan file. `candidates` must include columns
    [symbol, pred_mean, pred_std, close, rsi_14, mom_5, mom_20].
    `run_signature` is appended to the filename so distinct same-day
    runs (different horizon / units / hose-only) don't override each
    other — pass it from the mode caller for full uniqueness."""
    on = on or dt.date.today()
    cfg = load_config().modes["claude"]
    out_dir = reports_dir()
    from ..picks_meta import actionable_suffix
    act_suffix = actionable_suffix(candidates)
    if run_signature:
        path = out_dir / f"claude_news_plan_{on.isoformat()}_{run_signature}{act_suffix}.md"
    else:
        path = out_dir / f"claude_news_plan_{on.isoformat()}{act_suffix}.md"
    from .company_info import enrich
    candidates = enrich(candidates)

    from ..tracking import _next_trading_offset
    from .research_dimensions import ETF_GUIDANCE_MD, REFERENCE_MD

    # ETF rows need a different research rubric. Only inject the ETF guidance
    # block when at least one ETF candidate is present so stocks-only plans
    # keep their existing shape.
    has_etf_candidates = bool(
        "instrument_type" in candidates.columns
        and (candidates["instrument_type"].astype(str).str.upper() == "ETF").any()
    )

    # Sell-day reminder: when current_horizon is known, quote a concrete
    # target sell day so the in-session Claude can offer to schedule a
    # reminder once the picks are finalized. The reminder fires on the
    # sell day itself at 11:30 ICT — late morning, just before the noon
    # lunch break (and, for T+2, 30 min before settlement at noon).
    if current_horizon is not None:
        n = int(current_horizon)
        target_date = _next_trading_offset(pd.Timestamp(on), n).date()
        reminder_date = target_date
        if n == 2:
            sell_window = ("13:00–14:30 ICT (afternoon session, "
                           "after T+2 settlement at noon)")
            reminder_note = "30 min before T+2 settlement at noon"
        else:
            sell_window = "09:00–14:30 ICT (any time during the trading day)"
            reminder_note = "late morning of sell day, before lunch break"
        suggested_time = "11:30 ICT"
        target_iso = target_date.isoformat()
        reminder_iso = reminder_date.isoformat()
    else:
        target_iso = "(unknown — fall back to picks_*.json target_date)"
        reminder_iso = "(unknown — same as sell day)"
        sell_window = "afternoon session ICT"
        suggested_time = "11:30 ICT"
        reminder_note = "late morning of sell day"

    lines = [
        f"# Claude news re-rank plan — {on.isoformat()}",
        "",
        "## Method — emergent research, not a fixed checklist",
        "",
        "For each candidate, you decide what to research. We do not give you",
        "a fixed checklist of dimensions — different companies have different",
        "drivers, and you know better than us what matters for each ticker.",
        "",
        "Per ticker, work in this order:",
        "",
        "1. **Identify the business** from the company name (`organ_name`)",
        "   plus what you already know about the company.",
        "2. **Derive 3-7 research dimensions yourself** for THIS specific",
        "   ticker. They may match common categories (sector, macro, policy,",
        "   geopolitics, calendar) or be idiosyncratic (a single major",
        "   customer, a peer's earnings, a peg, a one-off contract). Write",
        "   them down under Step 2 of the per-ticker section, in your own",
        "   words. **Skip categories that don't apply; add ones that do.**",
        "3. **Research broadly across the dimensions you derived.** Use",
        "   `WebFetch` on the per-ticker URLs we provide as a SEED, then",
        "   `WebSearch` Google for whatever else you need. Also query the",
        "   official Vietnamese government portal (chinhphu.vn) and SBV",
        "   (sbv.gov.vn) when policy / decrees matter for this ticker. Cross-",
        "   check claims across at least 2 sources before scoring.",
        "4. **Score** based on what you actually found. Price/technical noise",
        "   alone = 0.",
        "",
        REFERENCE_MD,
        "",
    ]
    if has_etf_candidates:
        lines.extend([ETF_GUIDANCE_MD, ""])
    lines.extend([
        "Score key:",
        "  +1 = material bullish development (earnings beat, sector tailwind,",
        "       contract win, favorable policy)",
        "   0 = nothing material — or only generic market noise",
        "  -1 = material bearish development (earnings miss, sector headwind,",
        "       regulatory action, dividend cut)",
        "",
        "**Hard override**: if you find a delisting / trading halt / bankruptcy",
        "filing for a ticker, score it `-1` AND write `DROP` in the rationale.",
        "Such tickers should not be traded regardless of the ML score.",
        "",
        f"Re-rank rule: `adjusted = pred_mean * (1 + {cfg['news_weight']} * news_score)`",
        "",
        "When done, fill the score table at the bottom and run:",
        f"  `python -m stockpredict.cli claude-finalize reports/{path.name}`",
        "",
        "## Step 7 — Sell-day reminder (after finalize)",
        "",
        f"Once `claude-finalize` finishes, it prints the explained top-K and a",
        f"`==> SELL-REMINDER:` block. **If at least one finalized pick is",
        f"`actionable: True`**, ask the user — in plain conversation — whether",
        f"they would like a reminder scheduled in **GMT+7 (Asia/Ho_Chi_Minh,",
        f"Vietnamese ICT)** to prepare the exit.",
        "",
        f"- **Sell day** (the actual trade): {target_iso} ({sell_window}).",
        f"- **Reminder fires**: {reminder_iso} {suggested_time} — on the sell",
        f"  day itself ({reminder_note}). This gives the user time to review",
        f"  and queue exit orders for the afternoon session.",
        "",
        "If the user says yes, use whatever scheduling tool you have available",
        "(Claude Code's scheduled-tasks / cron / `at` / Windows `schtasks`),",
        "or — if no scheduler is available — give the user a paste-ready ICS",
        "calendar event with `TZID=Asia/Ho_Chi_Minh`. Do NOT silently set the",
        "reminder; always confirm reminder date+time, sell day, tickers, and",
        "method first.",
        "",
        "Skip Step 7 entirely when no pick is actionable.",
        "",
        "## Global / macro context (read once)",
        "",
    ])
    for name, url in global_urls().items():
        lines.append(f"- [{name}]({url})")
    lines.append("")
    lines.append("## Candidates (ranked by ML)")
    lines.append("")

    for _, row in candidates.iterrows():
        sym = row["symbol"]
        organ = row.get("organ_name", "") or "(name unknown — infer from ticker)"
        row_type = str(row.get("instrument_type", "STOCK") or "STOCK").upper()
        is_etf_row = (row_type == "ETF")
        type_tag = "  [ETF — apply ETF rubric, NOT company business]" if is_etf_row else ""
        lines.append(f"### {sym}  —  {organ}{type_tag}")
        lines.append("")
        lines.append(f"ML signal: pred_mean={row['pred_mean']:+.4f}  "
                     f"(±{row.get('pred_std', 0):.4f})  close={row.get('close', float('nan')):.0f}  "
                     f"rsi={row.get('rsi_14', float('nan')):.0f}  "
                     f"mom20={row.get('mom_20', float('nan')):+.3f}")
        # Position-sized pricing if available
        if "entry_vnd" in row and pd.notna(row.get("entry_vnd")):
            entry = int(row["entry_vnd"])
            tgt = int(row["target_vnd"])
            stop = int(row["stop_vnd"])
            fees = int(row.get("fees_round_trip_vnd", 0))
            net = int(row.get("net_reward_vnd", 0))
            rr = row.get("rr_ratio", float("nan"))
            pos = int(row.get("position_units", 100))
            actionable = bool(row.get("actionable", False))
            net_sign = "+" if net >= 0 else ""
            lines.append(
                f"Trade ({pos} units): entry={entry:,}  target={tgt:,}  stop={stop:,}  "
                f"fees={fees:,}  net={net_sign}{net:,}  rr={rr:.2f}  "
                f"{'ACTIONABLE' if actionable else 'skip (rr/net too low)'}"
            )
        lines.append("")
        if is_etf_row:
            lines.append("**Step 1 — Underlying index**: name the index this ETF tracks (e.g. VN30 / VN Diamond / VN100 / VN Midcap / VNFIN Lead) and the fund manager. **Do NOT describe a company business — this is a passive basket.**")
        else:
            lines.append("**Step 1 — Business**: in one line, write what this company does and the 1-2 main revenue lines.")
        lines.append("")
        lines.append("- ")
        lines.append("")
        if is_etf_row:
            lines.append("**Step 2 — Research dimensions (ETF)**: pick 3-5 from {index performance, foreign net flows, VSDC creation/redemption, NAV premium/discount, upcoming rebalancing, top-weight constituent binary events within the T+N exit horizon}. Skip those that don't apply, add ones we haven't listed.")
        else:
            lines.append("**Step 2 — Research dimensions**: derive 3-7 dimensions YOU think matter for THIS ticker on a T+2 horizon. Your own list — not ours. Skip categories that don't apply, add ones that do (idiosyncratic drivers like a key customer, a peer's earnings, a peg, a contract often matter more than any standard category).")
        lines.append("")
        lines.append("- ")
        lines.append("")
        lines.append("**Step 3 — Research findings per dimension**:")
        for name, url in vn_urls(sym).items():
            lines.append(f"- [{name}]({url})")
        lines.append("")
        lines.append("**Step 4 — Findings** (one bullet per dimension you investigated, tagged `[dimension-name]`, with dates and sources):")
        lines.append("")
        lines.append("- ")
        lines.append("")

    lines.append("## Scores")
    lines.append("")
    lines.append("Fill the `news_score` column with one of:")
    lines.append("  `+1` bullish, `0` neutral, `-1` bearish, or `DROP` to exclude entirely.")
    lines.append("Use `DROP` for delisting / suspension / bankruptcy / known fraud — these")
    lines.append("override the ML score and are never traded.")
    lines.append("")
    lines.append("| symbol | pred_mean | news_score |")
    lines.append("| --- | --- | --- |")
    for _, row in candidates.iterrows():
        lines.append(f"| {row['symbol']} | {row['pred_mean']:+.4f} | 0 |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_SCORE_ROW = re.compile(
    r"^\|\s*([A-Z0-9]+)\s*\|\s*([+\-]?\d*\.?\d+)\s*\|\s*(DROP|[+\-]?\d+)\s*\|\s*$",
    re.IGNORECASE,
)

# Sentinel: news_score == DROP_SENTINEL means "exclude entirely; never trade".
DROP_SENTINEL = -999


def _split_per_ticker_sections(text: str) -> dict[str, str]:
    """Slice the plan markdown into a {ticker: section_text} dict using the
    `### TICKER  —  Company` headings emitted by write_plan."""
    sections: dict[str, str] = {}
    current_sym: str | None = None
    current_lines: list[str] = []
    heading_re = re.compile(r"^###\s+([A-Z0-9]{2,8})\s+(?:—|--).*$")
    for line in text.splitlines():
        m = heading_re.match(line)
        if m:
            if current_sym:
                sections[current_sym] = "\n".join(current_lines)
            current_sym = m.group(1)
            current_lines = []
            continue
        if current_sym is not None:
            # Stop accumulating once we hit the global "## Scores" footer
            if line.strip().startswith("## Scores"):
                sections[current_sym] = "\n".join(current_lines)
                current_sym = None
                current_lines = []
                continue
            current_lines.append(line)
    if current_sym:
        sections[current_sym] = "\n".join(current_lines)
    return sections


def _extract_step(section: str, step_label: str) -> str:
    """Pull the user-written text under a `**Step N — <label>**` heading.
    Returns the joined non-empty bullet/paragraph lines, stripped."""
    pat = re.compile(
        rf"\*\*Step \d+ — {re.escape(step_label)}\*\*[^\n]*\n(.*?)(?=\*\*Step \d+ —|\Z)",
        re.DOTALL,
    )
    m = pat.search(section)
    if not m:
        return ""
    body = m.group(1)
    # Drop empty bullets ("-") and leading bullet markers
    out_lines: list[str] = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s or s == "-":
            continue
        if s.startswith("- "):
            s = s[2:].strip()
        out_lines.append(s)
    return " ".join(out_lines).strip()


def _extract_findings_list(section: str) -> list[str]:
    """Same as _extract_step('Findings') but keep each bullet as a list item."""
    pat = re.compile(
        r"\*\*Step \d+ — Findings\*\*[^\n]*\n(.*?)(?=\*\*Step \d+ —|\Z)",
        re.DOTALL,
    )
    m = pat.search(section)
    if not m:
        return []
    out: list[str] = []
    for raw in m.group(1).splitlines():
        s = raw.strip()
        if not s or s == "-":
            continue
        if s.startswith("- "):
            s = s[2:].strip()
        out.append(s)
    return out


# Dimension tags Claude writes in the form "[tag-name]" at the start of each
# Step 4 bullet. We extract them so the ledger can later aggregate hit-rate
# by dimension category and feed it back into the next run's prompt.
_DIMENSION_TAG_RE = re.compile(r"\[([a-z0-9][a-z0-9_/.+-]*)\]", re.IGNORECASE)


def _extract_dimension_tags(findings: list[str]) -> list[str]:
    """Pull `[tag-name]` markers out of Step 4 bullets, deduped, lower-cased,
    in first-seen order.

    Convention: each finding bullet starts with one bracket-tagged dimension
    name (kebab-case) — e.g. "[insider-action] Deputy GM registered to buy ...".
    Tags inside the body of a bullet (like a literal "[some text]" citation)
    will also be picked up; that's a known acceptable noise source for the
    ledger aggregation, since the alternative — anchoring strictly to bullet
    start — fails when Claude wraps lines or indents.

    Findings without any tag are skipped (they don't contribute to the
    by-dimension aggregation either way).
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for bullet in findings:
        for m in _DIMENSION_TAG_RE.finditer(bullet):
            tag = m.group(1).strip().lower()
            # Skip degenerate tags — empty, pure-numeric (often citation
            # markers like "[1]"), or "drop"/"net" sentinels that Claude
            # uses for the score-summary bullet at the end of Step 4.
            if not tag or tag.isdigit() or tag in {"drop", "net"}:
                continue
            if tag in seen_set:
                continue
            seen_set.add(tag)
            seen.append(tag)
    return seen


def parse_plan(path: str | Path) -> pd.DataFrame:
    """Read the filled markdown plan and return DataFrame[symbol, pred_mean,
    news_score, business, drivers, key_news]. news_score is `DROP_SENTINEL`
    for any row scored DROP."""
    text = Path(path).read_text(encoding="utf-8")
    sections = _split_per_ticker_sections(text)
    rows = []
    in_scores = False
    for line in text.splitlines():
        if line.strip().startswith("## Scores"):
            in_scores = True
            continue
        if not in_scores:
            continue
        m = _SCORE_ROW.match(line)
        if m:
            sym = m.group(1).upper()
            score_str = m.group(3).upper()
            if score_str == "DROP":
                ns = DROP_SENTINEL
            else:
                ns = int(score_str)
            sec = sections.get(sym, "")
            findings = _extract_findings_list(sec)
            rows.append({
                "symbol": sym,
                "pred_mean": float(m.group(2)),
                "news_score": ns,
                "business": _extract_step(sec, "Business"),
                # Step 2 was renamed from "Key drivers" to "Research dimensions"
                # — these are the dimensions Claude derived for THIS ticker.
                "dimensions": _extract_step(sec, "Research dimensions")
                              or _extract_step(sec, "Key drivers"),
                "key_news": findings,
                # dimension tags actually CITED in Step 4 (which dimensions
                # Claude found data for, vs. just listed as planned in Step 2).
                # Stored as comma-separated string so it round-trips through
                # parquet without list-type quirks. Empty string = no tags.
                "dimensions_cited": ",".join(_extract_dimension_tags(findings)),
            })
    return pd.DataFrame(rows)
