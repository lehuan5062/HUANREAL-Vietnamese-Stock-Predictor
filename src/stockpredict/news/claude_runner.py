"""Build a markdown news-research plan that an in-session Claude can execute via WebFetch.

Workflow (rebound strategy):
  1. the rebound model selects the top-N downtrend candidates by P/N score
  2. write_plan() produces a markdown checklist with per-ticker URLs
     and a +1 / 0 / -1 / DROP vetting rubric ("healthy bounce vs falling knife")
  3. Claude (running this session) fills in the rubric using WebFetch
  4. parse_plan() reads the filled markdown and returns the news scores
  5. modes/claude.py.finalize() re-ranks: adjusted = score * (1 + news_weight * news)
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
               current_signature: str | None = None,
               ab_verdict: str | None = None) -> Path:
    """Emit the markdown plan file. `candidates` must include columns
    [symbol, score, pred_days, pred_profit, pred_recovery_prob, close, rsi_14,
    mom_5, mom_20, high_prox_20].
    `run_signature` is appended to the filename so distinct same-day
    runs (hose-only / exclude) don't override each other — pass it from the
    mode caller for full uniqueness."""
    on = on or dt.date.today()
    cfg = load_config().modes["claude"]
    out_dir = reports_dir()
    from ..picks_meta import picks_suffix
    pk_suffix = picks_suffix(candidates)
    if run_signature:
        path = out_dir / f"claude_news_plan_{on.isoformat()}_{run_signature}{pk_suffix}.md"
    else:
        path = out_dir / f"claude_news_plan_{on.isoformat()}{pk_suffix}.md"
    from .company_info import enrich
    candidates = enrich(candidates)

    from .research_dimensions import ETF_GUIDANCE_MD, REFERENCE_MD

    # ETF rows need a different research rubric. Only inject the ETF guidance
    # block when at least one ETF candidate is present so stocks-only plans
    # keep their existing shape.
    has_etf_candidates = bool(
        "instrument_type" in candidates.columns
        and (candidates["instrument_type"].astype(str).str.upper() == "ETF").any()
    )

    has_union = "missed_only" in candidates.columns
    lines = [
        f"# Claude news re-rank plan — {on.isoformat()}",
        "",
    ]
    if has_union or ab_verdict:
        lines += ["## Two rankings to weigh — standard + missed-winners variant", ""]
        if ab_verdict:
            lines += [f"**A/B backtest verdict:** {ab_verdict}.", ""]
        lines += [
            "Candidates below are the UNION of the standard model's top picks and "
            "the missed-winners variant's top picks. A `[also-missed]` tag means "
            "BOTH models surfaced it; `[missed-only]` means only the experimental "
            "variant did. **Weigh the standard ranking higher** (it wins the A/B "
            "above) — only let a `[missed-only]` name into your final picks if the "
            "news strongly supports it. Your job: research the union, then choose "
            "the final N on conviction.", "",
        ]
    lines += [
        "## Method — vet the rebound, emergent research (not a fixed checklist)",
        "",
        "These are all DOWNTREND names the rebound model judged statistically",
        "likely to bounce back to a small profit (they cleared a per-ticker",
        "recovery-probability filter). **Your job is the human check the",
        "statistics can't do: is each one a healthy company in a temporary dip",
        "that will recover — or a falling knife (fraud, delisting, insolvency,",
        "structural decline) where the drop is justified and the bounce won't",
        "come?** Confirm the healthy ones (+1), flag the broken ones (-1), and",
        "`DROP` the un-tradeable ones outright.",
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
        "Score key — you are VETTING THE BOUNCE (is this a healthy pullback that",
        "will recover, or a falling knife that keeps dropping?):",
        "  +1 = news supports the rebound (a real recovery catalyst, or simply a",
        "       fundamentally sound company in a temporary/technical dip)",
        "   0 = nothing material — the statistical rebound case stands on its own",
        "  -1 = news works AGAINST the rebound (deteriorating fundamentals, sector",
        "       headwind, dilution, governance concern — the dip may be justified)",
        "",
        "**Hard override**: if you find a delisting / trading halt / bankruptcy",
        "filing / fraud for a ticker, write `DROP` — it must not be traded no",
        "matter how attractive the P/N score looks (this is exactly the falling",
        "knife the statistical filter can miss).",
        "",
        f"Re-rank rule: `adjusted = score * (1 + {cfg['news_weight']} * news_score)`",
        "",
        "When done, fill the score table at the bottom and run:",
        f"  `python -m stockpredict.cli claude-finalize reports/{path.name}`",
        "",
        "## Step 7 — Exit is flexible (hold until the target)",
        "",
        "This is a REBOUND trade with a **flexible exit**: there is no fixed",
        "sell day. Each pick shows a target price (`target_vnd`) and an expected",
        "hold (`N≈…d to bounce`). The user **monitors and sells manually** when",
        "the price reaches the target — that human judgement is deliberate.",
        "",
        "So do NOT schedule a hard T+2 sell reminder. Instead, after",
        "`claude-finalize`, tell the user per pick: the buy price, the target,",
        "and the expected days-to-bounce. If — and only if — the user asks for a",
        "nudge, offer an OPTIONAL check-in reminder around `as_of + N` trading",
        "days (in GMT+7, Asia/Ho_Chi_Minh) to re-examine any pick that hasn't",
        "recovered yet — framed as 'take a look', not 'sell now'. Never schedule",
        "silently; confirm date/time and tickers first.",
        "",
        "## Global / macro context (read once)",
        "",
        "**Major-conflict / geopolitical check — do this ONCE, before scoring "
        "any ticker.** Scan for major global conflicts or geopolitical shocks "
        "breaking or escalating today: wars, ceasefires / peace treaties, new "
        "sanctions or tariffs, oil-supply or shipping-route disruptions, sharp "
        "oil / gold / USD-VND moves. A market-wide geopolitical catalyst can "
        "move the entire VN-Index — and specific sectors (oil & gas, shipping "
        "/ logistics, exporters, gold, fertiliser) — regardless of any single "
        "company's news. If you find one, note it in the global context and "
        "carry it into EVERY ticker's `news_score` AND its `adj_entry_vnd` / "
        "`adj_target_vnd` (a broad risk-on melt-up means dip-limits won't fill, "
        "so raise the adjusted entry; a risk-off shock means gaps down). If "
        "today is geopolitically quiet, say so and move on.",
        "",
        "**VN-Index trend call — do this ONCE, before scoring any ticker.** "
        "Research where the VN-Index is likely headed over the expected holding "
        "window (a few days to a couple of weeks — see each pick's `N≈…d`). "
        "Look at: the index's recent trend and momentum "
        "(last few sessions + last few weeks), where it sits vs its 50/200-day "
        "moving averages and recent support/resistance, market breadth (are most "
        "stocks rising, or is the index propped up by a few large caps — 'xanh vỏ "
        "đỏ lòng'?), foreign-investor net buy/sell, liquidity, and scheduled macro "
        "events (SBV rates, FX, FTSE/MSCI review, big earnings). Use WebSearch + "
        "WebFetch on cafef.vn, vietstock.vn, fireant/fialda, TradingView VNINDEX, "
        "plus the VN-Index news sources below. **State an explicit directional "
        "view — UP / SIDEWAYS / DOWN for the holding window, with a confidence "
        "(low/med/high) and one line of reasoning — in the global context, and "
        "echo it in `global_summary` at finalize.** Then let it tilt EVERY ticker: "
        "in a likely-DOWN tape be more conservative (favour defensive / "
        "counter-trend names, lower `news_score` for high-beta names that just "
        "track the index, don't chase); in a likely-UP tape be more constructive "
        "and remember dip-limits may not fill (consider raising `adj_entry_vnd`). "
        "A stock that usually moves OPPOSITE the index, or has a strong "
        "company-specific catalyst, can override the index call — say so in its "
        "rationale.",
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
        if bool(row.get("missed_only", False)):
            type_tag += "  [missed-only — variant pick; needs strong news support]"
        elif bool(row.get("also_missed", False)):
            type_tag += "  [also-missed — both models like it]"
        lines.append(f"### {sym}  —  {organ}{type_tag}")
        lines.append("")
        lines.append(f"Rebound signal: score={row.get('score', float('nan')):.4f}  "
                     f"N≈{row.get('pred_days', float('nan')):.0f}d to bounce  "
                     f"P≈{row.get('pred_profit', float('nan')):+.3f}  "
                     f"recovery_prob={row.get('pred_recovery_prob', float('nan')):.0%}  "
                     f"close={row.get('close', float('nan')):.0f}  "
                     f"rsi={row.get('rsi_14', float('nan')):.0f}  "
                     f"mom20={row.get('mom_20', float('nan')):+.3f}  "
                     f"below20dHigh={row.get('high_prox_20', float('nan')):+.3f}")
        # Per-share pricing if available (rebound: buy at close, no stop / rr).
        if "close_vnd" in row and pd.notna(row.get("close_vnd")):
            entry = int(row["close_vnd"])
            tgt = int(row["target_vnd"])
            hold = row.get("hold_days")
            hold_s = f"{int(hold)}d" if pd.notna(hold) else "?"
            fees = int(row.get("fees_round_trip_vnd", 0))
            net = int(row.get("net_reward_vnd", 0))
            below = bool(row.get("below_recovery_bar", False))
            net_sign = "+" if net >= 0 else ""
            lines.append(
                f"Trade (per share): buy={entry:,}  target={tgt:,}  hold≈{hold_s}  "
                f"fees={fees:,}  net={net_sign}{net:,}  "
                f"{'BELOW RECOVERY BAR (weak)' if below else 'OK'}"
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
            lines.append("**Step 2 — Research dimensions**: derive 3-7 dimensions YOU think matter for THIS ticker's REBOUND — will it bounce back to a small profit within the next couple of weeks, or is it a broken company that keeps falling? Your own list — not ours. Skip categories that don't apply, add ones that do (idiosyncratic drivers like a key customer, a peer's earnings, a peg, a contract, or a solvency/dilution/fraud red flag often matter more than any standard category).")
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
    lines.append("**News-adjusted entry / target (optional).** The `adj_entry_vnd` and")
    lines.append("`adj_target_vnd` columns are pre-filled with the rebound buy price")
    lines.append("(today's close) and the profit target. These do NOT replace the")
    lines.append("mechanical prices — they add a parallel, news-aware trade the user can")
    lines.append("compare. If your research says the stock will gap up or down on a")
    lines.append("catalyst (so the plain close-entry / target no longer fits), overwrite")
    lines.append("these two cells with the entry and target you'd actually place in VND")
    lines.append("per share. Leave them as-is (or blank) to keep the mechanical prices.")
    lines.append("")
    lines.append("| symbol | score | news_score | adj_entry_vnd | adj_target_vnd |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in candidates.iterrows():
        ae = int(row["close_vnd"]) if "close_vnd" in row and pd.notna(row.get("close_vnd")) else ""
        at = int(row["target_vnd"]) if "target_vnd" in row and pd.notna(row.get("target_vnd")) else ""
        sc = row.get("score", float("nan"))
        lines.append(f"| {row['symbol']} | {sc:.4f} | 0 | {ae} | {at} |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# Sentinel: news_score == DROP_SENTINEL means "exclude entirely; never trade".
DROP_SENTINEL = -999

_SYM_RE = re.compile(r"^[A-Z0-9]{2,8}$")
_INT_SCORE_RE = re.compile(r"^[+\-]?\d+$")


def _parse_price_cell(cell: str) -> float:
    """Parse an optional adj_entry/adj_target cell into a float VND value.
    Blank / '-' / unparseable → NaN (caller falls back to the mechanical
    price). Commas and a stray VND suffix are tolerated."""
    s = (cell or "").strip().replace(",", "").replace("đ", "").replace("VND", "").strip()
    if not s or s == "-":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_score_row(line: str):
    """Parse one `## Scores` table row by splitting on `|`. Returns
    (symbol, model_score, news_score_str, adj_entry, adj_target) or None if the
    line isn't a data row (header, separator, prose). ``model_score`` is the
    rebound P/N score column (display/validation only — finalize uses the
    sidecar's own ``score``). Backward-compatible: a 3-column row (no adj
    columns) yields NaN for adj_entry / adj_target."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    if len(cells) < 3:
        return None
    sym = cells[0].upper()
    if not _SYM_RE.match(sym):
        return None
    try:
        model_score = float(cells[1])
    except ValueError:
        return None
    score_str = cells[2].upper()
    if score_str != "DROP" and not _INT_SCORE_RE.match(score_str):
        return None
    adj_entry = _parse_price_cell(cells[3]) if len(cells) > 3 else float("nan")
    adj_target = _parse_price_cell(cells[4]) if len(cells) > 4 else float("nan")
    return sym, model_score, score_str, adj_entry, adj_target


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
            # Stop accumulating at ANY level-2 heading — the per-ticker sections
            # only ever contain `###` headings and `**Step N**` labels, so a
            # `## ` line marks a global footer. The hybrid plan ends the ticker
            # block with `## Scores`; the LLM-only plan with `## Results` (and
            # may have other `## ` sections after the last ticker). Matching
            # only "## Scores" here let an LLM-only ticker section swallow the
            # seed-source list and the whole Results table into its findings.
            if line.strip().startswith("## "):
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
    """Read the filled markdown plan and return DataFrame[symbol, score,
    news_score, business, drivers, key_news]. news_score is `DROP_SENTINEL`
    for any row scored DROP. ``score`` is the rebound P/N model score parsed
    from the table (display/validation; finalize uses the sidecar's own)."""
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
        parsed = _parse_score_row(line)
        if parsed:
            sym, model_score, score_str, adj_entry, adj_target = parsed
            if score_str == "DROP":
                ns = DROP_SENTINEL
            else:
                ns = int(score_str)
            sec = sections.get(sym, "")
            findings = _extract_findings_list(sec)
            rows.append({
                "symbol": sym,
                "score": model_score,
                "news_score": ns,
                # News-adjusted entry / target the user/Claude wrote in the
                # score table. NaN when left blank → downstream falls back to
                # the mechanical entry_vnd / target_vnd.
                "adj_entry_vnd": adj_entry,
                "adj_target_vnd": adj_target,
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
