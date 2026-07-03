"""Generate a self-contained Gemini prompt that asks Gemini to research news for
candidate Vietnamese tickers and return a re-ranked list as JSON."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from .sources import global_urls, vn_urls


def build_prompt(candidates: pd.DataFrame, on: dt.date | None = None,
                 ab_verdict: str | None = None) -> str:
    on = on or dt.date.today()
    cfg = load_config().modes["gemini"]
    weight = cfg["news_weight"]
    # Rebound uses a flexible exit (hold until the target) — no fixed sell day
    # and no horizon parameter (see the flexible-exit note at the end).

    from .company_info import enrich
    candidates = enrich(candidates)

    from .research_dimensions import ETF_GUIDANCE_MD, REFERENCE_MD

    # Detect whether any ETF candidates are in the frame; the ETF research
    # rubric is only appended when at least one ETF row is present so the
    # stocks-only path keeps its existing prompt intact.
    has_etf_candidates = bool(
        "instrument_type" in candidates.columns
        and (candidates["instrument_type"].astype(str).str.upper() == "ETF").any()
    )

    parts: list[str] = []
    parts.append(
        f"You are a Vietnamese equities analyst VETTING rebound candidates. "
        f"Today is {on.isoformat()}. A model has narrowed the universe to "
        f"DOWNTREND names it judges statistically likely to bounce back to a "
        f"small profit (each cleared a per-ticker recovery-probability filter). "
        f"The trade: buy at today's close and HOLD until the price recovers to "
        f"the profit target (a flexible exit, typically a few days to a couple "
        f"of weeks — there is no fixed sell day). Your job is the human check "
        f"the statistics can't do: for each name, is it a healthy company in a "
        f"temporary dip that will recover, or a FALLING KNIFE (fraud, delisting, "
        f"insolvency, structural decline) where the drop is justified?"
    )
    parts.append("")
    if "missed_only" in candidates.columns or ab_verdict:
        parts.append(
            "**Two rankings to weigh.** The candidates are the UNION of the "
            "standard model's top picks and an experimental 'missed-winners' "
            "variant's top picks. In the table, `flag` = `also-missed` means BOTH "
            "models surfaced it; `missed-only` means only the experimental variant "
            "did."
            + (f" The out-of-sample A/B verdict is: {ab_verdict}." if ab_verdict else "")
            + " **Weigh the standard ranking higher** — only let a `missed-only` "
            "name into your final picks if the news strongly supports it.")
        parts.append("")
    parts.append(
        "**You decide what to research per ticker.** We do not give you a "
        "fixed checklist — different companies have different drivers, and "
        "you know better than us what matters for each one."
    )
    parts.append("")
    parts.append("**Method — per ticker:**")
    parts.append("")
    parts.append(
        "1. Identify the business from the company name + your own knowledge."
    )
    parts.append(
        "2. **Derive 3-7 research dimensions yourself** for THIS specific "
        "ticker. They may match common categories (sector, macro, policy, "
        "geopolitics, calendar) or be ticker-specific (a single major "
        "customer, a peer's earnings, a peg, a one-off contract). Skip "
        "categories that don't apply, add ones that do."
    )
    parts.append(
        "3. **Research broadly with Google search** across the dimensions "
        "you derived. Seed sources: baomoi.com/tim-kiem/<TICKER>.epi, "
        "cafef.vn, vietstock.vn, vneconomy.vn, ndh.vn, theinvestor.vn, "
        "fireant.vn, plus Reuters Asia, Bloomberg, FT, Yahoo Finance for "
        "macro, plus chinhphu.vn and sbv.gov.vn for policy / decrees / "
        "circulars. These are STARTING POINTS — search beyond them. Cross-"
        "check claims across at least 2 sources before scoring."
    )
    parts.append("")
    parts.append(REFERENCE_MD)
    parts.append("")
    if has_etf_candidates:
        parts.append(ETF_GUIDANCE_MD)
        parts.append("")
    parts.append(
        "4. **Score the rebound** -1 / 0 / +1: `+1` = news supports the bounce "
        "(recovery catalyst, or a sound company in a temporary/technical dip); "
        "`0` = nothing material, the statistical case stands; `-1` = news works "
        "AGAINST the bounce (deteriorating fundamentals, dilution, governance "
        "concern — the dip may be justified). Price/technical noise alone = 0."
    )
    parts.append("")
    parts.append(
        "**Hard override**: if a ticker is delisted / suspended / in bankruptcy "
        "/ fraud, write `DROP:` at the start of the rationale — it must not be "
        "traded no matter how attractive its score (this is exactly the falling "
        "knife the statistical filter can miss)."
    )
    parts.append("")
    parts.append(f"Adjusted score formula: adjusted = score * (1 + {weight} * news_score)")
    parts.append("")

    parts.append("## Vietnamese news sources to consult")
    for sym in candidates["symbol"].tolist():
        urls = vn_urls(sym)
        url_list = "; ".join(f"{n}: {u}" for n, u in urls.items())
        parts.append(f"- {sym}: {url_list}")
    parts.append("")

    parts.append("## Global / macro sources")
    for name, url in global_urls().items():
        parts.append(f"- {name}: {url}")
    parts.append("")
    parts.append("Also consult vnexpress.net/kinh-doanh, cafef.vn for general VN market news.")
    parts.append("")
    parts.append(
        "**Major-conflict / geopolitical check — do this ONCE, before scoring "
        "any ticker.** Scan for major global conflicts or geopolitical shocks "
        "breaking or escalating today: wars, ceasefires / peace treaties, new "
        "sanctions or tariffs, oil-supply or shipping-route disruptions, sharp "
        "oil / gold / USD-VND moves. A market-wide geopolitical catalyst can "
        "move the entire VN-Index — and specific sectors (oil & gas, shipping "
        "/ logistics, exporters, gold, fertiliser) — regardless of any single "
        "company's news. If you find one, summarise it in `global_summary` and "
        "carry it into EVERY ticker's `news_score` AND its `adj_entry_vnd` / "
        "`adj_target_vnd` (a broad risk-on melt-up means dip-limits won't fill, "
        "so raise the adjusted entry; a risk-off shock means gaps down). If "
        "today is geopolitically quiet, say so in `global_summary` and move on."
    )
    parts.append("")
    parts.append(
        f"**VN-Index trend call — do this ONCE, before scoring any ticker.** "
        "Research where the VN-Index is likely headed over the expected holding "
        "window (a few days to a couple of weeks). Look at: the index's recent trend "
        f"and momentum (last few sessions + last few weeks), where it sits vs its "
        f"50/200-day moving averages and recent support/resistance, market breadth "
        f"(are most stocks rising or is the index propped up by a few large caps — "
        f"'xanh vỏ đỏ lòng'), foreign-investor net buy/sell, liquidity, and any "
        f"scheduled macro events (SBV rates, FX, FTSE/MSCI review, big earnings). "
        f"Use cafef.vn, vietstock.vn, fialda/fireant, tradingview VNINDEX, plus the "
        f"VN-Index news search above. **State an explicit directional view in "
        f"`global_summary`: UP / SIDEWAYS / DOWN for the holding window, "
        f"with a confidence (low/med/high) and one line of reasoning.** Then let it "
        f"tilt EVERY ticker: in a likely-DOWN tape be more conservative (favour "
        f"defensive / counter-trend names, lower `news_score` for high-beta names "
        f"that just track the index, and don't chase); in a likely-UP tape be more "
        f"constructive and remember dip-limits may not fill (consider raising "
        f"`adj_entry_vnd`). A stock that usually moves OPPOSITE the index, or has a "
        f"strong company-specific catalyst, can override the index call — say so in "
        f"its rationale."
    )
    parts.append("")

    parts.append("## Candidates")
    parts.append("")
    parts.append("| symbol | type | company | score | N_days | P | recov_prob | buy_vnd | target_vnd | net_vnd | below_bar |")
    parts.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in candidates.iterrows():
        name = (r.get("organ_name") or "")[:50]
        itype = str(r.get("instrument_type", "STOCK") or "STOCK").upper()
        entry = int(r["close_vnd"]) if "close_vnd" in r and pd.notna(r.get("close_vnd")) else 0
        target = int(r["target_vnd"]) if "target_vnd" in r and pd.notna(r.get("target_vnd")) else 0
        net = int(r["net_reward_vnd"]) if "net_reward_vnd" in r and pd.notna(r.get("net_reward_vnd")) else 0
        score = r.get("score", float("nan"))
        nd = r.get("pred_days", float("nan"))
        nd_str = f"{nd:.0f}" if pd.notna(nd) else "-"
        pp = r.get("pred_profit", float("nan"))
        pp_str = f"{pp:+.3f}" if pd.notna(pp) else "-"
        rp = r.get("pred_recovery_prob", float("nan"))
        rp_str = f"{rp:.0%}" if pd.notna(rp) else "-"
        below = "yes" if r.get("below_recovery_bar", False) else "no"
        parts.append(
            f"| {r['symbol']} | {itype} | {name} | {score:.4f} | {nd_str} | "
            f"{pp_str} | {rp_str} | {entry:,} | {target:,} | {net:+,} | {below} |"
        )
    parts.append("")
    parts.append("All VND values are PER SHARE; position sizing is left to the user.")
    parts.append("`score` = P/N × recovery_prob (profit-per-day × bounce probability). "
                 "`N_days` = expected trading days to bounce; `P` = expected profit at the "
                 "bounce; `net_vnd` already accounts for ACBS round-trip fees.")
    parts.append("")

    parts.append("## Output format")
    parts.append("")
    parts.append(
        "Return ONLY valid JSON, no prose. Save your response as "
        f"`reports/gemini_response_{on.isoformat()}.json` so the program can "
        f"merge it into the final picks via `gemini-finalize`."
    )
    parts.append("")
    parts.append("Schema:")
    parts.append("```json")
    parts.append("{")
    parts.append('  "as_of": "YYYY-MM-DD",')
    parts.append('  "global_summary": "1-2 sentences on macro drivers, and your explicit VN-Index trend call: UP/SIDEWAYS/DOWN for the holding window + confidence + one-line reason",')
    parts.append('  "picks": [')
    parts.append('    {"symbol": "XYZ",')
    parts.append('     "business": "1-line description of what the company does",')
    parts.append('     "dimensions": ["the 3-7 dimensions YOU derived for this ticker"],')
    parts.append('     "drivers": ["the 2-3 most material drivers among the dimensions"],')
    parts.append('     "score": 0.0167, "news_score": 1, "adjusted": 0.0184,')
    parts.append('     "adj_entry_vnd": 13400, "adj_target_vnd": 13900,')
    parts.append('     "rationale": "1-2 sentences citing specific findings with dates",')
    parts.append('     "key_news": ["finding 1 (date, source)", "finding 2"]}')
    parts.append("  ]")
    parts.append("}")
    parts.append("```")
    parts.append("")
    parts.append(
        "**`adj_entry_vnd` / `adj_target_vnd` (optional, news-adjusted trade).** "
        "The `buy_vnd` / `target_vnd` in the candidates table are mechanical: buy "
        "at today's close, target = close × (1 + expected profit). If your "
        "research says a ticker will gap up or down on a catalyst (so the plain "
        "close-entry / target no longer fits), set `adj_entry_vnd` and "
        "`adj_target_vnd` to the entry and target you'd actually place (VND per "
        "share). These do NOT replace the mechanical prices — the program keeps "
        "both and shows the news-adjusted trade alongside. Omit both (or set them "
        "equal to the mechanical prices) when you have no price view."
    )
    parts.append("")
    parts.append("List ALL candidates below, sorted by `adjusted` descending "
                 "(drop any you judge should be excluded on the news).")
    parts.append("")
    parts.append("## Final step — flexible exit (after the JSON, in chat)")
    parts.append("")
    parts.append(
        "After the JSON, remind the user in plain text that this is a REBOUND "
        "trade with a FLEXIBLE exit: there is no fixed sell day. Each pick has a "
        "target price (`target_vnd`) and an expected hold (`N_days` to bounce). "
        "The user monitors and sells manually when the price reaches the target. "
        "So do NOT propose a hard sell-day alarm; if the user wants a nudge, "
        "suggest an optional check-in around N_days out (GMT+7) to re-examine any "
        "pick that hasn't recovered yet — 'take a look', not 'sell now'."
    )
    return "\n".join(parts)


def write_prompt(candidates: pd.DataFrame, on: dt.date | None = None,
                 ab_verdict: str | None = None) -> Path:
    on = on or dt.date.today()
    text = build_prompt(candidates, on=on, ab_verdict=ab_verdict)
    path = reports_dir() / f"gemini_prompt_{on.isoformat()}.txt"
    path.write_text(text, encoding="utf-8")
    return path
